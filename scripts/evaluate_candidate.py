
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SCHEMA_PATH = ARTIFACTS_DIR / "inference_schema.json"
CONFIG_PATH = ARTIFACTS_DIR / "inference_config.json"
RETRAINING_DIR = ARTIFACTS_DIR / "retraining"

TRACKING_URI = "sqlite:///C:/mlflow_new/mlflow.db"
MODEL_NAME = "FraudIntel_LightGBM"


def prepare_features(frame, schema):

    expected_features = schema["feature_names"]

    missing = [
        column for column in expected_features
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features: {missing[:10]}"
        )

    frame = frame[expected_features].copy()

    for column in schema["categorical_columns"]:
        categories = schema["categorical_categories"][column]

        if "__UNKNOWN__" not in categories:
            raise ValueError(
                f"Missing __UNKNOWN__ category for {column}."
            )

        values = frame[column].astype("string")

        if "__MISSING__" in categories:
            values = values.fillna("__MISSING__")

        values = values.where(
            values.isna() | values.isin(categories),
            "__UNKNOWN__"
        )

        frame[column] = pd.Categorical(
            values,
            categories=categories
        )

    numeric_columns = [
        col for col in expected_features
        if col not in schema["categorical_columns"]
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise"
        )

    return frame


def evaluate(
    y_true,
    probabilities,
    threshold,
    missed_cost,
    false_cost
):

    predictions = (probabilities >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    cost = fn * missed_cost + fp * false_cost

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": cost,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def main(candidate_version):

    print("\nLoading evaluation configuration...")

    mlflow.set_tracking_uri(TRACKING_URI)

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    candidate_version = str(candidate_version)

    candidate_report_path = (
        RETRAINING_DIR / f"candidate_version_{candidate_version}.json"
    )

    if not candidate_report_path.exists():
        raise FileNotFoundError(candidate_report_path)

    with open(candidate_report_path, encoding="utf-8") as f:
        candidate_report = json.load(f)

    if str(candidate_report["candidate_version"]) != candidate_version:
        raise ValueError("Candidate version does not match its report.")

    champion_version = str(config["model_version"])

    if str(candidate_report["active_version_before_run"]) != champion_version:
        raise ValueError(
            "Active champion changed since candidate training. "
            "Stop and review the model configuration."
        )

    if not candidate_report["accepted"]:
        raise RuntimeError(
            "Candidate failed its original validation acceptance gate."
        )

    missed_cost = float(config["cost_per_missed_fraud"])
    false_cost = float(config["cost_per_false_alarm"])
    minimum_recall = float(config["minimum_recall_requirement"])

    threshold = float(candidate_report["threshold"])

    if threshold != float(config["threshold"]):
        raise ValueError(
            "Candidate threshold differs from the active threshold."
        )

    print("Champion version:", champion_version)
    print("Candidate version:", candidate_version)
    print("Evaluation threshold:", threshold)

    print("\nLoading untouched test dataset...")

    X_test = pd.read_parquet(
        DATA_DIR / "test_features.parquet"
    )

    test_target = pd.read_parquet(
        DATA_DIR / "test_target.parquet"
    )

    y_test = test_target["isFraud"].astype(int).to_numpy()

    if len(X_test) != len(y_test):
        raise ValueError("Test feature/target row counts differ.")

    if len(X_test) == 0:
        raise ValueError("Test dataset is empty.")

    if set(np.unique(y_test)) - {0, 1}:
        raise ValueError("Test target must be binary.")

    if len(schema["feature_names"]) != 432:
        raise ValueError("Expected 432 features in inference schema.")

    X_test = prepare_features(X_test, schema)

    print("Test records:", len(X_test))
    print("Test fraud rate:", round(float(y_test.mean()), 5))
    print("Features:", X_test.shape[1])

    results = {}

    for version in [champion_version, candidate_version]:

        print(f"\nEvaluating MLflow model version {version}...")

        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = mlflow.lightgbm.load_model(model_uri)

        probabilities = model.predict_proba(X_test)[:, 1]

        results[version] = evaluate(
            y_test,
            probabilities,
            threshold,
            missed_cost,
            false_cost
        )

        print("Precision:", round(results[version]["precision"], 5))
        print("Recall:", round(results[version]["recall"], 5))
        print("F1:", round(results[version]["f1"], 5))
        print("Cost:", results[version]["cost"])

    champion = results[champion_version]
    candidate = results[candidate_version]

    cost_difference = champion["cost"] - candidate["cost"]

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "champion_version": champion_version,
        "candidate_version": candidate_version,
        "test_records": len(X_test),
        "test_fraud_rate": float(y_test.mean()),
        "threshold": threshold,
        "minimum_recall": minimum_recall,
        "cost_assumptions": {
            "missed_fraud": missed_cost,
            "false_alarm": false_cost
        },
        "champion": champion,
        "candidate": candidate,
        "candidate_recall_requirement_met": (
            candidate["recall"] >= minimum_recall
        ),
        "cost_reduction": cost_difference,
        "cost_reduction_percent": (
            cost_difference / champion["cost"] * 100
            if champion["cost"] else 0.0
        ),
        "active_model_changed": False,
        "promotion_performed": False
    }

    output_path = (
        RETRAINING_DIR
        / f"test_evaluation_v{champion_version}_vs_v{candidate_version}.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 48)
    print("INDEPENDENT TEST EVALUATION COMPLETE")
    print("=" * 48)

    print("Champion cost:", champion["cost"])
    print("Candidate cost:", candidate["cost"])
    print("Cost reduction:", cost_difference)

    print(
        "Cost reduction (%):",
        round(report["cost_reduction_percent"], 4)
    )

    print(
        "Candidate recall requirement:",
        report["candidate_recall_requirement_met"]
    )

    print("\nActive model remains version", champion_version)
    print("Promotion performed: False")
    print("Report:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FraudIntel independent candidate evaluation"
    )

    parser.add_argument(
        "--candidate-version",
        required=True,
        help="Registered MLflow candidate version."
    )

    args = parser.parse_args()

    main(candidate_version=args.candidate_version)
