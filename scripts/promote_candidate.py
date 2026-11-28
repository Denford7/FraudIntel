

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CONFIG_PATH = ARTIFACTS_DIR / "inference_config.json"
SCHEMA_PATH = ARTIFACTS_DIR / "inference_schema.json"
RETRAINING_DIR = ARTIFACTS_DIR / "retraining"

TRACKING_URI = "sqlite:///C:/mlflow_new/mlflow.db"
MODEL_NAME = "FraudIntel_LightGBM"

MIN_COST_IMPROVEMENT_PERCENT = 1.0
INFERENCE_SAMPLE_SIZE = 10


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path, data):
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    temporary_path.replace(path)


def prepare_features(frame, schema):
    frame = frame[schema["feature_names"]].copy()

    for column in schema["categorical_columns"]:
        categories = schema["categorical_categories"][column]

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
        col for col in schema["feature_names"]
        if col not in schema["categorical_columns"]
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise"
        )

    return frame


def verify_candidate(version, schema):
    print("\nChecking candidate inference compatibility...")

    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.lightgbm.load_model(model_uri)

    sample = pd.read_parquet(
        DATA_DIR / "validation_features.parquet"
    ).head(INFERENCE_SAMPLE_SIZE)

    sample = prepare_features(sample, schema)

    probabilities = model.predict_proba(sample)[:, 1]

    if len(probabilities) != len(sample):
        raise RuntimeError("Candidate returned an unexpected prediction count.")

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Candidate returned invalid prediction values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Candidate returned probabilities outside [0, 1].")

    print("Candidate inference: PASSED")
    print("Sample predictions:", probabilities)

    return True


def main(candidate_version, minimum_improvement):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    candidate_report_path = (
            RETRAINING_DIR
            / f"candidate_version_{candidate_version}.json"
    )

    if not candidate_report_path.exists():
        raise FileNotFoundError(candidate_report_path)

    config = load_json(CONFIG_PATH)
    schema = load_json(SCHEMA_PATH)
    candidate_report = load_json(candidate_report_path)

    evaluation_path = (
            RETRAINING_DIR
            / f"test_evaluation_v{config['model_version']}_vs_v{candidate_version}.json"
    )

    if not evaluation_path.exists():
        raise FileNotFoundError(evaluation_path)

    evaluation = load_json(evaluation_path)
    active_version = str(config["model_version"])
    reported_champion = str(evaluation["champion_version"])
    reported_candidate = str(evaluation["candidate_version"])

    print("\nFraudIntel promotion decision")
    print("-" * 45)
    print("Current active version:", active_version)
    print("Candidate version:", candidate_version)

    # Protect against evaluating or promoting against the wrong champion.
    if active_version != reported_champion:
        raise RuntimeError(
            "Evaluation champion does not match the active model. "
            "A fresh comparison is required."
        )

    if reported_candidate != str(candidate_version):
        raise RuntimeError("Evaluation report candidate version mismatch.")

    if str(candidate_report["candidate_version"]) != str(candidate_version):
        raise RuntimeError("Candidate report version mismatch.")

    if not candidate_report["accepted"]:
        raise RuntimeError(
            "Candidate failed its original validation acceptance gate."
        )

    if evaluation["threshold"] != candidate_report["threshold"]:
        raise RuntimeError("Candidate thresholds do not match.")

    if evaluation["threshold"] != float(config["threshold"]):
        raise RuntimeError(
            "Candidate threshold differs from the active threshold. "
            "Review before promotion."
        )

    minimum_recall = float(config["minimum_recall_requirement"])

    champion = evaluation["champion"]
    candidate = evaluation["candidate"]

    champion_cost = float(champion["cost"])
    candidate_cost = float(candidate["cost"])

    if champion_cost <= 0:
        raise RuntimeError("Champion cost must be greater than zero.")

    cost_reduction = champion_cost - candidate_cost

    improvement_percent = (
        cost_reduction / champion_cost
    ) * 100

    recall_pass = (
        float(candidate["recall"]) >= minimum_recall
    )

    cost_pass = (
        improvement_percent >= minimum_improvement
    )

    test_report_recall_pass = bool(
        evaluation["candidate_recall_requirement_met"]
    )

    print("\nIndependent test results")
    print("Champion cost:", champion_cost)
    print("Candidate cost:", candidate_cost)
    print("Cost reduction:", cost_reduction)
    print("Cost improvement (%):", round(improvement_percent, 4))
    print("Candidate recall:", round(float(candidate["recall"]), 5))
    print("Minimum recall:", minimum_recall)

    print("\nChecking promotion requirements...")

    print("Validation acceptance:", "PASS")
    print("Recall requirement:", "PASS" if recall_pass else "FAIL")
    print("Test recall check:", "PASS" if test_report_recall_pass else "FAIL")
    print(
        "Minimum cost improvement:",
        "PASS" if cost_pass else "FAIL"
    )

    inference_pass = verify_candidate(candidate_version, schema)

    approved = (
        recall_pass
        and test_report_recall_pass
        and cost_pass
        and inference_pass
    )

    reasons = []

    if not recall_pass:
        reasons.append("Candidate failed minimum recall requirement.")

    if not test_report_recall_pass:
        reasons.append("Independent test recall requirement failed.")

    if not cost_pass:
        reasons.append(
            f"Cost improvement of {improvement_percent:.4f}% "
            f"is below the required {minimum_improvement:.2f}%."
        )

    if approved:
        decision = "PROMOTED"
        print("\nPROMOTION APPROVED.")

        backup_path = CONFIG_PATH.with_suffix(".json.bak")
        shutil.copy2(CONFIG_PATH, backup_path)

        config["model_version"] = int(candidate_version)
        config["threshold"] = float(candidate["threshold"])

        save_json_atomic(CONFIG_PATH, config)

        print("Active configuration updated.")
        print("Previous configuration backup:", backup_path)

    else:
        decision = "REJECTED"
        print("\nCANDIDATE REJECTED.")

        for reason in reasons:
            print("-", reason)

        print("Active champion remains version", active_version)

    decision_report = {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "previous_champion_version": active_version,
        "candidate_version": str(candidate_version),
        "decision": decision,
        "promoted": approved,
        "minimum_cost_improvement_percent": minimum_improvement,
        "actual_cost_improvement_percent": improvement_percent,
        "minimum_recall": minimum_recall,
        "candidate_recall": float(candidate["recall"]),
        "champion_test_cost": champion_cost,
        "candidate_test_cost": candidate_cost,
        "inference_verification": inference_pass,
        "reasons": reasons,
        "active_version_after_decision": (
            str(candidate_version) if approved else active_version
        )
    }

    decision_path = (
        RETRAINING_DIR
        / f"decision_candidate_version_{candidate_version}.json"
    )

    save_json_atomic(decision_path, decision_report)

    client.set_model_version_tag(
        MODEL_NAME,
        str(candidate_version),
        "promotion_decision",
        decision.lower()
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(candidate_version),
        "promotion_reason",
        "; ".join(reasons) if reasons else "Passed all promotion gates."
    )

    print("\nDecision:", decision)
    print("Decision report:", decision_path)
    print(
        "Active model version:",
        decision_report["active_version_after_decision"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FraudIntel candidate promotion and rejection gate"
    )

    parser.add_argument(
        "--candidate-version",
        default="2",
        help="Registered MLflow candidate version."
    )

    parser.add_argument(
        "--min-cost-improvement",
        type=float,
        default=MIN_COST_IMPROVEMENT_PERCENT,
        help="Minimum test cost improvement percentage."
    )

    args = parser.parse_args()

    if args.min_cost_improvement < 0:
        parser.error("Minimum cost improvement cannot be negative.")

    main(
        candidate_version=args.candidate_version,
        minimum_improvement=args.min_cost_improvement
    )

