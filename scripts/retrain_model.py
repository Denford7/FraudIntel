import argparse
import json
import math
import inspect
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

import joblib
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CONFIG_PATH = ARTIFACTS_DIR / "inference_config.json"
SCHEMA_PATH = ARTIFACTS_DIR / "inference_schema.json"

OUTPUT_DIR = ARTIFACTS_DIR / "retraining"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_URI = "sqlite:///C:/mlflow_new/mlflow.db"
MODEL_NAME = "FraudIntel_LightGBM"

THRESHOLDS = [
    0.010, 0.015, 0.020, 0.025, 0.030,
    0.035, 0.040, 0.045, 0.050
]

RANDOM_STATE = 42


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_data():
    print("\n[1/7] Loading processed datasets...")

    X_train = pd.read_parquet(
        DATA_DIR / "train_features.parquet"
    )

    X_val = pd.read_parquet(
        DATA_DIR / "validation_features.parquet"
    )

    train_target = pd.read_parquet(
        DATA_DIR / "train_target.parquet"
    )

    val_target = pd.read_parquet(
        DATA_DIR / "validation_target.parquet"
    )

    y_train = train_target["isFraud"].astype(int)
    y_val = val_target["isFraud"].astype(int)

    if len(X_train) != len(y_train):
        raise ValueError("Training feature/target row counts differ.")

    if len(X_val) != len(y_val):
        raise ValueError("Validation feature/target row counts differ.")

    if set(y_train.unique()) - {0, 1}:
        raise ValueError("Training target must be binary.")

    if set(y_val.unique()) - {0, 1}:
        raise ValueError("Validation target must be binary.")

    with open(SCHEMA_PATH, encoding="utf-8") as file:
        schema = json.load(file)

    expected_features = schema["feature_names"]
    categorical_columns = schema["categorical_columns"]

    if len(expected_features) != 432:
        raise ValueError(
            f"Expected 432 features, found {len(expected_features)}."
        )

    if len(categorical_columns) != 31:
        raise ValueError(
            f"Expected 31 categorical features, "
            f"found {len(categorical_columns)}."
        )

    for name, frame in [
        ("training", X_train),
        ("validation", X_val)
    ]:
        missing = [
            col for col in expected_features
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{name} data is missing features: {missing[:10]}"
            )

    X_train = X_train[expected_features].copy()
    X_val = X_val[expected_features].copy()

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Training fraud rate:", round(float(y_train.mean()), 5))
    print("Validation fraud rate:", round(float(y_val.mean()), 5))

    return X_train, X_val, y_train, y_val, schema


# ---------------------------------------------------------
# INFERENCE-COMPATIBLE PREPROCESSING
# ---------------------------------------------------------

def prepare_features(frame, schema):
    prepared = frame[schema["feature_names"]].copy()

    categorical_columns = schema["categorical_columns"]
    category_map = schema["categorical_categories"]

    for column in categorical_columns:
        categories = category_map[column]

        if "__UNKNOWN__" not in categories:
            raise ValueError(
                f"{column} has no __UNKNOWN__ category."
            )

        values = prepared[column].astype("string")

        if "__MISSING__" in categories:
            values = values.fillna("__MISSING__")

        values = values.where(
            values.isna() | values.isin(categories),
            "__UNKNOWN__"
        )

        prepared[column] = pd.Categorical(
            values,
            categories=categories
        )

    numeric_columns = [
        col for col in schema["feature_names"]
        if col not in categorical_columns
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="raise"
        )

    return prepared


# ---------------------------------------------------------
# THRESHOLD EVALUATION
# ---------------------------------------------------------

def evaluate_thresholds(y_true, probabilities, thresholds,
                        missed_fraud_cost, false_alarm_cost):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    results = []

    for threshold in thresholds:
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

        cost = (
            fn * missed_fraud_cost
            + fp * false_alarm_cost
        )

        results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "cost": float(cost),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    return results


def choose_threshold(results, minimum_recall):
    eligible = [
        row for row in results
        if row["recall"] >= minimum_recall
    ]

    if not eligible:
        return None

    return min(
        eligible,
        key=lambda row: (
            row["cost"],
            -row["recall"],
            -row["f1"]
        )
    )


# ---------------------------------------------------------
# MAIN RETRAINING WORKFLOW
# ---------------------------------------------------------

def main(promote=False):

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    with open(CONFIG_PATH, encoding="utf-8") as file:
        active_config = json.load(file)

    with open(SCHEMA_PATH, encoding="utf-8") as file:
        schema = json.load(file)

    minimum_recall = float(
        active_config["minimum_recall_requirement"]
    )

    missed_fraud_cost = float(
        active_config["cost_per_missed_fraud"]
    )

    false_alarm_cost = float(
        active_config["cost_per_false_alarm"]
    )

    active_version = str(active_config["model_version"])
    active_threshold = float(active_config["threshold"])

    X_train, X_val, y_train, y_val, schema = load_data()

    print("\n[2/7] Applying inference-compatible preprocessing...")

    X_train = prepare_features(X_train, schema)
    X_val = prepare_features(X_val, schema)

    categorical_columns = schema["categorical_columns"]

    print("Preprocessing complete.")
    print("Categorical columns:", len(categorical_columns))

    # -----------------------------------------------------
    # LOAD CURRENT CHAMPION
    # -----------------------------------------------------

    print("\n[3/7] Evaluating current champion...")

    champion_uri = (
        f"models:/{MODEL_NAME}/{active_version}"
    )

    champion = mlflow.lightgbm.load_model(champion_uri)

    champion_probabilities = champion.predict_proba(X_val)[:, 1]

    champion_results = evaluate_thresholds(
        y_val,
        champion_probabilities,
        [active_threshold],
        missed_fraud_cost,
        false_alarm_cost
    )[0]

    print("Champion version:", active_version)
    print("Champion threshold:", active_threshold)
    print("Champion recall:", round(champion_results["recall"], 4))
    print("Champion cost:", champion_results["cost"])

    # -----------------------------------------------------
    # TRAIN CANDIDATE
    # -----------------------------------------------------

    print("\n[4/7] Training full-feature LightGBM candidate...")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_columns,
        callbacks=[
            early_stopping(30, verbose=False),
            log_evaluation(50)
        ]
    )

    candidate_probabilities = model.predict_proba(X_val)[:, 1]

    candidate_results = evaluate_thresholds(
        y_val,
        candidate_probabilities,
        THRESHOLDS,
        missed_fraud_cost,
        false_alarm_cost
    )

    selected = choose_threshold(
        candidate_results,
        minimum_recall
    )

    if selected is None:
        raise RuntimeError(
            "Candidate failed: no threshold meets minimum recall."
        )

    print("\nCandidate threshold:", selected["threshold"])
    print("Candidate precision:", round(selected["precision"], 4))
    print("Candidate recall:", round(selected["recall"], 4))
    print("Candidate F1:", round(selected["f1"], 4))
    print("Candidate cost:", selected["cost"])

    # -----------------------------------------------------
    # ACCEPTANCE GATE
    # -----------------------------------------------------

    better_cost = (
        selected["cost"] < champion_results["cost"]
    )

    equal_cost_better_f1 = (
        math.isclose(
            selected["cost"],
            champion_results["cost"],
            rel_tol=0,
            abs_tol=1e-9
        )
        and selected["f1"] > champion_results["f1"]
    )

    accepted = (
        selected["recall"] >= minimum_recall
        and (better_cost or equal_cost_better_f1)
    )

    print("\nAcceptance result:", "PASS" if accepted else "FAIL")

    # -----------------------------------------------------
    # LOG AND REGISTER CANDIDATE
    # -----------------------------------------------------

    print("\n[5/7] Logging candidate to MLflow...")

    experiment_name = "FraudIntel_Retraining"

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="full_feature_retraining"
    ) as run:

        mlflow.set_tags({
            "model_name": MODEL_NAME,
            "model_role": "candidate",
            "simulation": "true",
            "accepted": str(accepted).lower()
        })

        mlflow.log_params({
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "feature_count": len(schema["feature_names"]),
            "categorical_count": len(categorical_columns),
            "minimum_recall": minimum_recall,
            "missed_fraud_cost": missed_fraud_cost,
            "false_alarm_cost": false_alarm_cost
        })

        mlflow.log_metrics({
            "candidate_precision": selected["precision"],
            "candidate_recall": selected["recall"],
            "candidate_f1": selected["f1"],
            "candidate_cost": selected["cost"],
            "candidate_threshold": selected["threshold"],
            "champion_precision": champion_results["precision"],
            "champion_recall": champion_results["recall"],
            "champion_f1": champion_results["f1"],
            "champion_cost": champion_results["cost"],
            "champion_threshold": active_threshold
        })

        signature = infer_signature(
            X_val.head(5),
            model.predict_proba(X_val.head(5))[:, 1]
        )

        log_params = inspect.signature(
            mlflow.lightgbm.log_model
        ).parameters

        model_args = {
            "lgb_model": model,
            "input_example": X_val.head(5),
            "signature": signature
        }

        if "name" in log_params:
            model_args["name"] = "model"
        else:
            model_args["artifact_path"] = "model"

        model_info = mlflow.lightgbm.log_model(**model_args)

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME
        )

        candidate_version = str(registered.version)

        print("Registered candidate version:", candidate_version)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "candidate_version": candidate_version,
            "active_version_before_run": active_version,
            "accepted": accepted,
            "promoted": False,
            "threshold": selected["threshold"],
            "candidate": selected,
            "champion": champion_results,
            "minimum_recall": minimum_recall,
            "cost_assumptions": {
                "missed_fraud": missed_fraud_cost,
                "false_alarm": false_alarm_cost
            },
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
            "feature_count": len(schema["feature_names"]),
            "categorical_count": len(categorical_columns),
            "run_id": run.info.run_id
        }

        report_path = (
            OUTPUT_DIR
            / f"candidate_version_{candidate_version}.json"
        )

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluation"
        )

    # -----------------------------------------------------
    # OPTIONAL PROMOTION
    # -----------------------------------------------------

    print("\n[6/7] Promotion check...")

    if promote and accepted:

        backup_path = CONFIG_PATH.with_suffix(".json.bak")
        copy2(CONFIG_PATH, backup_path)

        active_config["model_version"] = int(candidate_version)
        active_config["threshold"] = float(selected["threshold"])

        temporary_path = CONFIG_PATH.with_suffix(".json.tmp")

        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(active_config, file, indent=4)

        temporary_path.replace(CONFIG_PATH)

        report["promoted"] = True

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("Candidate promoted successfully.")
        print("Previous config backup:", backup_path)

    elif promote and not accepted:
        print("Promotion blocked: candidate failed acceptance.")

    else:
        print("Candidate registered but not promoted.")
        print("Active champion remains unchanged.")

    print("\n[7/7] Retraining workflow complete.")

    print("Candidate version:", candidate_version)
    print("Accepted:", accepted)
    print("Promoted:", report["promoted"])
    print("Evaluation report:", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FraudIntel full-feature LightGBM retraining"
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote candidate only if it passes evaluation."
    )

    args = parser.parse_args()

    main(promote=args.promote)
