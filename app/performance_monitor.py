
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


class PerformanceMonitor:
    """Monitor chronological model performance and detect concept drift."""

    def __init__(
        self,
        archive_dir: Path,
        window_size: int = 1000,
        min_fraud_count: int = 20,
        significance_level: float = 0.05,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.window_size = window_size
        self.min_fraud_count = min_fraud_count
        self.significance_level = significance_level

        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        if min_fraud_count < 1:
            raise ValueError("min_fraud_count must be at least 1.")

        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> pd.DataFrame:
        """Load labelled records from the latest simulation."""

        records: list[dict[str, Any]] = []

        for path in self.archive_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    record = json.loads(line)

                    if (
                        record.get("actual_label") is not None
                        and record.get("prediction") is not None
                    ):
                        records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        df["actual_label"] = pd.to_numeric(
            df["actual_label"], errors="coerce"
        )
        df["prediction"] = pd.to_numeric(
            df["prediction"], errors="coerce"
        )

        df = df.dropna(subset=["actual_label", "prediction"])

        df = df[
            df["actual_label"].isin([0, 1])
            & df["prediction"].isin([0, 1])
        ].copy()

        if df.empty:
            return df

        # Use only the latest simulation to avoid mixing old random
        # simulations with the current chronological replay.
        if "simulation_id" in df.columns and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            )

            latest_simulation = (
                df.groupby("simulation_id")["timestamp"]
                .max()
                .idxmax()
            )

            df = df[df["simulation_id"] == latest_simulation].copy()

        # TransactionDT represents the original transaction event time.
        if "TransactionDT" not in df.columns:
            raise ValueError(
                "Archived records are missing TransactionDT."
            )

        df["TransactionDT"] = pd.to_numeric(
            df["TransactionDT"], errors="coerce"
        )

        df = df.dropna(subset=["TransactionDT"])
        df = df.sort_values("TransactionDT", kind="stable")

        return df.reset_index(drop=True)

    def calculate_metrics(self, records: pd.DataFrame) -> dict[str, Any]:
        """Calculate classification metrics for one window."""

        y_true = records["actual_label"].astype(int)
        y_pred = records["prediction"].astype(int)

        true_positives = int(
            ((y_true == 1) & (y_pred == 1)).sum()
        )

        false_negatives = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        fraud_count = int((y_true == 1).sum())

        false_negative_rate = (
            false_negatives / fraud_count
            if fraud_count > 0
            else None
        )

        return {
            "transactions": len(records),
            "fraud_count": fraud_count,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "precision": round(
                precision_score(y_true, y_pred, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, zero_division=0), 4
            ),
            "f1": round(
                f1_score(y_true, y_pred, zero_division=0), 4
            ),
            "false_negative_rate": (
                round(false_negative_rate, 4)
                if false_negative_rate is not None
                else None
            ),
        }

    def detect_concept_drift(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Test whether recall has significantly deteriorated."""

        baseline_fraud = baseline["fraud_count"]
        current_fraud = current["fraud_count"]

        result = {
            "drift_alert": False,
            "p_value": None,
            "baseline_recall": baseline["recall"],
            "current_recall": current["recall"],
            "drift_status": "insufficient_data",
        }

        if (
            baseline_fraud < self.min_fraud_count
            or current_fraud < self.min_fraud_count
        ):
            return result

        # Rows: baseline and current.
        # Columns: correctly detected fraud and missed fraud.
        contingency_table = [
            [
                baseline["true_positives"],
                baseline["false_negatives"],
            ],
            [
                current["true_positives"],
                current["false_negatives"],
            ],
        ]

        test_result = fisher_exact(
            contingency_table,
            alternative="greater",
        )

        p_value = float(test_result.pvalue)

        drift_alert = (
            current["recall"] < baseline["recall"]
            and p_value < self.significance_level
        )

        result.update({
            "drift_alert": drift_alert,
            "p_value": round(p_value, 6),
            "drift_status": (
                "concept_drift_detected"
                if drift_alert
                else "no_significant_drift"
            ),
        })

        return result

    def evaluate_rolling_windows(self) -> list[dict[str, Any]]:
        """Compare successive chronological windows against a baseline."""

        records = self.load_records()

        if len(records) < self.window_size:
            return []

        results = []
        baseline = None

        for start in range(0, len(records), self.window_size):
            window = records.iloc[
                start:start + self.window_size
            ]

            if len(window) < self.window_size:
                break

            metrics = self.calculate_metrics(window)

            metrics["window"] = len(results) + 1
            metrics["start_transaction_dt"] = float(
                window["TransactionDT"].min()
            )
            metrics["end_transaction_dt"] = float(
                window["TransactionDT"].max()
            )

            if baseline is None:
                baseline = metrics.copy()

                metrics["drift_status"] = "baseline"
                metrics["drift_alert"] = False
                metrics["p_value"] = None

            else:
                drift = self.detect_concept_drift(
                    baseline,
                    metrics,
                )

                metrics.update(drift)

            results.append(metrics)

        return results
