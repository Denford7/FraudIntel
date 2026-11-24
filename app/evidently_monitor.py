
import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd


class EvidentlyMonitor:
    """Persist simulated predictions and generate drift reports."""

    def __init__(
        self,
        reference_path: Path,
        archive_dir: Path,
        report_dir: Path,
        batch_size: int = 100,
    ) -> None:
        self.reference_path = Path(reference_path)
        self.archive_dir = Path(archive_dir)
        self.report_dir = Path(report_dir)
        self.batch_size = batch_size

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._lock = Lock()
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._batch_numbers: dict[str, int] = {}

        self.reference = pd.read_parquet(self.reference_path)

        if self.reference.empty:
            raise ValueError("The reference dataset is empty.")

        self.raw_dir = self.archive_dir / "raw"
        self.batch_dir = self.archive_dir / "batches"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_simulation_id(simulation_id: str) -> str:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", simulation_id)

        if not safe_id:
            raise ValueError("Invalid simulation ID.")

        return safe_id

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert values recursively into valid JSON-compatible values."""

        if isinstance(value, dict):
            return {
                str(key): EvidentlyMonitor._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                EvidentlyMonitor._json_safe(item)
                for item in value
            ]

        if hasattr(value, "item"):
            value = value.item()

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(value, float):
            import math

            if not math.isfinite(value):
                return None

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)
    def _raw_path(self, simulation_id: str) -> Path:
        return self.raw_dir / f"simulation_{simulation_id}.jsonl"

    def _simulation_batch_dir(self, simulation_id: str) -> Path:
        path = self.batch_dir / f"simulation_{simulation_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_transaction(
        self,
        transaction: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Persist a successful prediction before buffering it."""

        simulation_id = self._safe_simulation_id(
            str(metadata["simulation_id"])
        )

        record = {
            **transaction,
            **metadata,
            "simulation_id": simulation_id,
        }
        safe_record = self._json_safe(record)

        serialized = json.dumps(
            safe_record,
            allow_nan=False,
        )

        with self._lock:
            raw_path = self._raw_path(simulation_id)

            with raw_path.open("a", encoding="utf-8") as file:
                file.write(serialized + "\n")
                file.flush()
                os.fsync(file.fileno())

            buffer = self._buffers.setdefault(simulation_id, [])
            buffer.append(safe_record)

            if len(buffer) >= self.batch_size:
                self._flush_locked(simulation_id)

    def _flush_locked(self, simulation_id: str) -> None:
        """Export buffered records to Parquet."""

        buffer = self._buffers.get(simulation_id, [])

        if not buffer:
            return

        batch_number = self._batch_numbers.get(simulation_id, 0) + 1
        batch_name = f"batch_{batch_number:06d}.parquet"

        batch_dir = self._simulation_batch_dir(simulation_id)
        output_path = batch_dir / batch_name
        temporary_path = batch_dir / f"{batch_name}.tmp"

        batch = pd.DataFrame(buffer)

        try:
            batch.to_parquet(
                temporary_path,
                index=False,
                engine="pyarrow",
            )
            temporary_path.replace(output_path)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        self._buffers[simulation_id] = []
        self._batch_numbers[simulation_id] = batch_number

        print(
            f"Monitoring archive saved: {output_path} "
            f"({len(batch)} records)"
        )

        try:
            self.generate_report(
                batch=batch,
                simulation_id=simulation_id,
                batch_number=batch_number,
            )
        except Exception as exc:
            print(f"Evidently report generation failed: {exc}")

    def flush_simulation(self, simulation_id: str) -> None:
        """Save the final partial batch."""

        safe_id = self._safe_simulation_id(simulation_id)

        with self._lock:
            self._flush_locked(safe_id)

    def generate_report(
        self,
        batch: pd.DataFrame,
        simulation_id: str,
        batch_number: int,
    ) -> None:
        """Generate a drift report from archived feature data."""

        from evidently import Report
        from evidently.presets import DataDriftPreset

        feature_columns = [
            column
            for column in self.reference.columns
            if column in batch.columns
        ]

        if not feature_columns:
            print("Evidently: no reference features found.")
            return

        reference = self.reference[feature_columns].copy()
        current = batch[feature_columns].copy()

        reference = reference.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )
        current = current.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        usable_columns = [
            column
            for column in feature_columns
            if current[column].notna().any()
            and reference[column].notna().any()
        ]

        if not usable_columns:
            print("Evidently: no usable features for drift analysis.")
            return

        reference = reference[usable_columns].copy()
        current = current[usable_columns].copy()

        for column in usable_columns:
            if pd.api.types.is_numeric_dtype(reference[column]):
                current[column] = pd.to_numeric(
                    current[column],
                    errors="coerce",
                )
            else:
                current[column] = current[column].astype("string")
            # Exclude constant features from this drift comparison.
            # Keep all original records untouched in the archive.
        variable_columns = [
            column
            for column in usable_columns
            if reference[column].nunique(dropna=True) > 1
               and current[column].nunique(dropna=True) > 1
        ]

        if not variable_columns:
            print(
                "Evidently: insufficient feature variation "
                "for drift analysis."
            )
            return

        reference = reference[variable_columns].copy()
        current = current[variable_columns].copy()

        reference_sample = reference.sample(
            n=min(10_000, len(reference)),
            random_state=42,
        )

        report = Report(metrics=[DataDriftPreset()])

        snapshot = report.run(
            current_data=current,
            reference_data=reference_sample,
        )

        output_dir = self.report_dir / f"simulation_{simulation_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"drift_report_{batch_number:06d}.html"
        )

        snapshot.save_html(str(output_path))

        print(f"Evidently report generated: {output_path}")
