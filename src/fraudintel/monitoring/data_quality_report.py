
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
REPORT_DIR = BASE_DIR / "reports"

DATASETS = {
    "train_transaction.csv": RAW_DIR / "train_transaction.csv",
    "train_identity.csv": RAW_DIR / "train_identity.csv",
    "test_transaction.csv": RAW_DIR / "test_transaction.csv",
    "test_identity.csv": RAW_DIR / "test_identity.csv",
}

CHUNK_SIZE = 100_000


def load_pipeline_summary(
    summary_path: Path | None = None,
) -> dict:
    """Load a pipeline summary, or report why it is unavailable."""
    if summary_path is None:
        summary_path = REPORT_DIR / "pipeline_run_summary.json"

    if not summary_path.is_file():
        return {
            "status": "unavailable",
            "reason": "summary_not_found",
        }

    try:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "status": "unavailable",
            "reason": "invalid_json",
        }

    if not isinstance(summary, dict):
        return {
            "status": "unavailable",
            "reason": "invalid_summary_structure",
        }

    return summary


def profile_csv(path: Path) -> dict:
    """Profile CSV dimensions and missing values using chunks."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    row_count = 0
    column_names = None
    missing_counts = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if column_names is None:
            column_names = chunk.columns.tolist()
            missing_counts = pd.Series(
                0, index=column_names, dtype="int64"
            )

        row_count += len(chunk)
        missing_counts += chunk.isna().sum()

    if column_names is None:
        # Handles a header-only CSV.
        column_names = pd.read_csv(path, nrows=0).columns.tolist()
        missing_counts = pd.Series(
            0, index=column_names, dtype="int64"
        )

    columns = {}
    for column in column_names:
        missing = int(missing_counts[column])
        columns[column] = {
            "missing_count": missing,
            "missing_percentage": (
                round(missing / row_count * 100, 4)
                if row_count
                else 0.0
            ),
        }

    return {
        "file_name": path.name,
        "row_count": row_count,
        "column_count": len(column_names),
        "columns": columns,
    }

def build_data_quality_report() -> dict:
    """Build dataset profiles and include the available pipeline summary."""
    datasets = {}

    for name, path in DATASETS.items():
        datasets[name] = profile_csv(path)



    # The runner writes the final summary after stage 8 completes.
    # Therefore, a summary already on disk may belong to an older run.
    pipeline_summary = {
        "status": "unavailable",
        "reason": "current_run_summary_not_available_until_pipeline_finishes",
    }

    return {
        "project": "FraudIntel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_status": "profile_completed",
        "datasets": datasets,
        "pipeline_summary": pipeline_summary,
    }

def write_data_quality_report(
    report: dict,
    output_path: Path | None = None,
) -> Path:
    """Write the report as formatted JSON."""
    if output_path is None:
        output_path = REPORT_DIR / "data_quality_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    report = build_data_quality_report()
    output_path = write_data_quality_report(report)

    print(f"Data-quality report written to: {output_path}")
    print(f"Datasets profiled: {len(report['datasets'])}")
    print(f"Report status: {report['report_status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
