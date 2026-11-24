from datetime import datetime, timezone
import json
from pathlib import Path
import logging
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fraudintel.monitoring.logging_config import configure_logging
from src.fraudintel.monitoring.data_quality_report import (
    load_pipeline_summary,
    write_data_quality_report,
)

SUMMARY_PATH = BASE_DIR / "reports" / "pipeline_run_summary.json"

STAGES = [
    (
        "Kaggle data acquisition",
        [sys.executable, "scripts/download_data.py"],
    ),
    (
        "Raw-data validation",
        [sys.executable, "scripts/validate_raw_data.py"],
    ),
    (
        "CSV-to-Parquet ingestion",
        [sys.executable, "scripts/ingest_data.py"],
    ),
    (
        "Data-integrity validation",
        [sys.executable, "scripts/validate_data_integrity.py"],
    ),
    (
        "CSV vs Parquet structural verification",
        [sys.executable, "scripts/verify_parquet_integrity.py"],
    ),
    (
        "CSV vs Parquet value verification",
        [sys.executable, "scripts/verify_parquet_values.py"],
    ),
    (
        "Automated pytest suite",
        [sys.executable, "-m", "pytest", "-v"],
    ),
    (
        "JSON data-quality report",
        [
            sys.executable,
            "-m",
            "src.fraudintel.monitoring.data_quality_report",
        ],
    ),
]


def write_pipeline_summary(
    started_at: str,
    stage_results: list[dict],
    overall_status: str,
    exit_code: int,
) -> None:
    """Save actual pipeline execution results as JSON."""
    summary = {
        "project": "FraudIntel",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "exit_code": exit_code,
        "stages_expected": len(STAGES),
        "stages_completed": len(stage_results),
        "stages": stage_results,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
def update_data_quality_report() -> None:
    """Attach the completed pipeline summary to the data-quality report."""
    report_path = BASE_DIR / "reports" / "data_quality_report.json"

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Data-quality report not found: {report_path}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = load_pipeline_summary(SUMMARY_PATH)

    if summary.get("overall_status") not in {"passed", "failed"}:
        raise ValueError(
            "Pipeline summary does not contain a valid final status."
        )

    report["pipeline_summary"] = summary
    write_data_quality_report(report, report_path)

def main() -> int:
    log_file = configure_logging()
    logger = logging.getLogger("fraudintel.pipeline")
    started_at = datetime.now(timezone.utc).isoformat()
    stage_results = []

    logger.info("FraudIntel data pipeline started.")
    logger.info("Log file: %s", log_file)

    print("=" * 65)
    print("FraudIntel - End-to-End Data Pipeline")
    print("=" * 65)

    for stage_number, (stage_name, command) in enumerate(
        STAGES, start=1
    ):
        logger.info(
            "Starting stage %s/%s: %s",
            stage_number,
            len(STAGES),
            stage_name,
        )

        print(f"\n{'-' * 65}")
        print(f"STAGE {stage_number}/{len(STAGES)}: {stage_name}")
        print(f"{'-' * 65}")

        if command[1].startswith("scripts/"):
            script_path = BASE_DIR / command[1]

            if not script_path.is_file():
                logger.error("Script not found: %s", script_path)
                print(f"ERROR: Script not found: {script_path}")

                stage_results.append({
                    "stage": stage_name,
                    "status": "failed",
                    "exit_code": 1,
                })
                write_pipeline_summary(
                    started_at, stage_results, "failed", 1
                )
                return 1

        try:
            result = subprocess.run(
                command,
                cwd=BASE_DIR,
                check=False,
            )
            exit_code = result.returncode
        except OSError:
            logger.exception(
                "Could not execute stage: %s", stage_name
            )
            stage_results.append({
                "stage": stage_name,
                "status": "failed",
                "exit_code": 1,
            })
            write_pipeline_summary(
                started_at, stage_results, "failed", 1
            )
            return 1

        status = "passed" if exit_code == 0 else "failed"
        stage_results.append({
            "stage": stage_name,
            "status": status,
            "exit_code": exit_code,
        })

        if exit_code != 0:
            logger.error(
                "Stage failed: %s (exit code %s)",
                stage_name,
                exit_code,
            )
            print(f"\nPIPELINE FAILED at stage: {stage_name}")
            print(f"Exit code: {exit_code}")

            write_pipeline_summary(
                started_at, stage_results, "failed", exit_code
            )
            return exit_code

        logger.info("Stage passed: %s", stage_name)
        print(f"\nSTAGE PASSED: {stage_name}")

    logger.info("All %s pipeline stages passed.", len(STAGES))

    write_pipeline_summary(
        started_at, stage_results, "passed", 0
    )
    logger.info("Pipeline summary written to: %s", SUMMARY_PATH)

    try:
        update_data_quality_report()
        logger.info("Data-quality report updated with final pipeline summary.")
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("Failed to update data-quality report.")
        return 1

    print("\n" + "=" * 65)
    print("ALL DATA PIPELINE STAGES COMPLETED SUCCESSFULLY.")
    print(f"Stages passed: {len(stage_results)}/{len(STAGES)}")
    print(f"Summary: {SUMMARY_PATH}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
