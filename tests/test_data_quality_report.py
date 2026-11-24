
import json

import pandas as pd
import pytest

from src.fraudintel.monitoring.data_quality_report import (
    profile_csv,
    write_data_quality_report,
    build_data_quality_report,
    load_pipeline_summary,
)




def test_profile_csv_calculates_missing_values(tmp_path, monkeypatch):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "amount": [10.0, None, 30.0, None],
            "category": ["A", "B", None, "A"],
        }
    ).to_csv(csv_path, index=False)

    result = profile_csv(csv_path)

    assert result["file_name"] == "sample.csv"
    assert result["row_count"] == 4
    assert result["column_count"] == 3

    assert result["columns"]["TransactionID"] == {
        "missing_count": 0,
        "missing_percentage": 0.0,
    }
    assert result["columns"]["amount"] == {
        "missing_count": 2,
        "missing_percentage": 50.0,
    }
    assert result["columns"]["category"] == {
        "missing_count": 1,
        "missing_percentage": 25.0,
    }


def test_profile_csv_handles_header_only_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("TransactionID,amount\n", encoding="utf-8")

    result = profile_csv(csv_path)

    assert result["row_count"] == 0
    assert result["column_count"] == 2
    assert result["columns"]["TransactionID"]["missing_count"] == 0
    assert result["columns"]["amount"]["missing_percentage"] == 0.0


def test_profile_csv_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        profile_csv(tmp_path / "missing.csv")


def test_write_report_creates_valid_json(tmp_path):
    report = {
        "project": "FraudIntel",
        "report_status": "profile_completed",
        "datasets": {},
    }
    output_path = tmp_path / "reports" / "report.json"

    returned_path = write_data_quality_report(report, output_path)

    assert returned_path == output_path
    assert output_path.is_file()

    saved_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_report == report


def test_load_pipeline_summary_reads_valid_summary(tmp_path):
    summary_path = tmp_path / "pipeline_run_summary.json"
    summary = {
        "project": "FraudIntel",
        "overall_status": "passed",
        "exit_code": 0,
        "stages_completed": 8,
        "stages_expected": 8,
        "stages": [
            {
                "stage": "Raw-data validation",
                "status": "passed",
                "exit_code": 0,
            }
        ],
    }
    summary_path.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    result = load_pipeline_summary(summary_path)

    assert result == summary


def test_load_pipeline_summary_returns_unavailable_when_missing(
    tmp_path,
):
    result = load_pipeline_summary(
        tmp_path / "missing_summary.json"
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "summary_not_found"


def test_load_pipeline_summary_handles_invalid_json(tmp_path):
    summary_path = tmp_path / "invalid_summary.json"
    summary_path.write_text("{invalid json", encoding="utf-8")

    result = load_pipeline_summary(summary_path)

    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid_json"

def test_load_pipeline_summary_reads_valid_summary(tmp_path):
    summary_path = tmp_path / "pipeline_run_summary.json"
    summary = {
        "project": "FraudIntel",
        "overall_status": "passed",
        "exit_code": 0,
        "stages_completed": 8,
        "stages_expected": 8,
        "stages": [
            {
                "stage": "Raw-data validation",
                "status": "passed",
                "exit_code": 0,
            }
        ],
    }
    summary_path.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    result = load_pipeline_summary(summary_path)

    assert result == summary


def test_load_pipeline_summary_returns_unavailable_when_missing(
    tmp_path,
):
    result = load_pipeline_summary(
        tmp_path / "missing_summary.json"
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "summary_not_found"


def test_load_pipeline_summary_handles_invalid_json(tmp_path):
    summary_path = tmp_path / "invalid_summary.json"
    summary_path.write_text("{invalid json", encoding="utf-8")

    result = load_pipeline_summary(summary_path)

    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid_json"

def test_report_marks_current_pipeline_summary_unavailable():
    report = build_data_quality_report()

    assert "pipeline_summary" in report
    assert report["pipeline_summary"]["status"] == "unavailable"
    assert (
        report["pipeline_summary"]["reason"]
        == "current_run_summary_not_available_until_pipeline_finishes"
    )