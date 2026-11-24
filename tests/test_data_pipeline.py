
import subprocess
import json

from scripts import run_data_pipeline


def test_pipeline_runs_all_eight_stages_in_order(monkeypatch):
    executed_commands = []

    def fake_run(command, cwd, check):
        executed_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_data_pipeline.main()

    expected_scripts = [
        "scripts/download_data.py",
        "scripts/validate_raw_data.py",
        "scripts/ingest_data.py",
        "scripts/validate_data_integrity.py",
        "scripts/verify_parquet_integrity.py",
        "scripts/verify_parquet_values.py",
    ]

    assert result == 0
    assert len(executed_commands) == 8

    for command, expected_script in zip(
        executed_commands[:6], expected_scripts
    ):
        assert command[1] == expected_script

    assert executed_commands[6] == [
        run_data_pipeline.sys.executable,
        "-m",
        "pytest",
        "-v",
    ]

    assert executed_commands[7] == [
        run_data_pipeline.sys.executable,
        "-m",
        "src.fraudintel.monitoring.data_quality_report",
    ]


def test_pipeline_stops_when_a_stage_fails(monkeypatch):
    executed_commands = []

    def fake_run(command, cwd, check):
        executed_commands.append(command)

        if command[1] == "scripts/validate_raw_data.py":
            return subprocess.CompletedProcess(command, 2)

        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_data_pipeline.main()

    assert result == 2
    assert len(executed_commands) == 2
    assert executed_commands[1][1] == "scripts/validate_raw_data.py"


def test_pipeline_detects_a_missing_script(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_data_pipeline,
        "BASE_DIR",
        tmp_path,
    )

    result = run_data_pipeline.main()

    assert result == 1


def test_pipeline_summary_records_actual_results(tmp_path, monkeypatch):
    summary_path = tmp_path / "reports" / "pipeline_run_summary.json"
    monkeypatch.setattr(
        run_data_pipeline,
        "SUMMARY_PATH",
        summary_path,
    )

    started_at = "2026-10-10T12:00:00+00:00"
    stage_results = [
        {
            "stage": "Raw-data validation",
            "status": "passed",
            "exit_code": 0,
        },
        {
            "stage": "Data-integrity validation",
            "status": "failed",
            "exit_code": 2,
        },
    ]

    run_data_pipeline.write_pipeline_summary(
        started_at=started_at,
        stage_results=stage_results,
        overall_status="failed",
        exit_code=2,
    )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["project"] == "FraudIntel"
    assert summary["started_at_utc"] == started_at
    assert summary["overall_status"] == "failed"
    assert summary["exit_code"] == 2
    assert summary["stages_completed"] == 2
    assert summary["stages"] == stage_results
    assert summary["finished_at_utc"]


def test_pipeline_summary_creates_parent_directory(tmp_path, monkeypatch):
    summary_path = (
        tmp_path / "new_reports" / "nested" / "summary.json"
    )
    monkeypatch.setattr(
        run_data_pipeline,
        "SUMMARY_PATH",
        summary_path,
    )

    run_data_pipeline.write_pipeline_summary(
        started_at="2026-10-10T12:00:00+00:00",
        stage_results=[],
        overall_status="passed",
        exit_code=0,
    )

    assert summary_path.is_file()
