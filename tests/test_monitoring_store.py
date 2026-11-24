
from datetime import datetime, timezone

import pytest

from app import monitoring_store


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    """Use an isolated DuckDB database for each test."""

    monitoring_dir = tmp_path / "monitoring"
    database_path = monitoring_dir / "fraudintel.duckdb"

    monkeypatch.setattr(
        monitoring_store, "MONITORING_DIR", monitoring_dir
    )
    monkeypatch.setattr(
        monitoring_store, "DB_PATH", database_path
    )

    monitoring_store.initialize_database()

    return database_path


def sample_event(event_id=1, run_id="run_001", prediction=1,
                 probability=0.08, risk_level="HIGH",
                 timestamp=None):

    return {
        "event_id": event_id,
        "run_id": run_id,
        "transaction_id": 1000 + event_id,
        "timestamp": timestamp or datetime.now(timezone.utc),
        "transaction_amount": 250.0,
        "fraud_probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": 0.05,
    }


def test_initialize_database_creates_database_and_table(
    temporary_database,
):
    assert temporary_database.exists()

    import duckdb

    with duckdb.connect(
        str(temporary_database), read_only=True
    ) as connection:
        tables = connection.execute(
            "SHOW TABLES"
        ).fetchall()

    assert ("transaction_events",) in tables


def test_get_database_stats_returns_stored_statistics(
    temporary_database,
):
    monitoring_store.save_event(sample_event(1, "run_001", 1))
    monitoring_store.save_event(
        sample_event(
            2, "run_001", 0, 0.01, "LOW"
        )
    )
    monitoring_store.save_event(
        sample_event(
            3, "run_002", 1, 0.07, "HIGH"
        )
    )

    result = monitoring_store.get_database_stats()

    assert result.iloc[0]["total_transactions"] == 3
    assert result.iloc[0]["fraud_alerts"] == 2
    assert result.iloc[0]["simulation_runs"] == 2
    assert result.iloc[0]["average_probability"] == 0.0533


def test_get_database_stats_handles_empty_database(
    temporary_database,
):
    result = monitoring_store.get_database_stats()

    assert result.iloc[0]["total_transactions"] == 0
    assert result.iloc[0]["fraud_alerts"] is None or (
        result.iloc[0]["fraud_alerts"] != result.iloc[0]["fraud_alerts"]
    )
    assert result.iloc[0]["simulation_runs"] == 0


def test_get_hourly_stats_groups_transactions_by_hour(
    temporary_database,
):
    monitoring_store.save_event(
        sample_event(
            1,
            probability=0.08,
            timestamp=datetime(
                2026, 11, 18, 10, 15, tzinfo=timezone.utc
            ),
        )
    )

    monitoring_store.save_event(
        sample_event(
            2,
            prediction=0,
            probability=0.02,
            risk_level="LOW",
            timestamp=datetime(
                2026, 11, 18, 10, 45, tzinfo=timezone.utc
            ),
        )
    )

    monitoring_store.save_event(
        sample_event(
            3,
            probability=0.09,
            timestamp=datetime(
                2026, 11, 18, 11, 10, tzinfo=timezone.utc
            ),
        )
    )

    results = monitoring_store.get_hourly_stats()

    assert len(results) == 2

    latest = results[0]
    assert latest["total_transactions"] == 1
    assert latest["fraud_alerts"] == 1
    assert latest["avg_probability"] == 0.09

    previous = results[1]
    assert previous["total_transactions"] == 2
    assert previous["fraud_alerts"] == 1
    assert previous["avg_probability"] == 0.05


def test_get_hourly_stats_respects_limit(
    temporary_database,
):
    for hour in range(10, 14):
        monitoring_store.save_event(
            sample_event(
                event_id=hour,
                timestamp=datetime(
                    2026, 11, 18, hour, 15,
                    tzinfo=timezone.utc,
                ),
            )
        )

    results = monitoring_store.get_hourly_stats(limit=2)

    assert len(results) == 2
