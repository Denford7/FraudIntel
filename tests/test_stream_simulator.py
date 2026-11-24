
import json
from urllib.error import URLError

import pandas as pd
import pytest

import app.stream_simulator as simulator


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "TransactionAmt": [100.0, 250.0, 75.0],
        "V1": [0.1, 0.2, 0.3],
    })


@pytest.fixture
def configure_stream(tmp_path, monkeypatch, sample_transactions):
    events_dir = tmp_path / "stream"
    events_file = events_dir / "transaction_events.jsonl"
    data_path = tmp_path / "validation_features.parquet"

    data_path.touch()

    monkeypatch.setattr(simulator, "DATA_PATH", data_path)
    monkeypatch.setattr(simulator, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(simulator, "EVENTS_FILE", events_file)

    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda path: sample_transactions,
    )

    saved_events = []
    database_initializations = []

    monkeypatch.setattr(
        simulator,
        "initialize_database",
        lambda: database_initializations.append(True),
    )

    monkeypatch.setattr(
        simulator,
        "save_event",
        lambda event: saved_events.append(event),
    )

    monkeypatch.setattr(simulator.time, "sleep", lambda seconds: None)

    return {
        "events_file": events_file,
        "saved_events": saved_events,
        "database_initializations": database_initializations,
    }


def successful_prediction(transaction):
    return {
        "fraud_probability": 0.85,
        "prediction": 1,
        "risk_level": "HIGH",
        "threshold": 0.05,
    }


def test_send_transaction_sends_valid_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"prediction": 1}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(simulator, "urlopen", fake_urlopen)

    transaction = {"TransactionAmt": 100.0}

    result = simulator.send_transaction(transaction)

    assert result == {"prediction": 1}
    assert captured["request"].full_url == simulator.API_URL
    assert captured["request"].method == "POST"
    assert captured["request"].get_header("Content-type") == "application/json"
    assert captured["request"].get_header("X-fintech-id") == "default"
    assert json.loads(captured["request"].data.decode("utf-8")) == transaction
    assert captured["timeout"] == 30


def test_run_stream_processes_and_persists_transactions(
    configure_stream,
    monkeypatch,
):
    monkeypatch.setattr(
        simulator,
        "send_transaction",
        successful_prediction,
    )

    simulator.run_stream(max_transactions=2, delay_seconds=0)

    saved_events = configure_stream["saved_events"]

    assert len(saved_events) == 2
    assert len(configure_stream["database_initializations"]) == 1

    assert saved_events[0]["transaction_id"] == 1
    assert saved_events[1]["transaction_id"] == 2

    assert saved_events[0]["prediction"] == 1
    assert saved_events[0]["risk_level"] == "HIGH"
    assert saved_events[0]["threshold"] == 0.05

    assert saved_events[0]["transaction_amount"] == 100.0
    assert saved_events[1]["transaction_amount"] == 250.0

    assert all(event["run_id"] == saved_events[0]["run_id"]
               for event in saved_events)

    assert configure_stream["events_file"].exists()

    with open(configure_stream["events_file"], encoding="utf-8") as file:
        persisted_events = [json.loads(line) for line in file]

    assert len(persisted_events) == 2
    assert persisted_events[0]["prediction"] == 1


def test_run_stream_rejects_invalid_transaction_limit():
    with pytest.raises(ValueError, match="max_transactions must be positive"):
        simulator.run_stream(max_transactions=0)


def test_run_stream_rejects_negative_delay():
    with pytest.raises(ValueError, match="delay_seconds cannot be negative"):
        simulator.run_stream(delay_seconds=-1)


def test_run_stream_rejects_missing_dataset(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.parquet"

    monkeypatch.setattr(simulator, "DATA_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Validation Parquet file not found"):
        simulator.run_stream(max_transactions=1)


def test_run_stream_rejects_empty_dataset(
    configure_stream,
    monkeypatch,
):
    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda path: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="contains no transactions"):
        simulator.run_stream(max_transactions=1)


def test_run_stream_stops_after_api_failure(
    configure_stream,
    monkeypatch,
):
    def failed_request(transaction):
        raise URLError("API unavailable")

    monkeypatch.setattr(
        simulator,
        "send_transaction",
        failed_request,
    )

    simulator.run_stream(max_transactions=3, delay_seconds=0)

    assert configure_stream["saved_events"] == []

    with open(configure_stream["events_file"], encoding="utf-8") as file:
        assert file.read() == ""


def test_run_stream_handles_manual_interruption(
    configure_stream,
    monkeypatch,
):
    calls = {"count": 0}

    def interrupted_request(transaction):
        calls["count"] += 1

        if calls["count"] == 3:
            raise KeyboardInterrupt

        return successful_prediction(transaction)

    monkeypatch.setattr(
        simulator,
        "send_transaction",
        interrupted_request,
    )

    simulator.run_stream(max_transactions=None, delay_seconds=0)

    assert calls["count"] == 3
    assert len(configure_stream["saved_events"]) == 2
