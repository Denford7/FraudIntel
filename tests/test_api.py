from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path
import app.main as main
from app.performance_monitor import PerformanceMonitor
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "validation_features.parquet"

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "FraudIntel_LightGBM"
    assert data["model_version"] == 1

def test_predict_endpoint():
    response = client.get("/schema")

    assert response.status_code == 200

    schema = response.json()

    assert schema["model"] == "FraudIntel_LightGBM"
    assert schema["model_version"] == 1
    assert schema["feature_count"] == 432
    assert "required_features" in schema
    assert len(schema["required_features"]) == 432

import pandas as pd


def test_predict_single_transaction():
    df = pd.read_parquet(DATA_PATH)

    transaction = (
        df.iloc[0]
        .astype(object)
        .where(pd.notna(df.iloc[0]), None)
        .to_dict()
    )

    response = client.post(
        "/predict",
        json=transaction,
        headers={"X-Fintech-ID": "default"},
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "prediction" in result
    assert "risk_level" in result

def test_explain_single_transaction():
    df = pd.read_parquet(DATA_PATH)

    transaction = (
        df.iloc[1]
        .astype(object)
        .where(pd.notna(df.iloc[1]), None)
        .to_dict()
    )

    response = client.post(
        "/explain",
        json=transaction,
        headers={"X-Fintech-ID": "default"},
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "top_contributions" in result
    assert "analyst_explanation" in result
    assert len(result["top_contributions"]) > 0

def test_predict_rejects_unknown_fintech():
    response = client.post(
        "/predict",
        json={},
        headers={"X-Fintech-ID": "unknown_fintech"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown fintech integration."

def test_predict_rejects_invalid_transaction():
    response = client.post(
        "/predict",
        json={"invalid_field": "not_a_transaction"},
        headers={"X-Fintech-ID": "default"},
    )

    assert response.status_code == 422

def test_predict_batch():
    df = pd.read_parquet(DATA_PATH)

    transactions = []

    for i in range(3):
        row = df.iloc[i]

        transaction = (
            row.astype(object)
            .where(pd.notna(row), None)
            .to_dict()
        )

        transactions.append(transaction)

    response = client.post(
        "/predict-batch",
        json=transactions,
        headers={"X-Fintech-ID": "default"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["total_transactions"] == 3
    assert len(result["results"]) == 3

    assert "fraud_detected" in result
    assert "legitimate_transactions" in result
    assert "fraud_detection_rate" in result

    for prediction in result["results"]:
        assert "fraud_probability" in prediction
        assert "prediction" in prediction
        assert "risk_level" in prediction


def test_predict_csv():
    df = pd.read_parquet(DATA_PATH)

    csv_data = df.head(3).to_csv(index=False)

    response = client.post(
        "/predict-csv",
        files={
            "file": (
                "transactions.csv",
                csv_data,
                "text/csv",
            )
        },
        headers={"X-Fintech-ID": "default"},
    )

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

    result_df = pd.read_csv(
        pd.io.common.StringIO(response.text)
    )

    assert len(result_df) == 3

    expected_columns = [
        "fraud_probability",
        "prediction",
        "is_fraud",
        "risk_level",
        "threshold",
    ]

    for column in expected_columns:
        assert column in result_df.columns


def test_monitoring_summary():
    response = client.get("/monitoring/summary")

    assert response.status_code == 200

    result = response.json()

    assert "total_transactions" in result
    assert "fraud_alerts" in result
    assert "legitimate_transactions" in result
    assert "average_fraud_probability" in result

    assert result["total_transactions"] >= 0
    assert result["fraud_alerts"] >= 0
    assert result["legitimate_transactions"] >= 0

    assert (
        result["fraud_alerts"] + result["legitimate_transactions"]
        == result["total_transactions"]
    )

    assert 0 <= result["average_fraud_probability"] <= 1

def test_monitoring_recent():
    response = client.get("/monitoring/recent?limit=5")

    assert response.status_code == 200

    result = response.json()

    assert result["limit"] == 5
    assert "events" in result
    assert isinstance(result["events"], list)
    assert len(result["events"]) <= 5

    required_fields = {
        "event_id",
        "run_id",
        "transaction_id",
        "timestamp",
        "transaction_amount",
        "fraud_probability",
        "prediction",
        "risk_level",
        "threshold",
    }

    for event in result["events"]:
        assert required_fields.issubset(event.keys())

        assert event["event_id"] is not None
        assert event["run_id"]
        assert event["transaction_id"] is not None

        assert 0 <= event["fraud_probability"] <= 1
        assert event["prediction"] in (0, 1)
        assert event["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert 0 <= event["threshold"] <= 1

def test_monitoring_alerts():
    response = client.get("/monitoring/alerts?limit=5")

    assert response.status_code == 200

    result = response.json()

    assert "alerts" in result
    assert "total_returned" in result

    alerts = result["alerts"]

    assert isinstance(alerts, list)
    assert result["total_returned"] == len(alerts)
    assert len(alerts) <= 5

    required_fields = {
        "event_id",
        "run_id",
        "transaction_id",
        "timestamp",
        "transaction_amount",
        "fraud_probability",
        "risk_level",
    }

    for alert in alerts:
        assert required_fields.issubset(alert.keys())

        assert alert["event_id"] is not None
        assert alert["run_id"]
        assert alert["transaction_id"] is not None
        assert 0 <= alert["fraud_probability"] <= 1
        assert alert["risk_level"] in ("LOW", "MEDIUM", "HIGH")

def test_monitoring_risk_distribution():
    response = client.get("/monitoring/risk-distribution")

    assert response.status_code == 200

    result = response.json()

    assert "risk_distribution" in result

    distribution = result["risk_distribution"]

    assert isinstance(distribution, list)

    expected_order = ["HIGH", "MEDIUM", "LOW"]
    actual_order = [
        item["risk_level"] for item in distribution
    ]

    assert actual_order == sorted(
        actual_order,
        key=expected_order.index,
    )

    for item in distribution:
        assert set(item.keys()) == {
            "risk_level",
            "transactions",
            "fraud_alerts",
            "avg_probability",
        }

        assert item["risk_level"] in expected_order
        assert item["transactions"] > 0
        assert 0 <= item["fraud_alerts"] <= item["transactions"]
        assert 0 <= item["avg_probability"] <= 1

def test_monitoring_hourly():
    response = client.get("/monitoring/hourly?limit=5")

    assert response.status_code == 200

    result = response.json()

    assert "hourly_statistics" in result

    statistics = result["hourly_statistics"]

    assert isinstance(statistics, list)
    assert len(statistics) <= 5

    from datetime import datetime

    for item in statistics:
        assert set(item.keys()) == {
            "hour",
            "total_transactions",
            "fraud_alerts",
            "avg_probability",
        }

        # Verify that the hour is a valid ISO timestamp.
        parsed_hour = datetime.fromisoformat(item["hour"])
        assert parsed_hour.minute == 0
        assert parsed_hour.second == 0

        assert item["total_transactions"] > 0
        assert 0 <= item["fraud_alerts"] <= item["total_transactions"]
        assert 0 <= item["avg_probability"] <= 1

    # Newest hour must appear first.
    hours = [
        datetime.fromisoformat(item["hour"])
        for item in statistics
    ]

    assert hours == sorted(hours, reverse=True)

def test_prometheus_metrics():
    # Generate a request that the middleware can record.
    health_response = client.get("/health")
    assert health_response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    metrics = response.text

    expected_metrics = [
        "fraudintel_http_requests_total",
        "fraudintel_http_request_duration_seconds",
        "fraudintel_predictions_total",
        "fraudintel_prediction_errors_total",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing Prometheus metric: {metric}"

    # Verify that a previously recorded request is exposed.
    assert 'endpoint="/health"' in metrics
    assert 'method="GET"' in metrics

def test_risk_level_uses_configured_threshold():
    from app.predictor import FraudPredictor

    predictor = FraudPredictor()

    # Current champion threshold: 0.05
    assert predictor._risk_level(0.05) == "HIGH"
    assert predictor._risk_level(0.04) == "MEDIUM"
    assert predictor._risk_level(0.025) == "MEDIUM"
    assert predictor._risk_level(0.02) == "LOW"

    # Verify that risk levels adapt to a changed threshold.
    predictor.threshold = 0.10

    assert predictor._risk_level(0.10) == "HIGH"
    assert predictor._risk_level(0.075) == "MEDIUM"
    assert predictor._risk_level(0.04) == "LOW"


def test_concept_drift_evaluation_saves_results(monkeypatch):
    windows = [
        {
            "window": 1,
            "recall": 0.75,
            "drift_alert": False,
            "drift_status": "baseline",
        },
        {
            "window": 2,
            "recall": 0.60,
            "drift_alert": True,
            "drift_status": "concept_drift_detected",
        },
    ]

    class MockMonitor:
        def __init__(self, **kwargs):
            pass

        def evaluate_rolling_windows(self):
            return windows

    monkeypatch.setattr(main, "PerformanceMonitor", MockMonitor)
    monkeypatch.setattr(
        main,
        "save_performance_evaluations",
        lambda windows, window_size: "test-evaluation-id",
    )

    response = client.post("/monitoring/concept-drift/evaluate")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "concept_drift_detected"
    assert result["saved"] is True
    assert result["evaluation_id"] == "test-evaluation-id"
    assert result["windows_saved"] == 2
    assert result["latest_window"]["drift_alert"] is True


def test_concept_drift_evaluation_handles_insufficient_data(
    monkeypatch,
):
    class MockMonitor:
        def __init__(self, **kwargs):
            pass

        def evaluate_rolling_windows(self):
            return []

    monkeypatch.setattr(main, "PerformanceMonitor", MockMonitor)

    response = client.post("/monitoring/concept-drift/evaluate")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "insufficient_data"
    assert result["saved"] is False


def test_concept_drift_history_returns_saved_evaluations(
    monkeypatch,
):
    expected = [
        {
            "evaluation_id": "test-evaluation-id",
            "window_number": 2,
            "recall": 0.60,
            "drift_alert": True,
            "drift_status": "concept_drift_detected",
        }
    ]

    monkeypatch.setattr(
        main,
        "get_performance_history",
        lambda limit: expected,
    )

    response = client.get("/monitoring/concept-drift/history?limit=10")

    assert response.status_code == 200

    result = response.json()

    assert result["limit"] == 10
    assert result["evaluations"] == expected
