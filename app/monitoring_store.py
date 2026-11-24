
from pathlib import Path
import duckdb
import uuid
from datetime import datetime, timezone


BASE_DIR = Path(__file__).resolve().parent.parent
MONITORING_DIR = BASE_DIR / "data" / "monitoring"
DB_PATH = MONITORING_DIR / "fraudintel.duckdb"



def _create_performance_table(connection):
    """Create the concept-drift performance history table."""

    connection.execute("""
        CREATE TABLE IF NOT EXISTS performance_evaluations (
            evaluation_id VARCHAR NOT NULL,
            evaluated_at TIMESTAMPTZ NOT NULL,
            window_size INTEGER NOT NULL,
            window_number INTEGER NOT NULL,
            transactions BIGINT,
            fraud_count BIGINT,
            true_positives BIGINT,
            false_negatives BIGINT,
            precision DOUBLE,
            recall DOUBLE,
            f1 DOUBLE,
            false_negative_rate DOUBLE,
            start_transaction_dt DOUBLE,
            end_transaction_dt DOUBLE,
            drift_status VARCHAR,
            drift_alert BOOLEAN,
            p_value DOUBLE,
            baseline_recall DOUBLE,
            current_recall DOUBLE
        )
    """)



def initialize_database():
    """Create the monitoring database and events table."""

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_PATH)) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transaction_events (
                event_id BIGINT PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                transaction_id BIGINT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                transaction_amount DOUBLE,
                fraud_probability DOUBLE NOT NULL,
                prediction INTEGER NOT NULL,
                risk_level VARCHAR NOT NULL,
                threshold DOUBLE NOT NULL
            )
        """)
        _create_performance_table(connection)

    print(f"Monitoring database initialized: {DB_PATH}")


def save_event(event: dict):
    """Persist one scored transaction."""

    with duckdb.connect(str(DB_PATH)) as connection:
        connection.execute("""
            INSERT INTO transaction_events (
                event_id,
                run_id,
                transaction_id,
                timestamp,
                transaction_amount,
                fraud_probability,
                prediction,
                risk_level,
                threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event["event_id"],
            event["run_id"],
            event["transaction_id"],
            event["timestamp"],
            event.get("transaction_amount"),
            event["fraud_probability"],
            event["prediction"],
            event["risk_level"],
            event["threshold"],
        ])

def save_events(events: list[dict]):
    """Save multiple prediction events using one DuckDB connection."""

    if not events:
        return

    with duckdb.connect(str(DB_PATH)) as connection:
        connection.executemany(
            """
            INSERT INTO transaction_events (
                event_id,
                run_id,
                transaction_id,
                timestamp,
                transaction_amount,
                fraud_probability,
                prediction,
                risk_level,
                threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event["event_id"],
                    event["run_id"],
                    event["transaction_id"],
                    event["timestamp"],
                    event.get("transaction_amount"),
                    event["fraud_probability"],
                    event["prediction"],
                    event["risk_level"],
                    event["threshold"],
                )
                for event in events
            ],
        )


def get_summary():
    """Return overall transaction monitoring statistics."""

    with duckdb.connect(str(DB_PATH)) as connection:
        result = connection.execute("""
            SELECT
                COUNT(*) AS total_transactions,
                COALESCE(SUM(prediction), 0) AS fraud_alerts,
                COUNT(*) - COALESCE(SUM(prediction), 0)
                    AS legitimate_transactions,
                ROUND(AVG(fraud_probability), 4)
                    AS average_fraud_probability
            FROM transaction_events
        """).fetchone()

    return {
        "total_transactions": result[0],
        "fraud_alerts": result[1],
        "legitimate_transactions": result[2],
        "average_fraud_probability": result[3] or 0.0,
    }


def get_recent_events(limit: int = 20):
    """Retrieve the most recent scored transactions."""

    with duckdb.connect(str(DB_PATH)) as connection:
        rows = connection.execute("""
            SELECT
                event_id,
                run_id,
                transaction_id,
                timestamp,
                transaction_amount,
                fraud_probability,
                prediction,
                risk_level,
                threshold
            FROM transaction_events
            ORDER BY timestamp DESC, event_id DESC
            LIMIT ?
        """, [limit]).fetchall()

        columns = [
            "event_id",
            "run_id",
            "transaction_id",
            "timestamp",
            "transaction_amount",
            "fraud_probability",
            "prediction",
            "risk_level",
            "threshold",
        ]

    return [dict(zip(columns, row)) for row in rows]

def get_risk_distribution():
    """Return transaction statistics grouped by risk level."""
    with duckdb.connect(str(DB_PATH)) as connection:
        rows = connection.execute("""
            SELECT
                risk_level,
                COUNT(*) AS transactions,
                COALESCE(SUM(prediction), 0) AS fraud_alerts,
                ROUND(AVG(fraud_probability), 4) AS avg_probability
            FROM transaction_events
            GROUP BY risk_level
            ORDER BY
                CASE risk_level
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    WHEN 'LOW' THEN 3
                    ELSE 4
                END
        """).fetchall()

    return [
        {
            "risk_level": row[0],
            "transactions": row[1],
            "fraud_alerts": row[2],
            "avg_probability": row[3] or 0.0,
        }
        for row in rows
    ]

def get_database_stats():
    """Inspect the stored monitoring events."""

    with duckdb.connect(str(DB_PATH))as connection:
        return connection.execute("""
            SELECT
                COUNT(*) AS total_transactions,
                SUM(prediction) AS fraud_alerts,
                COUNT(DISTINCT run_id) AS simulation_runs,
                ROUND(AVG(fraud_probability), 4) AS average_probability
            FROM transaction_events
        """).fetchdf()

def get_hourly_stats(limit: int = 24):
    """Return transaction and fraud statistics grouped by hour."""

    with duckdb.connect(str(DB_PATH)) as connection:
        rows = connection.execute("""
            SELECT
                date_trunc('hour', timestamp) AS hour,
                COUNT(*) AS total_transactions,
                COALESCE(SUM(prediction), 0) AS fraud_alerts,
                ROUND(AVG(fraud_probability), 4) AS avg_probability
            FROM transaction_events
            GROUP BY 1
            ORDER BY hour DESC
            LIMIT ?
        """, [limit]).fetchall()

    return [
        {
            "hour": row[0].isoformat(),
            "total_transactions": row[1],
            "fraud_alerts": row[2],
            "avg_probability": row[3] or 0.0,
        }
        for row in rows
    ]


def save_performance_evaluations(
    windows: list[dict],
    window_size: int,
) -> str | None:
    """Persist one complete concept-drift evaluation."""

    if not windows:
        return None

    evaluation_id = str(uuid.uuid4())
    evaluated_at = datetime.now(timezone.utc)

    with duckdb.connect(str(DB_PATH)) as connection:
        _create_performance_table(connection)

        connection.begin()

        try:
            connection.executemany(
                """
                INSERT INTO performance_evaluations VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    (
                        evaluation_id,
                        evaluated_at,
                        window_size,
                        window["window"],
                        window["transactions"],
                        window["fraud_count"],
                        window["true_positives"],
                        window["false_negatives"],
                        window["precision"],
                        window["recall"],
                        window["f1"],
                        window["false_negative_rate"],
                        window["start_transaction_dt"],
                        window["end_transaction_dt"],
                        window["drift_status"],
                        window["drift_alert"],
                        window.get("p_value"),
                        window.get("baseline_recall"),
                        window.get("current_recall"),
                    )
                    for window in windows
                ],
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

    return evaluation_id

def get_performance_history(limit: int = 50) -> list[dict]:
    """Retrieve saved concept-drift evaluation windows."""

    with duckdb.connect(str(DB_PATH)) as connection:
        rows = connection.execute(
            """
            SELECT
                evaluation_id,
                CAST(evaluated_at AS VARCHAR) AS evaluated_at,
                window_size,
                window_number,
                transactions,
                fraud_count,
                true_positives,
                false_negatives,
                precision,
                recall,
                f1,
                false_negative_rate,
                start_transaction_dt,
                end_transaction_dt,
                drift_status,
                drift_alert,
                p_value,
                baseline_recall,
                current_recall
            FROM performance_evaluations
            ORDER BY evaluated_at DESC, window_number DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        columns = [
            "evaluation_id",
            "evaluated_at",
            "window_size",
            "window_number",
            "transactions",
            "fraud_count",
            "true_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "false_negative_rate",
            "start_transaction_dt",
            "end_transaction_dt",
            "drift_status",
            "drift_alert",
            "p_value",
            "baseline_recall",
            "current_recall",
        ]

    return [dict(zip(columns, row)) for row in rows]
