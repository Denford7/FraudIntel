
import json
import time
import pandas as pd
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone
import re
import subprocess
import sys
import threading

import duckdb
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    Response,
    StreamingResponse,
    FileResponse,
)
# RETRAINING WORKFLOW STATE
BASE_DIR = Path(__file__).resolve().parent.parent
RETRAINING_DIR = BASE_DIR / "artifacts" / "retraining"

retraining_lock = threading.Lock()

retraining_status = {
    "status": "idle",
    "stage": None,
    "progress": 0,
    "candidate_version": None,
    "message": "No retraining workflow running.",
    "logs": [],
    "started_at": None,
    "completed_at": None,
    "promoted": False,
    "active_version": None,
}

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.stream_simulator import simulation_controller
from app.performance_monitor import PerformanceMonitor
from app.integrations import FintechAdapter
from starlette.concurrency import run_in_threadpool
from app.metrics import (
    PREDICTION_ERRORS,
    PREDICTIONS_TOTAL,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from app.monitoring_store import (
    DB_PATH,
    get_hourly_stats,
    get_recent_events,
    get_risk_distribution,
    get_summary,
    save_event,
    save_events,
    get_performance_history,
    save_performance_evaluations,
)
from app.predictor import FraudPredictor


app = FastAPI(
    title="FraudIntel API",
    description="Fraud detection inference API powered by LightGBM",
    version="1.0.0",
)

# RETRAINING WORKFLOW

def update_retraining_status(**updates):
    with retraining_lock:
        retraining_status.update(updates)


def append_retraining_log(message: str):
    with retraining_lock:
        retraining_status["logs"].append(message)


def run_retraining_command(command: list[str], stage: str):
    append_retraining_log(
        f"[{stage}] Running: {' '.join(command)}"
    )

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            append_retraining_log(f"[{stage}] {line}")

    if result.stderr:
        for line in result.stderr.splitlines():
            append_retraining_log(f"[{stage} ERROR] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {result.returncode}."
        )

    return result.stdout


def retraining_workflow():
    try:
        update_retraining_status(
            status="running",
            stage="retraining",
            progress=10,
            message="Training candidate model...",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            promoted=False,
        )

        retrain_output = run_retraining_command(
            [
                sys.executable,
                "scripts/retrain_model.py",
            ],
            "RETRAIN",
        )

        match = re.search(
            r"Candidate version:\s*(\d+)",
            retrain_output,
        )

        if not match:
            raise RuntimeError(
                "Could not determine candidate version from retraining output."
            )

        candidate_version = int(match.group(1))

        candidate_report_path = (
            RETRAINING_DIR
            / f"candidate_version_{candidate_version}.json"
        )

        if not candidate_report_path.exists():
            raise FileNotFoundError(candidate_report_path)

        with open(candidate_report_path, "r", encoding="utf-8") as file:
            candidate_report = json.load(file)

        update_retraining_status(
            stage="evaluation",
            progress=45,
            candidate_version=candidate_version,
            message=f"Evaluating candidate version {candidate_version}...",
        )

        if not candidate_report["accepted"]:
            update_retraining_status(
                status="rejected",
                stage="validation",
                progress=100,
                candidate_version=candidate_version,
                message="Candidate failed the initial validation acceptance gate.",
                completed_at=datetime.now(timezone.utc).isoformat(),
                promoted=False,
            )
            return

        run_retraining_command(
            [
                sys.executable,
                "scripts/evaluate_candidate.py",
                "--candidate-version",
                str(candidate_version),
            ],
            "EVALUATE",
        )

        update_retraining_status(
            stage="promotion",
            progress=75,
            message=f"Applying promotion gate to candidate {candidate_version}...",
        )

        run_retraining_command(
            [
                sys.executable,
                "scripts/promote_candidate.py",
                "--candidate-version",
                str(candidate_version),
            ],
            "PROMOTE",
        )

        decision_path = (
            RETRAINING_DIR
            / f"decision_candidate_version_{candidate_version}.json"
        )

        if not decision_path.exists():
            raise FileNotFoundError(decision_path)

        with open(decision_path, "r", encoding="utf-8") as file:
            decision = json.load(file)

        if decision["promoted"]:
            update_retraining_status(
                stage="reload",
                progress=90,
                message="Promotion approved. Reloading active model...",
            )

            predictor.reload()

            active_version = str(
                predictor.config["model_version"]
            )

            if active_version != str(candidate_version):
                raise RuntimeError(
                    "Model reload completed but active model version "
                    "does not match promoted candidate."
                )

            update_retraining_status(
                status="completed",
                stage="completed",
                progress=100,
                candidate_version=candidate_version,
                message=f"Candidate {candidate_version} promoted and loaded.",
                completed_at=datetime.now(timezone.utc).isoformat(),
                promoted=True,
                active_version=active_version,
            )

        else:
            active_version = str(
                predictor.config["model_version"]
            )

            update_retraining_status(
                status="rejected",
                stage="completed",
                progress=100,
                candidate_version=candidate_version,
                message=(
                    f"Candidate {candidate_version} was rejected. "
                    f"Champion version {active_version} remains active."
                ),
                completed_at=datetime.now(timezone.utc).isoformat(),
                promoted=False,
                active_version=active_version,
            )

    except Exception as exc:
        append_retraining_log(
            f"[WORKFLOW ERROR] {type(exc).__name__}: {exc}"
        )

        update_retraining_status(
            status="failed",
            stage="error",
            progress=100,
            message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
            promoted=False,
        )

    finally:
        with retraining_lock:
            retraining_status["_lock_available"] = True


@app.get("/retraining/status")
def get_retraining_status():
    with retraining_lock:
        status = dict(retraining_status)
        status.pop("_lock_available", None)
        return status


@app.post("/retraining/start")
def start_retraining():
    with retraining_lock:
        if retraining_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="A retraining workflow is already running.",
            )

        retraining_status.update({
            "status": "starting",
            "stage": "queued",
            "progress": 0,
            "candidate_version": None,
            "message": "Retraining workflow queued.",
            "logs": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "promoted": False,
            "active_version": None,
        })

    thread = threading.Thread(
        target=retraining_workflow,
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "message": "Retraining workflow started.",
    }


@app.post("/retraining/decline")
def decline_retraining():
    with retraining_lock:
        if retraining_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot decline while retraining is already running.",
            )

        active_version = str(
            predictor.config["model_version"]
        )

        retraining_status.update({
            "status": "declined",
            "stage": "decision",
            "progress": 100,
            "candidate_version": None,
            "message": "Analyst declined retraining.",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "promoted": False,
            "active_version": active_version,
        })

    return {
        "status": "declined",
        "message": "Retraining declined. Active champion remains unchanged.",
    }

# LIVE WEBSOCKET CONNECTION MANAGER
class ConnectionManager:
    """Manage dashboard WebSocket connections and broadcast events."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []

        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    """Stream transaction events to connected dashboard clients."""

    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception:
        manager.disconnect(websocket)
@app.get("/monitoring/evidently/latest")
async def get_latest_evidently_report():
    """Return metadata for the latest Evidently drift report."""

    report_dir = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "evidently"
    )

    reports = list(
        report_dir.rglob("drift_report_*.html")
    )

    if not reports:
        raise HTTPException(
            status_code=404,
            detail="No Evidently reports have been generated yet.",
        )

    latest_report = max(
        reports,
        key=lambda report: report.stat().st_mtime,
    )

    return {
        "report_url": f"/api/monitoring/evidently/report/{latest_report.name}",
        "report_path": str(latest_report),
        "filename": latest_report.name,
    }

@app.get("/monitoring/evidently/report/{filename}")
async def serve_evidently_report(filename: str):
    """Serve a specific Evidently HTML report."""

    report_dir = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "evidently"
    )

    reports = list(
        report_dir.rglob(filename)
    )

    if not reports:
        raise HTTPException(
            status_code=404,
            detail="Evidently report not found.",
        )

    return FileResponse(
        path=reports[0],
        media_type="text/html",
    )

# ============================================================
# LIVE SIMULATION CONTROL ENDPOINTS
# ============================================================
@app.post("/simulation/start")
async def start_simulation(
    max_transactions: int | None = Query(default=None, ge=1),
    base_rate: float = Query(default=5.0, gt=0),
):
    """Start a background transaction simulation."""
    try:
        return simulation_controller.start(
            max_transactions=max_transactions,
            base_rate=base_rate,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/simulation/pause")
async def pause_simulation():
    """Pause the active simulation."""

    try:
        return simulation_controller.pause()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


@app.post("/simulation/resume")
async def resume_simulation():
    """Resume a paused simulation."""

    try:
        return simulation_controller.resume()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the active simulation."""

    try:
        return simulation_controller.stop()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


@app.get("/simulation/status")
async def get_simulation_status():
    """Return the current simulation state."""

    return simulation_controller.get_status()



# API REQUEST MONITORING
@app.middleware("http")
async def monitor_requests(request, call_next):
    """Track API request counts and response times."""

    start_time = time.perf_counter()
    method = request.method
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        duration = time.perf_counter() - start_time
        endpoint = request.url.path

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

# MODEL INITIALIZATION
# Load the champion model once when the API starts.
predictor = FraudPredictor()
BASE_DIR = Path(__file__).resolve().parent.parent

TRANSACTION_DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.parquet"
demo_transactions = pd.read_parquet(TRANSACTION_DATA_PATH)

with open(
    BASE_DIR / "config" / "fintech_mappings.json",
    "r",
    encoding="utf-8",
) as file:
    fintech_mappings = json.load(file)


fintech_adapters = {
    fintech_id: FintechAdapter(
        field_mapping=mapping,
        required_features=predictor.schema["feature_names"],
    )
    for fintech_id, mapping in fintech_mappings.items()
}

@app.get("/demo-transactions")
def get_demo_transactions(limit: int = Query(default=10, ge=1, le=50)):
    """Return sample transactions for frontend demonstrations."""

    sample = demo_transactions.sample(
        n=min(limit, len(demo_transactions)),
        random_state=42,
    )

    display_columns = [
        "TransactionID",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_31",
    ]

    available_columns = [
        column for column in display_columns
        if column in sample.columns
    ]

    clean_sample = (
        sample[available_columns]
        .astype(object)
        .where(pd.notna(sample[available_columns]), None)
    )
    records = clean_sample.to_dict(orient="records")

    for record, row_index in zip(records, sample.index):
        record["demo_reference"] = str(row_index)

    return {
        "transactions": records,
        "total": len(records),
    }
def persist_prediction(
    result: dict,
    transaction_id: int,
    transaction_amount=None,
    run_id: str = "fraudintel-live",
):
    """Persist a prediction and return its event for live broadcasting."""

    event = {
        "event_id": time.time_ns(),
        "run_id": run_id,
        "transaction_id": transaction_id,
        "timestamp": datetime.now(timezone.utc),
        "transaction_amount": transaction_amount,
        "fraud_probability": float(result["fraud_probability"]),
        "prediction": int(result["prediction"]),
        "risk_level": str(result["risk_level"]),
        "threshold": float(result["threshold"]),
    }

    save_event(event)

    return event

@app.post("/predict-demo")
def predict_demo(demo_reference: str):
    """Predict using the exact full feature row selected from demo data."""

    try:
        row_index = int(demo_reference)

        if row_index not in demo_transactions.index:
            raise HTTPException(
                status_code=404,
                detail="Demo transaction not found."
            )

        feature_names = predictor.schema["feature_names"]

        transaction = demo_transactions.loc[
            [row_index], feature_names
        ].copy()

        result = predictor.predict(transaction)

        persist_prediction(
            result,
            transaction_id=int(demo_reference),
            transaction_amount=None,
        )

        prediction_label = str(result["prediction"])
        risk_level = str(result["risk_level"])

        PREDICTIONS_TOTAL.labels(
            prediction=prediction_label,
            risk_level=risk_level
        ).inc()

        return {
            "demo_reference": demo_reference,
            "model": "FraudIntel_LightGBM",
            "prediction": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(exc))

# --------------------------------------------------
# HEALTH AND MODEL INFORMATION
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": predictor.config["model_name"],
        "model_version": predictor.config["model_version"],
        "model_type": predictor.config["model_type"],
        "threshold": predictor.threshold,
        "feature_count": predictor.schema["feature_count"],
    }


@app.get("/schema")
def get_schema():
    """Return the input requirements for fintech integrations."""

    return {
        "model": predictor.config["model_name"],
        "model_version": predictor.config["model_version"],
        "feature_count": predictor.schema["feature_count"],
        "required_features": predictor.schema["feature_names"],
        "categorical_features": predictor.schema["categorical_columns"],
        "numeric_features": [
            feature
            for feature in predictor.schema["feature_names"]
            if feature not in predictor.schema["categorical_columns"]
        ],
    }


@app.get("/integrations")
def get_integrations():
    """List configured fintech integrations."""

    return {
        "available_integrations": list(fintech_adapters.keys()),
        "total_integrations": len(fintech_adapters),
    }




# --------------------------------------------------
# SINGLE TRANSACTION PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(
    transaction: dict[str, Any],
    fintech_id: str = Header(
        default="default",
        alias="X-Fintech-ID",
    ),
    simulation_id: str | None = Header(
        default=None,
        alias="X-Simulation-ID",
    ),
    source_record_id: str | None = Header(
        default=None,
        alias="X-Source-Record-ID",
    ),
):
    """Predict fraud risk for a single transaction."""

    try:
        if fintech_id not in fintech_adapters:
            raise HTTPException(
                status_code=400,
                detail="Unknown fintech integration.",
            )

        # Validate simulation metadata.
        if simulation_id:
            if source_record_id is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Source record ID is required "
                        "for simulated transactions."
                    ),
                )

            try:
                transaction_id = int(source_record_id)

            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid source record ID.",
                ) from exc

            run_id = f"simulation-{simulation_id}"

        else:
            transaction_id = int(
                time.time_ns() % (2**63)
            )
            run_id = "fraudintel-live"

        # Transform incoming transaction.
        mapped_transaction = fintech_adapters[
            fintech_id
        ].transform_transaction(transaction)

        data = pd.DataFrame([mapped_transaction])

        try:
            # Run model inference outside the FastAPI event loop.
            result = await run_in_threadpool(
                predictor.predict,
                data,
            )

            # Persist prediction and capture the saved event.
            event = persist_prediction(
                result,
                transaction_id=transaction_id,
                transaction_amount=transaction.get(
                    "TransactionAmt"
                ),
                run_id=run_id,
            )

            # Update Prometheus metrics.
            PREDICTIONS_TOTAL.labels(
                prediction=str(result["prediction"]),
                risk_level=result["risk_level"],
            ).inc()

            # Broadcast the persisted transaction
            # to connected live dashboard clients.
            live_event = {
                **event,
                "timestamp": event["timestamp"].isoformat(),
            }

            await manager.broadcast({
                "type": "transaction",
                "event": live_event,
            })

        except Exception:
            PREDICTION_ERRORS.inc()
            raise

        return {
            "status": "success",
            "model": predictor.config["model_name"],
            "model_version": predictor.config["model_version"],
            **result,
        }

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc



# --------------------------------------------------
# TRANSACTION EXPLAINABILITY
# --------------------------------------------------

@app.post("/explain")
def explain(
    transaction: dict[str, Any],
    fintech_id: str = Header(default="default", alias="X-Fintech-ID"),
):
    """Explain why the model assigned a fraud risk to a transaction."""

    try:
        if fintech_id not in fintech_adapters:
            raise HTTPException(
                status_code=400,
                detail="Unknown fintech integration.",
            )

        mapped_transaction = fintech_adapters[
            fintech_id
        ].transform_transaction(transaction)

        data = pd.DataFrame([mapped_transaction])

        try:
            result = predictor.explain(data)

        except Exception:
            PREDICTION_ERRORS.inc()
            raise

        return {
            "status": "success",
            "model": predictor.config["model_name"],
            "model_version": predictor.config["model_version"],
            **result,
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# BATCH TRANSACTION PREDICTION
# --------------------------------------------------

@app.post("/predict-batch")
def predict_batch(transactions: list[dict[str, Any]]):
    """Predict fraud risk for multiple transactions."""

    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="Transaction list cannot be empty.",
        )

    try:
        data = pd.DataFrame(transactions)

        try:
            results = predictor.predict_batch(data)

        except Exception:
            PREDICTION_ERRORS.inc()
            raise

        for result in results:
            PREDICTIONS_TOTAL.labels(
                prediction=str(result["prediction"]),
                risk_level=result["risk_level"],
            ).inc()


        fraud_count = sum(
            result["prediction"] for result in results
        )

        return {
            "status": "success",
            "model": predictor.config["model_name"],
            "model_version": predictor.config["model_version"],
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "legitimate_transactions": len(results) - fraud_count,
            "fraud_detection_rate": fraud_count / len(results),
            "results": results,
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# CSV TRANSACTION PREDICTION
# --------------------------------------------------

@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    report_type: Literal["analyst", "full"] = Query("analyst"),
):
    """Upload transactions and download fraud prediction results."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty.",
            )

        data = pd.read_csv(BytesIO(contents))

        if data.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no transactions.",
            )

        try:
            results = predictor.predict_batch(data)

        except Exception:
            PREDICTION_ERRORS.inc()
            raise


        for result in results:
            PREDICTIONS_TOTAL.labels(
                prediction=str(result["prediction"]),
                risk_level=result["risk_level"],
            ).inc()

        # Persist batch predictions for transaction history and alerts.
        batch_run_id = f"batch-{time.time_ns()}"
        batch_timestamp = datetime.now(timezone.utc)
        batch_events = []

        for index, result in enumerate(results):
            row = data.iloc[index]

            event_id = time.time_ns() + index

            if "TransactionID" in data.columns and pd.notna(row.get("TransactionID")):
                transaction_id = int(row["TransactionID"])
            else:
                transaction_id = event_id

            transaction_amount = (
                float(row["TransactionAmt"])
                if "TransactionAmt" in data.columns
                and pd.notna(row["TransactionAmt"])
                else None
            )

            batch_events.append({
                "event_id": event_id,
                "run_id": batch_run_id,
                "transaction_id": transaction_id,
                "timestamp": batch_timestamp,
                "transaction_amount": transaction_amount,
                "fraud_probability": float(result["fraud_probability"]),
                "prediction": int(result["prediction"]),
                "risk_level": str(result["risk_level"]),
                "threshold": float(result["threshold"]),
            })

        save_events(batch_events)

        # Preserve original transaction data.
        output = data.copy()

        output["fraud_probability"] = [
            result["fraud_probability"] for result in results
        ]

        output["prediction"] = [
            result["prediction"] for result in results
        ]

        output["is_fraud"] = [
            result["is_fraud"] for result in results
        ]

        output["risk_level"] = [
            result["risk_level"] for result in results
        ]

        output["threshold"] = [
            result["threshold"] for result in results
        ]

        # Build the analyst report.
        if report_type == "analyst":

            analyst_columns = [
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card4",
                "card6",
                "fraud_probability",
                "prediction",
                "is_fraud",
                "risk_level",
                "threshold",
            ]

            available_columns = [
                column
                for column in analyst_columns
                if column in output.columns
            ]

            report = output[available_columns].copy()

            # Add a file row reference, not a bank transaction ID.
            report.insert(
                0,
                "row_reference",
                range(1, len(report) + 1),
            )

        else:
            # Full technical report.
            report = output

        # Generate downloadable CSV.
        csv_buffer = StringIO()
        report.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        fraud_count = sum(
            result["prediction"] for result in results
        )

        filename = f"fraudintel_{report_type}_report.csv"

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Total-Transactions": str(len(results)),
                "X-Fraud-Detected": str(fraud_count),
                "X-Legitimate-Transactions": str(
                    len(results) - fraud_count
                ),
            },
        )

    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# MONITORING ENDPOINTS
# --------------------------------------------------
@app.get("/monitoring/summary")
def monitoring_summary():
    """Return overall fraud monitoring statistics."""

    return get_summary()


@app.get("/monitoring/recent")
def monitoring_recent(limit: int = Query(default=20, ge=1, le=100)):
    """Return the most recent scored transactions."""

    return {
        "events": get_recent_events(limit),
        "limit": limit,
    }

@app.get("/monitoring/alerts")
def monitoring_alerts(limit: int = Query(default=20, ge=1, le=100)):
    """Return transactions flagged as potential fraud."""

    with duckdb.connect(str(DB_PATH)) as connection:
        rows = connection.execute(
            """
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
            WHERE prediction = 1
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

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

    return {
        "alerts": [
            dict(zip(columns, row))
            for row in rows
        ],
        "total_returned": len(rows),
    }
@app.get("/monitoring/risk-distribution")
def monitoring_risk_distribution():
    """Return transaction statistics grouped by risk level."""

    return {
        "risk_distribution": get_risk_distribution(),
    }


@app.get("/monitoring/hourly")
def monitoring_hourly(limit: int = Query(default=24, ge=1, le=168)):
    """Return hourly transaction monitoring statistics."""

    return {
        "hourly_statistics": get_hourly_stats(limit),
    }


# --------------------------------------------------
# PROMETHEUS METRICS
# --------------------------------------------------

@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Expose application metrics for Prometheus."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/monitoring/concept-drift/evaluate")
def evaluate_concept_drift(
    window_size: int = Query(default=1000, ge=100),
    min_fraud_count: int = Query(default=20, ge=1),
):
    """Evaluate and persist a concept-drift performance snapshot."""

    archive_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "monitoring"
        / "evidently"
        / "raw"
    )

    monitor = PerformanceMonitor(
        archive_dir=archive_dir,
        window_size=window_size,
        min_fraud_count=min_fraud_count,
    )

    try:
        windows = monitor.evaluate_rolling_windows()

    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Concept-drift evaluation failed: {exc}",
        ) from exc

    if not windows:
        return {
            "status": "insufficient_data",
            "message": "Not enough labelled transactions.",
            "saved": False,
        }

    evaluation_id = save_performance_evaluations(
        windows=windows,
        window_size=window_size,
    )

    latest = windows[-1]

    return {
        "status": (
            "concept_drift_detected"
            if latest["drift_alert"]
            else "evaluation_saved"
        ),
        "evaluation_id": evaluation_id,
        "saved": True,
        "windows_saved": len(windows),
        "latest_window": latest,
    }


@app.get("/monitoring/concept-drift/history")
def monitoring_concept_drift_history(
    limit: int = Query(default=50, ge=1, le=500),
):
    """Retrieve persisted concept-drift performance evaluations."""

    return {
        "evaluations": get_performance_history(limit),
        "limit": limit,
    }


@app.get("/monitoring/concept-drift")
def monitoring_concept_drift(
    window_size: int = Query(default=1000, ge=100),
    min_fraud_count: int = Query(default=20, ge=1),
):
    """Evaluate chronological model performance for concept drift."""

    archive_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "monitoring"
        / "evidently"
        / "raw"
    )

    monitor = PerformanceMonitor(
        archive_dir=archive_dir,
        window_size=window_size,
        min_fraud_count=min_fraud_count,
    )

    try:
        windows = monitor.evaluate_rolling_windows()

    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Concept-drift evaluation failed: {exc}",
        ) from exc

    if not windows:
        return {
            "status": "insufficient_data",
            "drift_alert": False,
            "message": (
                "Not enough labelled transactions "
                "to establish a performance baseline."
            ),
            "required_records": window_size,
            "evaluation_windows": [],
        }

    baseline = windows[0]
    latest = windows[-1]

    if len(windows) == 1:
        status = "baseline_established"
    elif latest["drift_alert"]:
        status = "concept_drift_detected"
    else:
        status = "no_significant_drift"

    return {
        "status": status,
        "drift_alert": latest["drift_alert"],
        "baseline": baseline,
        "latest_window": latest,
        "total_windows": len(windows),
        "evaluation_windows": windows,
    }
