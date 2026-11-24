
import json
import random
import threading
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from app.evidently_monitor import EvidentlyMonitor

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation_features.parquet"
)

EVENT_LOG_PATH = (
    BASE_DIR
    / "data"
    / "stream"
    / "transaction_events.jsonl"
)

EVIDENTLY_REFERENCE_PATH = (
    BASE_DIR / "data" / "processed" / "train_features.parquet"
)

EVIDENTLY_ARCHIVE_DIR = (
    BASE_DIR / "data" / "monitoring" / "evidently"
)

EVIDENTLY_REPORT_DIR = (
    BASE_DIR / "reports" / "evidently"
)

evidently_monitor = EvidentlyMonitor(
    reference_path=EVIDENTLY_REFERENCE_PATH,
    archive_dir=EVIDENTLY_ARCHIVE_DIR,
    report_dir=EVIDENTLY_REPORT_DIR,
    batch_size=100,
)


API_URL = "http://127.0.0.1:8000/predict"

MAX_TRANSACTIONS = 100
BASE_RATE_PER_SECOND = 5.0
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RANDOM_SEED = None

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================
def load_transactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation features and their corresponding fraud labels."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {DATA_PATH}"
        )

    target_path = (
        BASE_DIR / "data" / "processed" / "validation_target.parquet"
    )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Validation targets not found: {target_path}"
        )

    transactions = pd.read_parquet(DATA_PATH)
    validation_targets = pd.read_parquet(target_path)

    if transactions.empty:
        raise ValueError(
            "The validation dataset contains no transactions."
        )

    if len(transactions) != len(validation_targets):
        raise ValueError(
            "Validation features and targets have different row counts."
        )

    if "isFraud" not in validation_targets.columns:
        raise ValueError(
            "Validation targets are missing the 'isFraud' column."
        )

    return transactions, validation_targets


# ============================================================
# ACTIVITY PATTERN
# ============================================================

def get_activity_multiplier() -> float:
    """Simulate realistic banking transaction activity."""

    hour = datetime.now().hour

    if 6 <= hour < 10:
        return 1.5

    if 10 <= hour < 17:
        return 1.0

    if 17 <= hour < 22:
        return 1.4

    return 0.25


# ============================================================
# EVENT LOGGING
# ============================================================

def log_event(event: dict[str, Any]) -> None:
    """Append a successful simulated transaction to JSONL."""

    EVENT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EVENT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                event,
                default=lambda value: (
                    value.item()
                    if hasattr(value, "item")
                    else str(value)
                ),
            )
            + "\n"
        )


# ============================================================
# SEND TRANSACTION TO FASTAPI
# ============================================================

def send_transaction(
    transaction: dict[str, Any],
    simulation_id: str,
    source_record_id: int,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Send one transaction to the existing prediction API."""

    if stop_event and stop_event.is_set():
        raise InterruptedError("Simulation stopped.")

    payload = json.dumps(
        transaction,
        default=lambda value: (
            value.item()
            if hasattr(value, "item")
            else str(value)
        ),
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Fintech-ID": "default",
            "X-Simulation-ID": simulation_id,
            "X-Source-Record-ID": str(source_record_id),
        },
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):

        if stop_event and stop_event.is_set():
            raise InterruptedError("Simulation stopped.")

        try:
            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                response_body = response.read().decode(
                    "utf-8"
                )

                result = json.loads(response_body)

                if result.get("status") != "success":
                    raise RuntimeError(
                        f"Prediction API returned: {result}"
                    )

                return result

        except (HTTPError, URLError, TimeoutError, OSError) as exc:

            if stop_event and stop_event.is_set():
                raise InterruptedError(
                    "Simulation stopped."
                ) from exc

            print(
                f"API request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
            )

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Maximum API retries exceeded."
                ) from exc

            # Interruptible retry delay.
            if stop_event:
                if stop_event.wait(1):
                    raise InterruptedError(
                        "Simulation stopped."
                    )
            else:
                time.sleep(1)

    raise RuntimeError("Prediction request failed.")


# ============================================================
# INTERRUPTIBLE WAIT
# ============================================================

def wait_for_next_transaction(
    delay: float,
    stop_event: threading.Event,
    resume_event: threading.Event,
) -> bool:
    """
    Wait between transactions.

    Stop interrupts the wait immediately.
    Pause suspends the remaining delay.
    Resume continues the remaining delay.
    """

    remaining = max(0.0, delay)

    while remaining > 0:

        if stop_event.is_set():
            return False

        # Wait while paused.
        if not resume_event.wait(timeout=0.1):
            continue

        started = time.monotonic()

        interval = min(0.1, remaining)

        if stop_event.wait(interval):
            return False

        elapsed = time.monotonic() - started
        remaining = max(0.0, remaining - elapsed)

    return not stop_event.is_set()


# ============================================================
# CORE STREAM PROCESSOR
# ============================================================


def _run_stream(
    max_transactions: int | None,
    base_rate: float,
    stop_event: threading.Event,
    resume_event: threading.Event,
    simulation_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Process transactions chronologically until the limit or stop signal."""

    transactions, validation_targets = load_transactions()

    if "TransactionDT" not in transactions.columns:
        raise ValueError(
            "TransactionDT is required for chronological simulation."
        )

    # Sort features and labels using exactly the same row positions.
    chronological_order = (
        transactions["TransactionDT"]
        .argsort(kind="stable")
        .to_numpy()
    )

    transactions = (
        transactions.iloc[chronological_order]
        .reset_index(drop=True)
    )

    validation_targets = (
        validation_targets.iloc[chronological_order]
        .reset_index(drop=True)
    )

    total_rows = len(transactions)

    print(f"Loaded transactions: {total_rows:,}")
    print("Ordering: Chronological (TransactionDT)")
    print(f"Simulation ID: {simulation_id}")
    print(f"Baseline arrival rate: {base_rate}")
    print(
        "Mode: "
        + (
            f"Limited ({max_transactions})"
            if max_transactions is not None
            else "Continuous"
        )
    )
    print(f"Event log: {EVENT_LOG_PATH}")
    print("=" * 60)

    processed = 0
    failed = 0
    sequence_number = 0

    while (
        max_transactions is None
        or processed < max_transactions
    ):
        if stop_event.is_set():
            break

        # Wait here if the simulation is paused.
        if not resume_event.wait(timeout=0.1):
            continue

        if stop_event.is_set():
            break

        # Select the next transaction in chronological order.
        position = processed % total_rows

        actual_label = int(
            validation_targets.iloc[position]["isFraud"]
        )

        row = transactions.iloc[position]

        source_record_id = int(position)

        transaction = row.to_dict()
        sequence_number += 1

        try:
            result = send_transaction(
                transaction=transaction,
                simulation_id=simulation_id,
                source_record_id=source_record_id,
                stop_event=stop_event,
            )

            prediction_result = result.get(
                "result",
                result,
            )

            probability = float(
                prediction_result.get(
                    "fraud_probability",
                    prediction_result.get(
                        "probability",
                        0.0,
                    ),
                )
            )

            prediction = int(
                prediction_result["prediction"]
            )

            risk_level = prediction_result["risk_level"]

            threshold = prediction_result.get("threshold")
            timestamp = datetime.now(timezone.utc).isoformat()

            evidently_monitor.add_transaction(
                transaction=transaction,
                metadata={
                    "simulation_id": simulation_id,
                    "sequence_number": sequence_number,
                    "source_record_id": source_record_id,
                    "timestamp": timestamp,
                    "fraud_probability": probability,
                    "prediction": prediction,
                    "risk_level": risk_level,
                    "threshold": threshold,
                    "transaction_amount": transaction.get("TransactionAmt"),
                    "is_simulated": True,
                    "actual_label": actual_label,
                },
            )

            amount = transaction.get("TransactionAmt")

            event = {
                "simulation_id": simulation_id,
                "sequence_number": sequence_number,
                "source_record_id": source_record_id,
                "timestamp": timestamp,
                "transaction_amount": amount,
                "fraud_probability": probability,
                "prediction": prediction,
                "risk_level": risk_level,
                "threshold": threshold,
                "is_simulated": True,
            }

            log_event(event)

            processed += 1

            print(
                f"[{processed}] "
                f"Probability={probability:.4f} | "
                f"Prediction={prediction} | "
                f"Actual={actual_label} | "
                f"Risk={risk_level} | "
                f"Source={source_record_id}"
            )

            if progress_callback:
                progress_callback(processed, failed)

        except InterruptedError:
            break

        except Exception:
            failed += 1

            if progress_callback:
                progress_callback(processed, failed)

            raise

        if (
            max_transactions is not None
            and processed >= max_transactions
        ):
            break

        if stop_event.is_set():
            break

        # Calculate the next transaction delay.
        activity_multiplier = get_activity_multiplier()

        effective_rate = base_rate * activity_multiplier

        delay = 1.0 / effective_rate

        # Stop and pause are respected during this delay.
        if not wait_for_next_transaction(
            delay=delay,
            stop_event=stop_event,
            resume_event=resume_event,
        ):
            break

    return processed, failed

# ============================================================
# STANDALONE SIMULATOR
# ============================================================
def run_stream(
    max_transactions: int | None = None,
    base_rate: float = BASE_RATE_PER_SECOND,
) -> None:
    """
    Run simulator independently from the controller.

    Example:
        run_stream(max_transactions=1)
    """

    if max_transactions is not None and max_transactions < 1:
        raise ValueError(
            "max_transactions must be at least 1."
        )

    if base_rate <= 0:
        raise ValueError(
            "base_rate must be greater than zero."
        )

    stop_event = threading.Event()
    resume_event = threading.Event()

    resume_event.set()

    simulation_id = str(uuid.uuid4())

    print("=" * 60)
    print("FRAUDINTEL TRANSACTION SIMULATOR")
    print("=" * 60)

    try:
        _run_stream(
            max_transactions=max_transactions,
            base_rate=base_rate,
            stop_event=stop_event,
            resume_event=resume_event,
            simulation_id=simulation_id,
        )

    except KeyboardInterrupt:
        stop_event.set()
        print("\nStandalone simulation interrupted.")

    except Exception as exc:
        print(f"\nSimulation error: {exc}")

    finally:
        try:
            evidently_monitor.flush_simulation(simulation_id)
        except Exception as exc:
            print(f"Monitoring archive flush failed: {exc}")

    print("\nStandalone simulation finished.")


# ============================================================
# BACKGROUND SIMULATION CONTROLLER
# ============================================================

class SimulationController:
    """Control the background transaction simulator."""

    def __init__(self):

        # RLock prevents nested status calls from deadlocking.
        self._lock = threading.RLock()

        self._stop_event = threading.Event()
        self._resume_event = threading.Event()

        self._thread: threading.Thread | None = None

        self.state = "stopped"

        self.simulation_id: str | None = None

        self.processed = 0
        self.failed = 0

        self.error: str | None = None

        self.max_transactions: int | None = None
        self.base_rate = BASE_RATE_PER_SECOND

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def get_status(self) -> dict[str, Any]:

        with self._lock:

            thread_alive = (
                self._thread is not None
                and self._thread.is_alive()
            )

            return {
                "state": self.state,
                "simulation_id": self.simulation_id,
                "processed": self.processed,
                "failed": self.failed,
                "error": self.error,
                "is_running": (
                    thread_alive
                    and self.state == "running"
                ),
                "is_paused": (
                    thread_alive
                    and self.state == "paused"
                ),
                "max_transactions": self.max_transactions,
                "base_rate": self.base_rate,
            }

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    def _update_progress(
        self,
        processed: int,
        failed: int,
    ) -> None:

        with self._lock:
            self.processed = processed
            self.failed = failed

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(
        self,
        max_transactions: int | None = None,
        base_rate: float = BASE_RATE_PER_SECOND,
    ) -> dict[str, Any]:

        if max_transactions is not None:
            if (
                isinstance(max_transactions, bool)
                or not isinstance(max_transactions, int)
                or max_transactions < 1
            ):
                raise ValueError(
                    "max_transactions must be a positive integer."
                )

        if base_rate <= 0:
            raise ValueError(
                "base_rate must be greater than zero."
            )

        with self._lock:

            if (
                self._thread is not None
                and self._thread.is_alive()
            ):
                raise RuntimeError(
                    "A simulation is already active."

                )

            self._stop_event = threading.Event()
            self._resume_event = threading.Event()

            self._resume_event.set()

            self.simulation_id = str(uuid.uuid4())

            self.processed = 0
            self.failed = 0
            self.error = None

            self.max_transactions = max_transactions
            self.base_rate = float(base_rate)

            self.state = "running"

            self._thread = threading.Thread(
                target=self._worker,
                name="FraudIntelSimulationWorker",
                daemon=True,
            )

            self._thread.start()

            return self.get_status()

    # --------------------------------------------------------
    # PAUSE
    # --------------------------------------------------------

    def pause(self) -> dict[str, Any]:

        with self._lock:

            if self.state != "running":
                raise RuntimeError(
                    "Only a running simulation can be paused."
                )

            self._resume_event.clear()
            self.state = "paused"

            return self.get_status()

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    def resume(self) -> dict[str, Any]:

        with self._lock:

            if self.state != "paused":
                raise RuntimeError(
                    "Only a paused simulation can be resumed."
                )

            self._resume_event.set()
            self.state = "running"

            return self.get_status()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self) -> dict[str, Any]:

        with self._lock:

            if (
                self._thread is None
                or not self._thread.is_alive()
            ):
                self.state = "stopped"
                return self.get_status()

            # Signal the worker without waiting for its API call.
            self._stop_event.set()

            # Release a paused worker so it can exit.
            self._resume_event.set()

            self.state = "stopping"

            # IMPORTANT:
            # Do not join the worker here.
            # This endpoint must return immediately.

            return self.get_status()

    # --------------------------------------------------------
    # BACKGROUND WORKER
    # --------------------------------------------------------

    def _worker(self) -> None:
        try:
            processed, failed = _run_stream(
                max_transactions=self.max_transactions,
                base_rate=self.base_rate,
                stop_event=self._stop_event,
                resume_event=self._resume_event,
                simulation_id=self.simulation_id,
                progress_callback=self._update_progress,
            )

            with self._lock:
                self.processed = processed
                self.failed = failed

                if self._stop_event.is_set():
                    self.state = "stopped"
                else:
                    self.state = "completed"

        except Exception as exc:
            with self._lock:
                self.error = str(exc)
                self.state = "error"

            print(f"Simulation worker error: {exc}")

        finally:
            try:
                if self.simulation_id is not None:
                    evidently_monitor.flush_simulation(
                        self.simulation_id
                    )
            except Exception as exc:
                with self._lock:
                    self.error = f"Monitoring archive flush failed: {exc}"
                    self.state = "error"

                print(f"Monitoring archive flush failed: {exc}")

            with self._lock:
                if (
                        self._stop_event.is_set()
                        and self.state != "error"
                ):
                    self.state = "stopped"
# ============================================================
# SHARED CONTROLLER INSTANCE
# ============================================================

simulation_controller = SimulationController()


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    run_stream(
        max_transactions=MAX_TRANSACTIONS,
        base_rate=BASE_RATE_PER_SECOND,
    )
