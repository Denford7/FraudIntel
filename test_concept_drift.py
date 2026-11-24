
import json
from pathlib import Path

from app.performance_monitor import PerformanceMonitor


SOURCE_DIR = Path("data/monitoring/evidently/raw")
EXPERIMENT_DIR = Path(
    "data/monitoring/concept_drift_experiment/raw"
)

WINDOW_SIZE = 1000
INJECTED_FRAUD_CASES = 50


def main() -> None:
    records = []

    # Read the original monitoring archives.
    for path in SOURCE_DIR.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

    if not records:
        raise ValueError("No archived records were found.")

    # Select the latest simulation.
    simulations = {}

    for record in records:
        simulation_id = record.get("simulation_id")
        timestamp = record.get("timestamp", "")

        if simulation_id:
            simulations.setdefault(simulation_id, []).append(record)

    if not simulations:
        raise ValueError("No simulation IDs found in the archive.")

    latest_simulation_id = max(
        simulations,
        key=lambda key: max(
            record.get("timestamp", "")
            for record in simulations[key]
        ),
    )

    records = simulations[latest_simulation_id]

    # Keep records with valid labels and event timestamps.
    records = [
        record for record in records
        if (
            record.get("actual_label") in (0, 1)
            and record.get("prediction") in (0, 1)
            and record.get("TransactionDT") is not None
        )
    ]

    records.sort(key=lambda record: record["TransactionDT"])

    required_records = WINDOW_SIZE * 2

    if len(records) < required_records:
        raise ValueError(
            f"Need {required_records} records, "
            f"but found {len(records)}."
        )

    # Copy two chronological windows.
    experiment_records = [
        dict(record)
        for record in records[:required_records]
    ]

    baseline = experiment_records[:WINDOW_SIZE]
    drift_window = experiment_records[WINDOW_SIZE:]

    # Inject concept drift by changing 50 previously legitimate,
    # correctly classified transactions into fraud cases.
    candidates = [
        record
        for record in drift_window
        if (
            int(record["actual_label"]) == 0
            and int(record["prediction"]) == 0
        )
    ]

    if len(candidates) < INJECTED_FRAUD_CASES:
        raise ValueError(
            "Not enough legitimate, correctly classified "
            "transactions to inject the experiment."
        )

    for record in candidates[:INJECTED_FRAUD_CASES]:
        record["actual_label"] = 1

    # Keep the experiment separate from the original simulation.
    for record in experiment_records:
        record["simulation_id"] = "controlled-concept-drift"
        record["experiment"] = True

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EXPERIMENT_DIR / "experiment.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for record in experiment_records:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    print("Controlled concept-drift experiment created.")
    print(f"Original simulation: {latest_simulation_id}")
    print(f"Baseline records: {len(baseline)}")
    print(f"Drift-window records: {len(drift_window)}")
    print(f"Injected additional fraud cases: {INJECTED_FRAUD_CASES}")
    print(f"Experiment archive: {output_path}")

    # Evaluate the experiment using our existing monitor.
    monitor = PerformanceMonitor(
        archive_dir=EXPERIMENT_DIR,
        window_size=WINDOW_SIZE,
        min_fraud_count=20,
    )

    print("\nConcept-drift evaluation:")

    for result in monitor.evaluate_rolling_windows():
        print(result)


if __name__ == "__main__":
    main()
