
from pathlib import Path
import csv
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"

REQUIRED_COLUMNS = {
    "train_transaction.csv": {"TransactionID", "isFraud", "TransactionDT"},
    "train_identity.csv": {"TransactionID"},
    "test_transaction.csv": {"TransactionID", "TransactionDT"},
    "test_identity.csv": {"TransactionID"},
    "sample_submission.csv": {"TransactionID", "isFraud"},
}


def main() -> int:
    print("=" * 60)
    print("FraudIntel - Raw Data Validation")
    print("=" * 60)

    errors = []

    for filename, required_columns in REQUIRED_COLUMNS.items():
        path = RAW_DIR / filename

        if not path.is_file():
            errors.append(f"Missing file: {filename}")
            continue

        if path.stat().st_size == 0:
            errors.append(f"Empty file: {filename}")
            continue

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.reader(file)
                header = next(reader, None)

            if not header:
                errors.append(f"Missing CSV header: {filename}")
                continue

            if len(header) != len(set(header)):
                errors.append(f"Duplicate column names: {filename}")

            missing_columns = required_columns - set(header)
            if missing_columns:
                errors.append(
                    f"{filename}: missing required columns "
                    f"{sorted(missing_columns)}"
                )
                continue

            print(
                f"PASS: {filename} | "
                f"{path.stat().st_size:,} bytes | "
                f"{len(header)} columns"
            )

        except (OSError, csv.Error, UnicodeError) as exc:
            errors.append(f"Cannot read {filename}: {exc}")

    if errors:
        print("\nVALIDATION FAILED:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("\nRAW DATA VALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
