
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
PARQUET_DIR = BASE_DIR / "data" / "interim"

FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def main():
    print("=" * 65)
    print("FraudIntel - CSV vs Parquet Verification")
    print("=" * 65)

    failures = []

    for filename in FILES:
        csv_path = RAW_DIR / filename
        parquet_path = PARQUET_DIR / filename.replace(".csv", ".parquet")

        print(f"\nChecking {filename}...")

        if not csv_path.exists():
            failures.append(f"Missing CSV: {csv_path}")
            print("FAIL: Source CSV not found.")
            continue

        if not parquet_path.exists():
            failures.append(f"Missing Parquet: {parquet_path}")
            print("FAIL: Parquet file not found.")
            continue

        # Compare row and column counts without loading all CSV columns.
        csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        parquet_columns = pd.read_parquet(parquet_path).columns.tolist()

        csv_rows = sum(
            len(chunk)
            for chunk in pd.read_csv(csv_path, chunksize=100_000)
        )

        parquet_rows = len(pd.read_parquet(parquet_path))

        checks_passed = True

        if csv_rows != parquet_rows:
            failures.append(
                f"{filename}: row count differs "
                f"(CSV={csv_rows}, Parquet={parquet_rows})."
            )
            checks_passed = False

        if csv_columns != parquet_columns:
            failures.append(
                f"{filename}: column names or order differ."
            )
            checks_passed = False

        print(f"CSV rows:      {csv_rows:,}")
        print(f"Parquet rows:  {parquet_rows:,}")
        print(f"CSV columns:   {len(csv_columns)}")
        print(f"Parquet columns: {len(parquet_columns)}")

        if checks_passed:
            print("PASS: Row counts and column names/order match.")
        else:
            print("FAIL: One or more comparisons failed.")

    print("\n" + "=" * 65)

    if failures:
        print("PARQUET VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PARQUET VERIFICATION PASSED")
    print("All four CSV and Parquet schemas and row counts match.")


if __name__ == "__main__":
    main()
