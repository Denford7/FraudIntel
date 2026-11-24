
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
PARQUET_DIR = BASE_DIR / "data" / "interim"
CHUNK_SIZE = 50_000

FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def main():
    print("=" * 65)
    print("FraudIntel - CSV vs Parquet Value Verification")
    print("=" * 65)

    failures = []

    for filename in FILES:
        csv_path = RAW_DIR / filename
        parquet_path = PARQUET_DIR / filename.replace(".csv", ".parquet")

        print(f"\nChecking values: {filename}...")

        if not csv_path.exists() or not parquet_path.exists():
            failures.append(f"{filename}: CSV or Parquet file is missing.")
            print("FAIL: Required file not found.")
            continue

        # Read the Parquet file in batches using PyArrow, avoiding a
        # full in-memory load of the large transaction tables.
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        parquet_batches = iter(
            parquet_file.iter_batches(batch_size=CHUNK_SIZE)
        )

        rows_checked = 0
        file_failed = False

        try:
            for csv_chunk in pd.read_csv(
                csv_path,
                chunksize=CHUNK_SIZE,
            ):
                try:
                    parquet_batch = next(parquet_batches)
                except StopIteration:
                    raise AssertionError(
                        "Parquet ended before the CSV."
                    )

                parquet_chunk = parquet_batch.to_pandas()

                # Compare values, column order, and missing-value positions.
                # Dtype differences are allowed because CSV type inference
                # and Parquet storage may use different compatible dtypes.
                assert_frame_equal(
                    csv_chunk.reset_index(drop=True),
                    parquet_chunk.reset_index(drop=True),
                    check_dtype=False,
                    check_names=True,
                    check_like=False,
                    check_exact=True,
                )

                rows_checked += len(csv_chunk)

            try:
                next(parquet_batches)
                raise AssertionError(
                    "Parquet contains more rows than the CSV."
                )
            except StopIteration:
                pass

            print(f"Rows compared: {rows_checked:,}")
            print("PASS: All values match.")

        except (AssertionError, ValueError, TypeError) as exc:
            file_failed = True
            failures.append(f"{filename}: {exc}")
            print(f"FAIL: Value comparison failed: {exc}")

        if file_failed:
            continue

    print("\n" + "=" * 65)

    if failures:
        print("PARQUET VALUE VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PARQUET VALUE VERIFICATION PASSED")
    print("All CSV values match their corresponding Parquet values.")


if __name__ == "__main__":
    main()
