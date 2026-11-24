
from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
CHUNK_SIZE = 100_000


def scan_csv(filename, check_target=False):
    """Check transaction IDs and, optionally, the fraud target."""
    path = RAW_DIR / filename
    seen_ids = set()
    target_counts = Counter()
    errors = []
    rows = 0

    columns = ["TransactionID"]
    if check_target:
        columns.append("isFraud")

    print(f"\nChecking {filename}...")

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
    ):
        rows += len(chunk)

        ids = chunk["TransactionID"]

        if ids.isna().any():
            errors.append("Missing TransactionID values found.")

        if ids.duplicated().any():
            errors.append("Duplicate TransactionID values within a chunk.")

        valid_ids = ids.dropna().astype("int64")
        current_ids = set(valid_ids.tolist())

        if seen_ids.intersection(current_ids):
            errors.append("Duplicate TransactionID across chunks.")

        seen_ids.update(current_ids)

        if check_target:
            target = chunk["isFraud"]

            if target.isna().any():
                errors.append("Missing isFraud target values found.")

            invalid = ~target.isin([0, 1]) & target.notna()
            if invalid.any():
                errors.append("Invalid isFraud values found.")

            target_counts.update(
                target.dropna().astype(int).tolist()
            )

    print(f"Rows scanned: {rows:,}")
    print(f"Unique non-missing IDs: {len(seen_ids):,}")

    if check_target:
        print(f"Target counts: {dict(target_counts)}")

    if errors:
        for error in sorted(set(errors)):
            print(f"FAIL: {error}")
    else:
        print("PASS: ID integrity checks completed.")

    return seen_ids, rows, errors, target_counts


def main():
    print("=" * 65)
    print("FraudIntel - Data Integrity Validation")
    print("=" * 65)

    all_errors = []

    train_ids, train_rows, errors, target_counts = scan_csv(
        "train_transaction.csv",
        check_target=True,
    )
    all_errors.extend(
        f"train_transaction.csv: {error}" for error in errors
    )

    test_ids, test_rows, errors, _ = scan_csv(
        "test_transaction.csv",
    )
    all_errors.extend(
        f"test_transaction.csv: {error}" for error in errors
    )

    train_identity_ids, _, errors, _ = scan_csv(
        "train_identity.csv",
    )
    all_errors.extend(
        f"train_identity.csv: {error}" for error in errors
    )

    test_identity_ids, _, errors, _ = scan_csv(
        "test_identity.csv",
    )
    all_errors.extend(
        f"test_identity.csv: {error}" for error in errors
    )

    # Identity records should reference existing transactions.
    unmatched_train = train_identity_ids - train_ids
    unmatched_test = test_identity_ids - test_ids

    if unmatched_train:
        all_errors.append(
            f"{len(unmatched_train):,} training identity IDs "
            "have no matching training transaction."
        )

    if unmatched_test:
        all_errors.append(
            f"{len(unmatched_test):,} test identity IDs "
            "have no matching test transaction."
        )

    # Confirm expected row counts for the known dataset.
    if train_rows != 590_540:
        all_errors.append(
            f"Unexpected training transaction row count: {train_rows:,}"
        )

    if test_rows != 506_691:
        all_errors.append(
            f"Unexpected test transaction row count: {test_rows:,}"
        )

    print("\n" + "=" * 65)
    print("FINAL RESULT")

    if all_errors:
        print("DATA INTEGRITY VALIDATION FAILED")
        for error in sorted(set(all_errors)):
            print(f"- {error}")
        raise SystemExit(1)

    print("DATA INTEGRITY VALIDATION PASSED")
    print("No missing or duplicate transaction IDs were detected.")
    print("Identity references and fraud targets passed validation.")


if __name__ == "__main__":
    main()
