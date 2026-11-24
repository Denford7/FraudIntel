
from collections import Counter

import pandas as pd

from scripts import validate_data_integrity as validator


def write_csv(folder, filename, data):
    path = folder / filename
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_valid_transaction_data_passes(monkeypatch, tmp_path):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, 1002, 1003],
            "isFraud": [0, 1, 0],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)

    ids, rows, errors, target_counts = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert ids == {1001, 1002, 1003}
    assert rows == 3
    assert errors == []
    assert target_counts == Counter({0: 2, 1: 1})


def test_duplicate_ids_within_chunk_are_detected(
    monkeypatch, tmp_path
):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, 1001],
            "isFraud": [0, 1],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)

    _, _, errors, _ = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert any("Duplicate TransactionID" in error for error in errors)


def test_duplicate_ids_across_chunks_are_detected(
    monkeypatch, tmp_path
):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, 1002, 1001],
            "isFraud": [0, 0, 1],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)
    monkeypatch.setattr(validator, "CHUNK_SIZE", 2)

    _, _, errors, _ = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert "Duplicate TransactionID across chunks." in errors


def test_missing_transaction_ids_are_detected(
    monkeypatch, tmp_path
):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, None],
            "isFraud": [0, 1],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)

    _, _, errors, _ = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert "Missing TransactionID values found." in errors


def test_invalid_fraud_target_is_detected(
    monkeypatch, tmp_path
):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, 1002, 1003],
            "isFraud": [0, 1, 2],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)

    _, _, errors, _ = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert "Invalid isFraud values found." in errors


def test_missing_fraud_target_is_detected(
    monkeypatch, tmp_path
):
    write_csv(
        tmp_path,
        "train_transaction.csv",
        {
            "TransactionID": [1001, 1002],
            "isFraud": [0, None],
        },
    )
    monkeypatch.setattr(validator, "RAW_DIR", tmp_path)

    _, _, errors, _ = validator.scan_csv(
        "train_transaction.csv",
        check_target=True,
    )

    assert "Missing isFraud target values found." in errors


def test_unmatched_identity_references_fail_validation(
    monkeypatch, capsys
):
    # Simulate scan results while retaining main()'s actual
    # identity-reference and expected-row-count validation logic.
    results = iter([
        ({1001, 1002}, 590_540, [], Counter({0: 1, 1: 1})),
        ({2001, 2002}, 506_691, [], Counter()),
        ({1001, 9999}, 2, [], Counter()),
        ({2001}, 1, [], Counter()),
    ])

    monkeypatch.setattr(
        validator,
        "scan_csv",
        lambda *args, **kwargs: next(results),
    )

    try:
        validator.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError(
            "Expected validation to fail for an unmatched identity ID."
        )

    output = capsys.readouterr().out
    assert "training identity IDs have no matching" in output
    assert "DATA INTEGRITY VALIDATION FAILED" in output
