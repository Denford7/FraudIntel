
import csv

from scripts import validate_raw_data


def create_csv(folder, filename, columns):
    """Create a small CSV file containing the specified headers."""
    path = folder / filename

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerow(["1"] * len(columns))

    return path


def create_valid_dataset(folder):
    """Create all five CSV files with the required columns."""
    for filename, required_columns in (
        validate_raw_data.REQUIRED_COLUMNS.items()
    ):
        create_csv(folder, filename, sorted(required_columns))


def test_valid_raw_files_pass(monkeypatch, tmp_path):
    create_valid_dataset(tmp_path)
    monkeypatch.setattr(validate_raw_data, "RAW_DIR", tmp_path)

    assert validate_raw_data.main() == 0


def test_missing_file_fails(monkeypatch, tmp_path):
    create_valid_dataset(tmp_path)
    (tmp_path / "train_identity.csv").unlink()
    monkeypatch.setattr(validate_raw_data, "RAW_DIR", tmp_path)

    assert validate_raw_data.main() == 1


def test_empty_file_fails(monkeypatch, tmp_path):
    create_valid_dataset(tmp_path)
    (tmp_path / "train_identity.csv").write_text(
        "", encoding="utf-8"
    )
    monkeypatch.setattr(validate_raw_data, "RAW_DIR", tmp_path)

    assert validate_raw_data.main() == 1


def test_missing_required_column_fails(monkeypatch, tmp_path):
    create_valid_dataset(tmp_path)

    create_csv(
        tmp_path,
        "train_transaction.csv",
        ["TransactionID", "TransactionDT"],
    )

    monkeypatch.setattr(validate_raw_data, "RAW_DIR", tmp_path)

    assert validate_raw_data.main() == 1


def test_duplicate_column_names_fail(monkeypatch, tmp_path):
    create_valid_dataset(tmp_path)

    path = tmp_path / "train_transaction.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["TransactionID", "isFraud", "TransactionDT", "TransactionID"]
        )
        writer.writerow(["1", "0", "100", "1"])

    monkeypatch.setattr(validate_raw_data, "RAW_DIR", tmp_path)

    assert validate_raw_data.main() == 1
