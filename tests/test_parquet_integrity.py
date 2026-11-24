
import pandas as pd
import pytest

from scripts import verify_parquet_integrity as verifier


FILES = verifier.FILES


def create_matching_files(folder):
    """Create small CSV and Parquet files for all four datasets."""
    raw_dir = folder / "raw"
    parquet_dir = folder / "parquet"

    raw_dir.mkdir()
    parquet_dir.mkdir()

    for filename in FILES:
        df = pd.DataFrame({
            "TransactionID": [1001, 1002, 1003],
            "value": [10.5, 20.0, 30.5],
        })

        df.to_csv(raw_dir / filename, index=False)
        df.to_parquet(
            parquet_dir / filename.replace(".csv", ".parquet"),
            index=False,
        )

    return raw_dir, parquet_dir


def configure_paths(monkeypatch, raw_dir, parquet_dir):
    monkeypatch.setattr(verifier, "RAW_DIR", raw_dir)
    monkeypatch.setattr(verifier, "PARQUET_DIR", parquet_dir)


def test_matching_csv_and_parquet_files_pass(
    monkeypatch, tmp_path
):
    raw_dir, parquet_dir = create_matching_files(tmp_path)
    configure_paths(monkeypatch, raw_dir, parquet_dir)

    verifier.main()


def test_missing_csv_fails(monkeypatch, tmp_path):
    raw_dir, parquet_dir = create_matching_files(tmp_path)
    (raw_dir / FILES[0]).unlink()
    configure_paths(monkeypatch, raw_dir, parquet_dir)

    with pytest.raises(SystemExit) as exc:
        verifier.main()

    assert exc.value.code == 1


def test_missing_parquet_fails(monkeypatch, tmp_path):
    raw_dir, parquet_dir = create_matching_files(tmp_path)
    (parquet_dir / FILES[0].replace(".csv", ".parquet")).unlink()
    configure_paths(monkeypatch, raw_dir, parquet_dir)

    with pytest.raises(SystemExit) as exc:
        verifier.main()

    assert exc.value.code == 1


def test_row_count_mismatch_fails(monkeypatch, tmp_path):
    raw_dir, parquet_dir = create_matching_files(tmp_path)

    parquet_path = parquet_dir / FILES[0].replace(".csv", ".parquet")
    pd.DataFrame({
        "TransactionID": [1001, 1002],
        "value": [10.5, 20.0],
    }).to_parquet(parquet_path, index=False)

    configure_paths(monkeypatch, raw_dir, parquet_dir)

    with pytest.raises(SystemExit) as exc:
        verifier.main()

    assert exc.value.code == 1


def test_column_order_mismatch_fails(monkeypatch, tmp_path):
    raw_dir, parquet_dir = create_matching_files(tmp_path)

    parquet_path = parquet_dir / FILES[0].replace(".csv", ".parquet")
    pd.DataFrame({
        "value": [10.5, 20.0, 30.5],
        "TransactionID": [1001, 1002, 1003],
    }).to_parquet(parquet_path, index=False)

    configure_paths(monkeypatch, raw_dir, parquet_dir)

    with pytest.raises(SystemExit) as exc:
        verifier.main()

    assert exc.value.code == 1
