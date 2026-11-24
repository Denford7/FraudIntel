
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
INTERIM_DIR = BASE_DIR / "data" / "interim"

FILES = {
    "train_transaction": RAW_DIR / "train_transaction.csv",
    "train_identity": RAW_DIR / "train_identity.csv",
    "test_transaction": RAW_DIR / "test_transaction.csv",
    "test_identity": RAW_DIR / "test_identity.csv",
}


def ingest_csv_to_parquet(name: str, csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    output_path = INTERIM_DIR / f"{name}.parquet"

    print(f"\nReading: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Rows read: {len(df):,}")
    print(f"Columns read: {len(df.columns):,}")

    if "TransactionID" in df.columns:
        if df["TransactionID"].isna().any():
            raise ValueError(f"{name}: missing TransactionID values")

        if df["TransactionID"].duplicated().any():
            raise ValueError(f"{name}: duplicate TransactionID values")

    if name == "train_transaction":
        if "isFraud" not in df.columns:
            raise ValueError("Training transactions lack isFraud")

        if df["isFraud"].isna().any():
            raise ValueError("Training target contains missing values")

        if not df["isFraud"].isin([0, 1]).all():
            raise ValueError("Training target must contain only 0 and 1")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Verify the saved Parquet file.
    saved = pd.read_parquet(output_path)

    if saved.shape != df.shape:
        raise RuntimeError(
            f"{name}: shape mismatch after Parquet conversion: "
            f"{df.shape} vs {saved.shape}"
        )

    print(f"Saved: {output_path.relative_to(BASE_DIR)}")
    print(f"Verified rows: {len(saved):,}")
    print(f"Verified columns: {len(saved.columns):,}")


def main() -> None:
    print("=" * 65)
    print("FraudIntel - CSV to Parquet Ingestion")
    print("=" * 65)

    for name, csv_path in FILES.items():
        ingest_csv_to_parquet(name, csv_path)

    print("\nAll four datasets ingested and verified successfully.")


if __name__ == "__main__":
    main()
