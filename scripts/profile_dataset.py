from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"


def profile_file(file_path, sample_rows=10000):
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(file_path, nrows=sample_rows)

    print(f"Sample rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nTop 20 missing columns:")
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .mul(100)
        .round(2)
    )
    print(missing.to_string())

    if "isFraud" in df.columns:
        print("\nTarget distribution:")
        print(df["isFraud"].value_counts())

        print("\nTarget percentage:")
        print(
            df["isFraud"]
            .value_counts(normalize=True)
            .mul(100)
            .round(3)
        )

    print()


def main():
    transaction_file = DATA_DIR / "train_transaction.csv"
    identity_file = DATA_DIR / "train_identity.csv"

    profile_file(transaction_file)
    profile_file(identity_file)


if __name__ == "__main__":
    main()