
from pathlib import Path
import subprocess
import sys
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "ieee-fraud-detection"
DOWNLOAD_DIR = BASE_DIR / "data" / "external" / "kaggle_download"

COMPETITION = "ieee-fraud-detection"

EXPECTED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve existing data when all expected files are present.
    missing_files = [
        name for name in EXPECTED_FILES
        if not (RAW_DIR / name).is_file()
    ]

    if not missing_files:
        print("All five expected dataset files already exist.")
        print(f"Dataset location: {RAW_DIR}")
        print("Download skipped; existing data was preserved.")
        return

    print("Missing dataset files:")
    for name in missing_files:
        print(f"  - {name}")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading dataset from Kaggle...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DOWNLOAD_DIR),
        ],
        check=True,
    )

    zip_path = DOWNLOAD_DIR / f"{COMPETITION}.zip"

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Kaggle download completed, but ZIP was not found: {zip_path}"
        )

    print("\nExtracting missing files without replacing existing files...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        available_names = set(archive.namelist())

        for name in missing_files:
            if name not in available_names:
                raise FileNotFoundError(
                    f"{name} was not found inside the downloaded archive."
                )

            with archive.open(name) as source:
                destination = RAW_DIR / name

                # Exclusive creation prevents overwriting an existing file.
                with destination.open("xb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)

            print(f"Saved: {destination.name}")

    still_missing = [
        name for name in EXPECTED_FILES
        if not (RAW_DIR / name).is_file()
    ]

    if still_missing:
        raise RuntimeError(
            f"Dataset acquisition incomplete. Missing: {still_missing}"
        )

    print("\nDataset acquisition completed successfully.")
    print(f"Dataset location: {RAW_DIR}")


if __name__ == "__main__":
    main()
