
import logging
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
LOG_DIR = BASE_DIR / "logs"


def configure_logging(
    log_dir: Path = LOG_DIR,
    level: int = logging.INFO,
) -> Path:
    """Configure console and timestamped file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fraudintel_pipeline_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove only handlers owned by FraudIntel.
    for handler in root_logger.handlers[:]:
        if getattr(handler, "_fraudintel_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler._fraudintel_handler = True

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler._fraudintel_handler = True

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file
