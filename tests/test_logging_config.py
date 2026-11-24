
import logging

import pytest

from src.fraudintel.monitoring.logging_config import configure_logging


@pytest.fixture
def clean_fraudintel_handlers():
    """Remove FraudIntel handlers after each test."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level

    yield

    for handler in root_logger.handlers[:]:
        if getattr(handler, "_fraudintel_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(previous_level)


def test_logging_creates_file_and_records_message(
    tmp_path, clean_fraudintel_handlers
):
    log_file = configure_logging(log_dir=tmp_path)

    logging.getLogger("fraudintel.test").info(
        "Test pipeline message"
    )

    for handler in logging.getLogger().handlers:
        if getattr(handler, "_fraudintel_handler", False):
            handler.flush()

    assert log_file.is_file()
    assert "Test pipeline message" in log_file.read_text(
        encoding="utf-8"
    )
    assert "INFO" in log_file.read_text(encoding="utf-8")


def test_reconfiguration_does_not_duplicate_handlers(
    tmp_path, clean_fraudintel_handlers
):
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)

    fraudintel_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_fraudintel_handler", False)
    ]

    assert len(fraudintel_handlers) == 2
