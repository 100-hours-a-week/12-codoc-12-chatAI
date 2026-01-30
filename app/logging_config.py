import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler


LOG_DIR = "/home/ubuntu/ai"
TEXT_LOG_PATH = os.path.join(LOG_DIR, "app.log")
JSON_LOG_PATH = os.path.join(LOG_DIR, "app.json.log")


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON lines for CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "json_data", {}) or {}

        # Ensure required fields exist even if middleware missed them.
        payload.setdefault(
            "timestamp",
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        payload.setdefault("level", record.levelname)
        payload.setdefault("message", record.getMessage())

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _handler_exists(logger: logging.Logger, target_path: str) -> bool:
    """Prevent duplicate handlers when uvicorn reloads."""

    for handler in logger.handlers:
        if hasattr(handler, "baseFilename") and os.path.abspath(
            getattr(handler, "baseFilename")
        ) == os.path.abspath(target_path):
            return True
    return False


def setup_logging() -> None:
    """Configure text + JSON rotating file handlers for the root logger."""

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Human-friendly text log
    if not _handler_exists(logger, TEXT_LOG_PATH):
        text_handler = TimedRotatingFileHandler(
            TEXT_LOG_PATH,
            when="midnight",
            backupCount=14,
            encoding="utf-8",
            delay=True,
        )
        text_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        text_handler.setFormatter(text_formatter)
        logger.addHandler(text_handler)

    # CloudWatch-friendly JSON log
    if not _handler_exists(logger, JSON_LOG_PATH):
        json_handler = TimedRotatingFileHandler(
            JSON_LOG_PATH,
            when="midnight",
            backupCount=14,
            encoding="utf-8",
            delay=True,
        )
        json_handler.setFormatter(JsonFormatter())
        logger.addHandler(json_handler)

    # Drop uvicorn access logs; we log via middleware instead
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
