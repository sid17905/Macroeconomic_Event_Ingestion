"""Logging configuration and a retry decorator for flaky network/API calls."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure root logging once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers)
    # Quiet noisy third-party loggers.
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def api_retry(max_attempts: int = 4, logger: logging.Logger | None = None):
    """Exponential-backoff retry decorator for network/API calls.

    Retries on any exception (API client wrappers raise heterogeneous errors),
    re-raising the final failure so callers can handle/fall back.
    """
    log = logger or logging.getLogger("api")
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
