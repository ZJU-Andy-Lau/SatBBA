"""Central logging setup for SatBBA."""

from __future__ import annotations

import logging
from pathlib import Path

from satbba.config.settings import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure root logger with console and optional file output."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "satbba.log", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, config.level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
