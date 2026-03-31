from __future__ import annotations

import logging
import sys
from typing import Any


def _configure_root_logger() -> logging.Logger:
    root_logger = logging.getLogger("vectra")
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False
    return root_logger


def get_vectra_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def log_structured(
    logger: logging.Logger,
    event: str,
    payload: Any,
    *,
    level: str = "info",
) -> None:
    log_method = getattr(logger, level, logger.info)
    log_method("%s %s", event, payload)


def log_action_start(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    params: dict[str, Any],
) -> None:
    log_structured(
        logger,
        "action_start",
        {"action_id": action_id, "tool": tool, "params": params},
    )


def log_action_success(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    outputs: dict[str, Any],
) -> None:
    log_structured(
        logger,
        "action_success",
        {"action_id": action_id, "tool": tool, "outputs": outputs},
    )


def log_action_failure(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    error: str,
) -> None:
    log_structured(
        logger,
        "action_failure",
        {"action_id": action_id, "tool": tool, "error": error},
        level="error",
    )


def log_execution_report(logger: logging.Logger, report: Any) -> None:
    log_structured(logger, "execution_report", report)
