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


def log_action_start(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    params: dict[str, Any],
) -> None:
    logger.info(
        "action_start %s",
        {"action_id": action_id, "tool": tool, "params": params},
    )


def log_action_success(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    outputs: dict[str, Any],
) -> None:
    logger.info(
        "action_success %s",
        {"action_id": action_id, "tool": tool, "outputs": outputs},
    )


def log_action_failure(
    logger: logging.Logger,
    action_id: str | None,
    tool: str,
    error: str,
) -> None:
    logger.error(
        "action_failure %s",
        {"action_id": action_id, "tool": tool, "error": error},
    )
