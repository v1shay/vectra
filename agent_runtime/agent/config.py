from __future__ import annotations

import base64
import os
from pathlib import Path


def _flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(flag_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def vision_enabled() -> bool:
    return _flag_enabled("VECTRA_LLM_SUPPORTS_VISION", default=False)


def screenshot_to_data_url(path: str | None) -> str | None:
    if not path:
        return None

    file_path = Path(path)
    if not file_path.is_file():
        return None

    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
