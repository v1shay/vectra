from __future__ import annotations

from collections import deque
from typing import Any

from ..base import MemoryProvider

_DEFAULT_LIMIT = 256


class InMemoryMemoryProvider(MemoryProvider):
    """Small in-process provider for architecture and integration tests."""

    def __init__(self, *, limit: int = _DEFAULT_LIMIT, enabled: bool = True) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=limit)
        self._enabled = enabled

    def add_memory(self, record: dict[str, Any]) -> None:
        if self._enabled:
            self._records.append(dict(record))

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._enabled:
            return []

        normalized_query = query.strip().lower()
        ranked: list[dict[str, Any]] = []
        for record in reversed(self._records):
            haystack = " ".join(
                [
                    str(record.get("category", "")),
                    str(record.get("summary", "")),
                    str(record.get("prompt", "")),
                ]
            ).lower()
            if not normalized_query or normalized_query in haystack:
                ranked.append(dict(record))
            if len(ranked) >= top_k:
                break
        return ranked

    def clear_memory(self) -> None:
        self._records.clear()

    def is_enabled(self) -> bool:
        return self._enabled
