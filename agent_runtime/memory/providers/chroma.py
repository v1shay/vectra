from __future__ import annotations

from typing import Any

from ..base import MemoryProvider


class ChromaMemoryProvider(MemoryProvider):
    """Stub provider reserved for future DB-backed memory."""

    def __init__(self) -> None:
        self._client = None

    def add_memory(self, record: dict[str, Any]) -> None:
        del record
        if self._client is None:
            return None
        return None

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        del query, top_k
        if self._client is None:
            return []
        return []

    def clear_memory(self) -> None:
        return None

    def is_enabled(self) -> bool:
        return self._client is not None
