from __future__ import annotations

from typing import Any

from ..base import MemoryProvider


class NullMemoryProvider(MemoryProvider):
    """Disabled provider that safely no-ops every call."""

    def add_memory(self, record: dict[str, Any]) -> None:
        del record

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        del query, top_k
        return []

    def clear_memory(self) -> None:
        return None

    def is_enabled(self) -> bool:
        return False
