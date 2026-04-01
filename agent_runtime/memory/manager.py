from __future__ import annotations

import os
from typing import Any

from .base import MemoryProvider
from .providers import ChromaMemoryProvider, InMemoryMemoryProvider, NullMemoryProvider

_DEFAULT_PROVIDER = "in_memory"


def _flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(flag_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class MemoryManager:
    """Single access point for agent memory."""

    def __init__(self, provider: MemoryProvider | None = None) -> None:
        self._provider = provider or self._build_provider()

    @staticmethod
    def _build_provider() -> MemoryProvider:
        enabled = _flag_enabled("VECTRA_AGENT_MEMORY_ENABLED", default=False)
        if not enabled:
            return NullMemoryProvider()

        provider_name = os.getenv("VECTRA_AGENT_MEMORY_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        if provider_name == "chroma":
            return ChromaMemoryProvider()
        if provider_name == "in_memory":
            return InMemoryMemoryProvider(enabled=True)
        return NullMemoryProvider()

    def add_memory(self, record: dict[str, Any]) -> None:
        self._provider.add_memory(record)

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._provider.query_memory(query, top_k=top_k)

    def clear_memory(self) -> None:
        self._provider.clear_memory()

    def is_enabled(self) -> bool:
        return self._provider.is_enabled()

    @property
    def provider(self) -> MemoryProvider:
        return self._provider
