from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryProvider(ABC):
    """Abstract memory provider used by the agent loop."""

    @abstractmethod
    def add_memory(self, record: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def clear_memory(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError
