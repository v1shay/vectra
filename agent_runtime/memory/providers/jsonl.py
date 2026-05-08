from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import MemoryProvider


class JsonlMemoryProvider(MemoryProvider):
    """Append-only local memory provider for graph/planner outcomes."""

    def __init__(self, path: str | Path, *, enabled: bool = True, limit: int = 1000) -> None:
        self.path = Path(path)
        self._enabled = enabled
        self._limit = max(int(limit), 1)

    def add_memory(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(json.dumps(record, default=str))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._enabled or not self.path.exists():
            return []
        normalized_terms = [term for term in query.lower().split() if term]
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(self._read_recent()):
            haystack = " ".join(
                str(record.get(key, ""))
                for key in (
                    "category",
                    "summary",
                    "prompt",
                    "benchmark",
                    "failure_signature",
                    "semantic_role",
                )
            ).lower()
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                haystack += " " + json.dumps(metadata, sort_keys=True, default=str).lower()
            score = sum(1 for term in normalized_terms if term in haystack)
            if not normalized_terms or score > 0:
                scored.append((score, index, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [dict(record) for _score, _index, record in scored[: max(top_k, 0)]]

    def clear_memory(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def is_enabled(self) -> bool:
        return self._enabled

    def _read_recent(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records[-self._limit :]
