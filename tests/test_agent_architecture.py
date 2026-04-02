from __future__ import annotations

import pytest

from agent_runtime.agent.service import AgentService
from agent_runtime.director.models import DirectorTurn
from agent_runtime.memory.base import MemoryProvider
from agent_runtime.memory.manager import MemoryManager
from agent_runtime.memory.providers.null import NullMemoryProvider
from agent_runtime.models import AgentStepRequest, HistoryEntryModel


class FakeDbProvider(MemoryProvider):
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []
        self.records: list[dict[str, object]] = []

    def add_memory(self, record: dict[str, object]) -> None:
        self.records.append(record)

    def query_memory(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        self.queries.append((query, top_k))
        return [{"category": "pattern", "summary": "fake-memory-hit"}]

    def clear_memory(self) -> None:
        return None

    def is_enabled(self) -> bool:
        return True


def test_memory_manager_defaults_to_null_provider_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_AGENT_MEMORY_ENABLED", "false")

    manager = MemoryManager()

    assert isinstance(manager.provider, NullMemoryProvider)
    assert manager.query_memory("stairs") == []


def test_agent_service_runs_with_memory_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.agent.reasoner._DIRECTOR_LOOP.step",
        lambda context: DirectorTurn(
            status="ok",
            message="Prepared the next step.",
            narration="Creating the first cube.",
            understanding="The Director handled the request.",
            plan=["use mesh.create_primitive as the next step"],
            intended_actions=["mesh.create_primitive({'type': 'cube'})"],
            expected_outcome="The first cube exists.",
            continue_loop=True,
            assumptions=[],
            metadata={"actions": [{"tool": "mesh.create_primitive", "params": {"type": "cube"}}]},
        ),
    )
    service = AgentService(memory_manager=MemoryManager(provider=NullMemoryProvider()))

    response = service.step(
        AgentStepRequest(
            prompt="make two cubes",
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            history=[],
            iteration=1,
            execution_mode="vectra-dev",
        )
    )

    assert response.status == "ok"
    assert response.execution.kind == "tool_actions"


def test_agent_service_uses_memory_through_the_abstraction_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.agent.reasoner._DIRECTOR_LOOP.step",
        lambda context: DirectorTurn(
            status="ok",
            message="Prepared the next step.",
            narration="Spacing the scene out.",
            understanding="The Director used recent context.",
            plan=["use object.transform as the next step"],
            intended_actions=["object.transform({'delta': [3.0, 0.0, 0.0]})"],
            expected_outcome="The scene spacing improves.",
            continue_loop=True,
            assumptions=[],
            metadata={"actions": [{"tool": "object.transform", "params": {"delta": [3.0, 0.0, 0.0]}}]},
        ),
    )
    provider = FakeDbProvider()
    service = AgentService(memory_manager=MemoryManager(provider=provider))

    response = service.step(
        AgentStepRequest(
            prompt="fix spacing",
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            history=[
                HistoryEntryModel(
                    iteration=1,
                    role="verification",
                    summary="Previous spacing attempt moved the cubes a little",
                )
            ],
            iteration=2,
            execution_mode="vectra-dev",
        )
    )

    assert provider.queries == [("fix spacing", 5)]
    assert provider.records
    assert response.status == "ok"
