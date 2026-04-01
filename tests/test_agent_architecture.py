from __future__ import annotations

import pytest

from agent_runtime.agent.service import AgentService
from agent_runtime.memory.base import MemoryProvider
from agent_runtime.memory.manager import MemoryManager
from agent_runtime.memory.providers.null import NullMemoryProvider
from agent_runtime.models import AgentStepRequest, HistoryEntryModel
from agent_runtime.scene_intent import SceneEntity, SceneIntent


class FakeDbProvider(MemoryProvider):
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def add_memory(self, record: dict[str, object]) -> None:
        del record

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


def _shared_test_intent(prompt: str, scene_state: dict[str, object]) -> SceneIntent:
    del prompt, scene_state
    return SceneIntent(
        status="ok",
        confidence=0.9,
        reasoning="Shared SceneIntent reasoning.",
        entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=2, group_id="cube_pair_group")],
    )


def test_agent_service_runs_with_memory_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _shared_test_intent)
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
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _shared_test_intent)
    provider = FakeDbProvider()
    service = AgentService(memory_manager=MemoryManager(provider=provider))

    response = service.step(
        AgentStepRequest(
            prompt="create a staircase of cubes",
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            history=[
                HistoryEntryModel(
                    iteration=1,
                    role="verification",
                    summary="Previous staircase attempt worked well",
                )
            ],
            iteration=2,
            execution_mode="vectra-dev",
        )
    )

    assert provider.queries == [("create a staircase of cubes", 5)]
    assert response.status == "ok"
    assert response.execution.kind == "tool_actions"
