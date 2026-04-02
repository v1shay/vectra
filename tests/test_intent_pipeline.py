from __future__ import annotations

from agent_runtime.director.loop import DirectorLoop
from agent_runtime.director.models import DirectorContext, ProviderResult, ToolCall
from agent_runtime.director.resolver import ReferenceResolver


def _context(scene_state: dict[str, object]) -> DirectorContext:
    return DirectorContext(
        user_prompt="fix spacing",
        scene_state=scene_state,
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode="vectra-dev",
        memory_results=[],
    )


def test_reference_resolver_uses_active_object_for_vague_pronouns() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": "Cube_1",
                "selected_objects": ["Cube_1"],
                "objects": [{"name": "Cube_1", "location": [0.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]}],
            }
        )
    )

    result = resolver.resolve_target("it")

    assert result.value == "Cube_1"
    assert result.assumptions[0].reason.startswith("Used the active object")


def test_reference_resolver_uses_scene_centroid_or_origin_for_missing_location() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": None,
                "selected_objects": [],
                "objects": [
                    {"name": "Cube_1", "location": [2.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]},
                    {"name": "Cube_2", "location": [4.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]},
                ],
            }
        )
    )

    result = resolver.resolve_location("mesh.create_primitive", None)

    assert result.value == [3.0, 0.0, 0.0]
    assert result.metadata["anchor"] == "scene_centroid"


def test_director_loop_records_assumptions_instead_of_failing_on_missing_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: __import__("agent_runtime.director.models", fromlist=["ControllerDecision"]).ControllerDecision(),
    )
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools: ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            assistant_text="I'll spread the cubes apart.",
            tool_calls=[ToolCall(name="object.transform", arguments={"delta": [3.0, 0.0, 0.0]})],
        ),
    )

    turn = DirectorLoop().step(
        _context(
            {
                "active_object": "Cube_1",
                "selected_objects": ["Cube_1"],
                "objects": [{"name": "Cube_1", "location": [0.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]}],
            }
        )
    )

    assert turn.status == "ok"
    assert turn.metadata["actions"][0]["params"]["target"] == "Cube_1"
    assert turn.assumptions
