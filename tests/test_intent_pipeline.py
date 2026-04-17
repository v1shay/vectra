from __future__ import annotations

from agent_runtime.director.loop import DirectorLoop
from agent_runtime.director.models import DirectorContext, ParsedProviderResponse, ProviderResult, ToolCall
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


def test_reference_resolver_grounds_new_primitives_on_floor_anchor() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": None,
                "selected_objects": [],
                "objects": [
                    {"name": "Floor", "type": "MESH", "location": [0.0, 0.0, 0.0], "dimensions": [12.0, 12.0, 0.0]},
                    {"name": "Cube_1", "type": "MESH", "location": [1.0, 0.0, 1.0], "dimensions": [2.0, 2.0, 2.0]},
                    {"name": "Cube_2", "type": "MESH", "location": [3.0, 0.0, 1.0], "dimensions": [2.0, 2.0, 2.0]},
                ],
            }
        )
    )

    result = resolver.resolve_location("mesh.create_primitive", None, primitive_type="cube")

    assert result.value == [0.0, 0.0, 1.0]
    assert result.metadata["anchor"] == "Floor"


def test_reference_resolver_does_not_chain_new_primitive_to_active_object() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": "Cube_1",
                "selected_objects": ["Cube_1"],
                "objects": [
                    {"name": "Floor", "type": "MESH", "location": [0.0, 0.0, 0.0], "dimensions": [12.0, 12.0, 0.0]},
                    {"name": "Cube_1", "type": "MESH", "location": [0.0, 0.0, 1.0], "dimensions": [2.0, 2.0, 2.0]},
                    {"name": "FarWall", "type": "MESH", "location": [6.0, 0.0, 2.0], "dimensions": [0.2, 8.0, 4.0]},
                ],
            }
        )
    )

    result = resolver.resolve_location("mesh.create_primitive", None, primitive_type="cube")

    assert result.value == [0.0, 0.0, 1.0]
    assert result.metadata["anchor"] == "Floor"


def test_reference_resolver_preserves_validated_tuple_location() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": None,
                "selected_objects": [],
                "objects": [
                    {"name": "Floor", "type": "MESH", "location": [0.0, 0.0, 0.0], "dimensions": [12.0, 12.0, 0.0]},
                ],
            }
        )
    )

    result = resolver.resolve_location("mesh.create_primitive", (4.5, -2.0, 1.0), primitive_type="cube")

    assert result.value == [4.5, -2.0, 1.0]
    assert result.metadata["anchor"] == "explicit"


def test_reference_resolver_required_spatial_target_does_not_fallback_to_active_object() -> None:
    resolver = ReferenceResolver(
        _context(
            {
                "active_object": "Cube_1",
                "selected_objects": ["Cube_1"],
                "objects": [
                    {"name": "Cube_1", "location": [0.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]},
                    {"name": "Cube_2", "location": [3.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]},
                ],
            }
        )
    )

    result = resolver.resolve_required_target("it", "target")

    assert result.value is None
    assert result.metadata["anchor"] == "unresolved_required"
    assert "no fallback target" in result.assumptions[0].reason


def test_director_loop_records_assumptions_instead_of_failing_on_missing_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: __import__("agent_runtime.director.models", fromlist=["ControllerDecision"]).ControllerDecision(),
    )
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I'll spread the cubes apart.",
                tool_calls=[ToolCall(name="object.transform", arguments={"delta": [3.0, 0.0, 0.0]})],
                response_type="tool_calls",
            ),
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


def test_director_loop_supports_batched_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: __import__("agent_runtime.director.models", fromlist=["ControllerDecision"]).ControllerDecision(),
    )
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will create the base scene and add lighting in the same turn.",
                tool_calls=[
                    ToolCall(name="mesh.create_primitive", arguments={"type": "plane", "name": "Floor"}),
                    ToolCall(name="light.create", arguments={"type": "AREA"}),
                ],
                response_type="tool_calls",
            ),
        ),
    )

    turn = DirectorLoop().step(
        _context(
            {
                "active_object": None,
                "selected_objects": [],
                "objects": [],
                "lights": [],
                "groups": [],
                "scene_centroid": [0.0, 0.0, 0.0],
                "scene_bounds": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            }
        )
    )

    assert turn.status == "ok"
    assert turn.metadata["batch_size"] == 2
    assert len(turn.metadata["actions"]) == 2


def test_director_loop_resolves_same_batch_spatial_names_to_action_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: __import__("agent_runtime.director.models", fromlist=["ControllerDecision"]).ControllerDecision(),
    )
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will create two objects and place one on the other.",
                tool_calls=[
                    ToolCall(name="mesh.create_primitive", arguments={"type": "cube", "name": "Table"}),
                    ToolCall(name="mesh.create_primitive", arguments={"type": "cube", "name": "Lamp"}),
                    ToolCall(name="object.place_on_surface", arguments={"target": "Lamp", "reference": "Table", "surface": "top"}),
                ],
                response_type="tool_calls",
            ),
        ),
    )

    turn = DirectorLoop().step(
        _context(
            {
                "active_object": None,
                "selected_objects": [],
                "objects": [],
                "lights": [],
                "groups": [],
                "scene_centroid": [0.0, 0.0, 0.0],
                "scene_bounds": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            }
        )
    )

    assert turn.status == "ok"
    assert turn.metadata["actions"][2]["params"]["target"] == {"$ref": "step_2.object_name"}
    assert turn.metadata["actions"][2]["params"]["reference"] == {"$ref": "step_1.object_name"}


def test_director_loop_fails_visibly_for_future_spatial_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: __import__("agent_runtime.director.models", fromlist=["ControllerDecision"]).ControllerDecision(),
    )
    calls = {"count": 0}

    def _invalid_director_response(prompt_text, tools, allow_complete=False):
        del prompt_text, tools, allow_complete
        calls["count"] += 1
        return ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will place an object that does not exist yet.",
                tool_calls=[
                    ToolCall(name="object.place_on_surface", arguments={"target": "Lamp", "reference": "Table", "surface": "top"}),
                ],
                response_type="tool_calls",
            ),
        )

    monkeypatch.setattr("agent_runtime.director.loop.call_director", _invalid_director_response)

    turn = DirectorLoop().step(
        _context(
            {
                "active_object": "Cube_1",
                "selected_objects": ["Cube_1"],
                "objects": [{"name": "Cube_1", "location": [0.0, 0.0, 0.0], "dimensions": [2.0, 2.0, 2.0]}],
                "lights": [],
                "groups": [],
            }
        )
    )

    assert calls["count"] == 2
    assert turn.status == "error"
    assert turn.metadata["runtime_state"] == "tool_validation_failure"
    assert "Missing required param" in turn.metadata["failure_reason"]
