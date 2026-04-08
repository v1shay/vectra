from __future__ import annotations

import queue
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vectra.tools.registry as registry_module


class FakeScenes(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakeTimers:
    def __init__(self) -> None:
        self._registered: set[object] = set()

    def is_registered(self, callback) -> bool:
        return callback in self._registered

    def register(self, callback, first_interval=None):
        del first_interval
        self._registered.add(callback)

    def unregister(self, callback):
        self._registered.discard(callback)


class FakeUtils:
    def __init__(self, bpy_types) -> None:
        self._bpy_types = bpy_types
        self.registered_names: set[str] = set()

    def register_class(self, cls) -> None:
        identifier = getattr(cls, "bl_idname", cls.__name__)
        if identifier in self.registered_names:
            raise ValueError(
                f"register_class(...): already registered as a subclass '{cls.__name__}'"
            )
        self.registered_names.add(identifier)
        setattr(self._bpy_types, cls.__name__, cls)
        self._bpy_types.__dict__.setdefault("_subtype_registry", {})
        self._bpy_types._subtype_registry[cls.__name__] = cls
        bl_idname = getattr(cls, "bl_idname", None)
        if isinstance(bl_idname, str) and bl_idname:
            self._bpy_types._subtype_registry[bl_idname] = cls

    def unregister_class(self, cls) -> None:
        identifier = getattr(cls, "bl_idname", cls.__name__)
        if identifier not in self.registered_names:
            raise RuntimeError("not registered")
        self.registered_names.remove(identifier)
        existing = getattr(self._bpy_types, cls.__name__, None)
        if existing is cls:
            delattr(self._bpy_types, cls.__name__)
        subtype_registry = self._bpy_types.__dict__.setdefault("_subtype_registry", {})
        for candidate_id in (cls.__name__, getattr(cls, "bl_idname", None)):
            if isinstance(candidate_id, str) and subtype_registry.get(candidate_id) is cls:
                del subtype_registry[candidate_id]


def _make_fake_bpy() -> ModuleType:
    fake_bpy = ModuleType("bpy")
    subtype_registry: dict[str, type[object]] = {}

    def _finder(identifier: str, default=None):
        return subtype_registry.get(identifier, default)

    scene_type = type("Scene", (), {})
    addon_preferences_type = type(
        "AddonPreferences",
        (),
        {"bl_rna_get_subclass_py": staticmethod(_finder)},
    )
    operator_type = type(
        "Operator",
        (),
        {"bl_rna_get_subclass_py": staticmethod(_finder)},
    )
    panel_type = type(
        "Panel",
        (),
        {"bl_rna_get_subclass_py": staticmethod(_finder)},
    )
    fake_bpy.types = SimpleNamespace(
        Scene=scene_type,
        Context=type("Context", (), {}),
        Operator=operator_type,
        Panel=panel_type,
        AddonPreferences=addon_preferences_type,
    )
    fake_bpy.types._subtype_registry = subtype_registry
    fake_bpy.props = SimpleNamespace(
        StringProperty=lambda **kwargs: ("StringProperty", kwargs),
        BoolProperty=lambda **kwargs: ("BoolProperty", kwargs),
        EnumProperty=lambda **kwargs: ("EnumProperty", kwargs),
        IntProperty=lambda **kwargs: ("IntProperty", kwargs),
    )
    fake_bpy.utils = FakeUtils(fake_bpy.types)
    fake_bpy._subtype_registry = subtype_registry
    fake_bpy.app = SimpleNamespace(timers=FakeTimers())
    fake_bpy.data = SimpleNamespace(scenes=FakeScenes())
    fake_bpy.context = SimpleNamespace(
        scene=None,
        active_object=None,
        selected_objects=[],
        preferences=SimpleNamespace(addons={}),
    )
    return fake_bpy


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_addon_runtime_register_unregister_cycle_is_reload_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    scene = fake_bpy.types.Scene()
    scene.vectra_request_in_flight = True
    scene.vectra_phase = "sending"
    scene.vectra_status = "Sending request..."
    fake_bpy.context.scene = scene
    fake_bpy.data.scenes["Scene"] = scene

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    for module_name in (
        "vectra.addon_runtime",
        "vectra.addon_bootstrap",
        "vectra.operators.run_task",
        "vectra.ui.panel",
    ):
        sys.modules.pop(module_name, None)

    runtime_module = _reload_module("vectra.addon_runtime")
    run_task_module = sys.modules["vectra.operators.run_task"]

    runtime_module.register()
    runtime_module.register()

    assert fake_bpy.utils.registered_names == {"vectra.run_task", "VECTRA_PT_panel"}
    for attribute_name in (
        "vectra_prompt",
        "vectra_status",
        "vectra_phase",
        "vectra_runtime_state",
        "vectra_request_in_flight",
        "vectra_execution_mode",
        "vectra_agent_transcript",
        "vectra_pending_question",
        "vectra_iteration",
    ):
        assert hasattr(fake_bpy.types.Scene, attribute_name)

    registry = registry_module.get_default_registry()
    assert "mesh.create_primitive" in registry.list_tools()
    assert "object.transform" in registry.list_tools()

    run_task_module._request_scene_name = "Scene"
    run_task_module._request_queue = object()
    run_task_module._request_thread = None
    run_task_module._execution_engine = object()
    fake_bpy.app.timers.register(run_task_module._poll_request_result)

    runtime_module.unregister()

    assert fake_bpy.utils.registered_names == set()
    for attribute_name in (
        "vectra_prompt",
        "vectra_status",
        "vectra_phase",
        "vectra_runtime_state",
        "vectra_request_in_flight",
        "vectra_execution_mode",
        "vectra_agent_transcript",
        "vectra_pending_question",
        "vectra_iteration",
    ):
        assert not hasattr(fake_bpy.types.Scene, attribute_name)
    assert registry_module.get_default_registry().list_tools() == []
    assert run_task_module._request_queue is None
    assert run_task_module._request_thread is None
    assert run_task_module._execution_engine is None
    assert fake_bpy.app.timers.is_registered(run_task_module._poll_request_result) is False
    assert scene.vectra_request_in_flight is False
    assert scene.vectra_phase == "idle"
    assert scene.vectra_status == "Idle"


def test_addon_runtime_register_handles_cold_scene_without_vectra_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    fake_bpy.context.scene = fake_bpy.types.Scene()

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    for module_name in (
        "vectra.addon_runtime",
        "vectra.operators.run_task",
        "vectra.ui.panel",
    ):
        sys.modules.pop(module_name, None)

    runtime_module = _reload_module("vectra.addon_runtime")

    runtime_module.register()

    assert hasattr(fake_bpy.types.Scene, "vectra_prompt")
    assert hasattr(fake_bpy.types.Scene, "vectra_status")


def test_run_task_blocks_reload_while_worker_thread_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.operators.run_task", None)
    run_task_module = _reload_module("vectra.operators.run_task")

    run_task_module._request_thread = SimpleNamespace(is_alive=lambda: True)
    run_task_module._request_queue = None

    assert run_task_module.get_reload_block_reason() == (
        "Cannot reload Vectra while a request worker thread is still running"
    )


def test_run_task_marks_empty_error_response_as_error_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    scene = fake_bpy.types.Scene()
    scene.name = "Scene"
    scene.vectra_request_in_flight = True
    scene.vectra_phase = "sending"
    scene.vectra_status = "Sending request..."
    fake_bpy.context.scene = scene
    fake_bpy.data.scenes["Scene"] = scene

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.operators.run_task", None)
    run_task_module = _reload_module("vectra.operators.run_task")

    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    result_queue.put(
        (
            "success",
            {
                "status": "error",
                "message": "No actions returned: invalid request",
                "actions": [],
            },
        )
    )

    run_task_module._request_queue = result_queue
    run_task_module._request_scene_name = "Scene"
    run_task_module._request_thread = None

    result = run_task_module._poll_request_result()

    assert result is None
    assert scene.vectra_request_in_flight is False
    assert scene.vectra_phase == "error"
    assert scene.vectra_status == "No actions returned: invalid request"


def test_agent_result_surfaces_runtime_error_state_in_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    scene = fake_bpy.types.Scene()
    scene.name = "Scene"
    scene.vectra_request_in_flight = True
    scene.vectra_phase = "working"
    scene.vectra_status = "Awaiting model response..."
    scene.vectra_runtime_state = "awaiting_model_response"
    scene.vectra_agent_transcript = ""
    scene.vectra_pending_question = ""
    scene.vectra_history_json = "[]"
    scene.vectra_iteration = 1
    fake_bpy.context.scene = scene
    fake_bpy.data.scenes["Scene"] = scene

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.operators.run_task", None)
    run_task_module = _reload_module("vectra.operators.run_task")
    run_task_module._agent_loop_state = run_task_module.AgentLoopState(
        prompt="make something cool",
        execution_mode="vectra-dev",
    )

    result = run_task_module._handle_agent_result(
        scene,
        {
            "status": "error",
            "message": "Provider transport failed",
            "narration": "Provider transport failed",
            "assumptions": [],
            "metadata": {
                "runtime_state": "provider_transport_failure",
                "runtime_state_detail": "Provider transport failed",
            },
        },
    )

    assert result is None
    assert scene.vectra_phase == "error"
    assert scene.vectra_status == "Provider transport failed"
    assert scene.vectra_runtime_state == "provider_transport_failure"


def test_bootstrap_register_replaces_stale_preferences_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.addon_bootstrap", None)
    bootstrap_module = _reload_module("vectra.addon_bootstrap")

    stale_preferences_class = type("VectraAddonPreferences", (), {"bl_idname": "vectra"})
    fake_bpy.utils.registered_names.add("vectra")
    fake_bpy._subtype_registry["VectraAddonPreferences"] = stale_preferences_class

    bootstrap_module.register_bootstrap_classes()

    assert fake_bpy.types.VectraAddonPreferences is bootstrap_module.VectraAddonPreferences
    assert "vectra" in fake_bpy.utils.registered_names
    assert "vectra.reload_dev" in fake_bpy.utils.registered_names


def test_bootstrap_package_name_matches_full_dotted_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    module_name = "bl_ext.user_default.vectra.addon_bootstrap"
    module = ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    module.__dict__["__package__"] = "bl_ext.user_default.vectra"
    module.__dict__["bpy"] = fake_bpy
    module.__dict__["addon_loader"] = SimpleNamespace(get_runtime_status=lambda: None)

    source = (
        "ADDON_PACKAGE_NAME = __package__ or __name__.rsplit('.', 1)[0]\n"
    )
    exec(source, module.__dict__)

    assert module.__dict__["ADDON_PACKAGE_NAME"] == "bl_ext.user_default.vectra"


def test_current_dev_source_path_ignores_deferred_property_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    deferred_property = object()
    addon_entry = SimpleNamespace(preferences=SimpleNamespace(dev_source_path=deferred_property))
    fake_bpy.context.preferences.addons["vectra"] = addon_entry
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.addon_bootstrap", None)
    bootstrap_module = _reload_module("vectra.addon_bootstrap")
    bootstrap_module.ADDON_PACKAGE_NAME = "vectra"

    assert bootstrap_module.current_dev_source_path() is None


def test_current_dev_source_path_falls_back_to_short_addon_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    addon_entry = SimpleNamespace(preferences=SimpleNamespace(dev_source_path="/tmp/vectra"))
    fake_bpy.context.preferences.addons["vectra"] = addon_entry
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.addon_bootstrap", None)
    bootstrap_module = _reload_module("vectra.addon_bootstrap")
    bootstrap_module.ADDON_PACKAGE_NAME = "bl_ext.user_default.vectra"

    assert bootstrap_module.current_dev_source_path() == "/tmp/vectra"


def test_partial_structural_progress_continues_agent_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    scene = fake_bpy.types.Scene()
    scene.name = "Scene"
    scene.vectra_request_in_flight = True
    scene.vectra_phase = "working"
    scene.vectra_status = "Working"
    scene.vectra_agent_transcript = ""
    scene.vectra_runtime_state = "valid_action_batch_ready"
    fake_bpy.context.scene = scene
    fake_bpy.data.scenes["Scene"] = scene

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.operators.run_task", None)
    run_task_module = _reload_module("vectra.operators.run_task")
    run_task_module._agent_loop_state = run_task_module.AgentLoopState(
        prompt="make a room",
        execution_mode="vectra-dev",
        iteration=1,
        ineffective_turns=2,
    )

    started: list[str] = []
    monkeypatch.setattr(run_task_module, "_start_agent_iteration", lambda context: started.append("started"))
    monkeypatch.setattr(run_task_module, "_finalize_request", lambda: started.append("finalize"))

    result = run_task_module._maybe_continue_agent_loop(
        scene,
        {"status": "ok", "continue_loop": True},
        success=True,
        execution_payload={"actions": [], "metadata": {"action_families": ["create", "structure"]}},
        verification={
            "meaningful_change": True,
            "structural_progress": True,
            "low_progress": True,
            "progress_score": 1.2,
            "summary": "Created floor and back wall.",
        },
    )

    assert result == 0.1
    assert run_task_module._agent_loop_state.ineffective_turns == 0
    assert started == ["started"]
    assert "Partial structural progress counted" in scene.vectra_agent_transcript


def test_repeated_no_progress_still_stops_agent_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bpy = _make_fake_bpy()
    scene = fake_bpy.types.Scene()
    scene.name = "Scene"
    scene.vectra_request_in_flight = True
    scene.vectra_phase = "working"
    scene.vectra_status = "Working"
    scene.vectra_agent_transcript = ""
    scene.vectra_runtime_state = "valid_action_batch_ready"
    fake_bpy.context.scene = scene
    fake_bpy.data.scenes["Scene"] = scene

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("vectra.operators.run_task", None)
    run_task_module = _reload_module("vectra.operators.run_task")
    run_task_module._agent_loop_state = run_task_module.AgentLoopState(
        prompt="make a room",
        execution_mode="vectra-dev",
        iteration=3,
        ineffective_turns=2,
        last_action_families=["layout"],
    )

    finalized: list[str] = []
    monkeypatch.setattr(run_task_module, "_finalize_request", lambda: finalized.append("done"))

    result = run_task_module._maybe_continue_agent_loop(
        scene,
        {"status": "ok", "continue_loop": True},
        success=True,
        execution_payload={"actions": [], "metadata": {"action_families": ["layout"]}},
        verification={
            "meaningful_change": False,
            "structural_progress": False,
            "low_progress": False,
            "progress_score": 0.0,
            "summary": "No scene changes were detected.",
        },
    )

    assert result is None
    assert finalized == ["done"]
    assert scene.vectra_phase == "error"
    assert scene.vectra_status == "Stopping after repeated ineffective turns"
