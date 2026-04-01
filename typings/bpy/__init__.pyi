from collections.abc import Iterator
from typing import Any, Callable


class Object:
    name: str
    type: str
    location: Any
    rotation_euler: Any
    scale: Any
    def select_get(self) -> bool: ...


class _SceneObjectCollection:
    def __iter__(self) -> Iterator[Object]: ...


class types:
    class Scene:
        name: str
        frame_current: int
        objects: _SceneObjectCollection
        vectra_prompt: str
        vectra_status: str
        vectra_phase: str
        vectra_request_in_flight: bool
        vectra_execution_mode: str
        vectra_agent_transcript: str
        vectra_pending_question: str
        vectra_iteration: int

    class Context:
        scene: "types.Scene"
        active_object: Object | None
        selected_objects: list[Object]
        view_layer: Any

    class Panel:
        layout: Any

    class Operator:
        bl_idname: str
        bl_label: str
        bl_description: str
        def report(self, categories: set[str], message: str) -> None: ...


class _SceneCollection:
    def get(self, name: str) -> types.Scene | None: ...


class _ObjectCollection:
    def get(self, name: str) -> Object | None: ...


class _Data:
    scenes: _SceneCollection
    objects: _ObjectCollection


data: _Data
context: types.Context


class _Props:
    def StringProperty(self, **kwargs: Any) -> Any: ...
    def BoolProperty(self, **kwargs: Any) -> Any: ...
    def EnumProperty(self, **kwargs: Any) -> Any: ...
    def IntProperty(self, **kwargs: Any) -> Any: ...


props: _Props


class _Utils:
    def register_class(self, cls: type[Any]) -> None: ...
    def unregister_class(self, cls: type[Any]) -> None: ...


utils: _Utils


class _Timers:
    def is_registered(self, callback: Callable[..., Any]) -> bool: ...
    def register(self, callback: Callable[..., Any], first_interval: float = ...) -> None: ...
    def unregister(self, callback: Callable[..., Any]) -> None: ...


class _App:
    timers: _Timers


app: _App


class _MeshOps:
    def primitive_cube_add(self, **kwargs: Any) -> set[str]: ...
    def primitive_plane_add(self, **kwargs: Any) -> set[str]: ...
    def primitive_uv_sphere_add(self, **kwargs: Any) -> set[str]: ...


class _PreferencesOps:
    def addon_install(self, **kwargs: Any) -> set[str]: ...
    def addon_enable(self, **kwargs: Any) -> set[str]: ...


class _VectraOps:
    def run_task(self, **kwargs: Any) -> set[str]: ...


class _Ops:
    mesh: _MeshOps
    preferences: _PreferencesOps
    vectra: _VectraOps


ops: _Ops
