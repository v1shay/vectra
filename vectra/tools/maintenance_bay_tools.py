from __future__ import annotations

import math
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult
from .helpers import ensure_object_mode, look_at_rotation, resolve_object, validate_vector3
from .registry import register_tool


def _require_bpy() -> Any:
    if bpy is None:
        raise ToolExecutionError("Blender Python API is unavailable")
    return bpy


def _ensure_collection(name: str) -> Any:
    api = _require_bpy()
    collection = api.data.collections.get(name)
    if collection is None:
        collection = api.data.collections.new(name)
        api.context.scene.collection.children.link(collection)
    return collection


def _material(name: str, color: tuple[float, float, float], *, metallic: float = 0.0, roughness: float = 0.65) -> Any:
    api = _require_bpy()
    material = api.data.materials.get(name)
    if material is None:
        material = api.data.materials.new(name=name)
    material.use_nodes = True
    principled = next(
        (node for node in material.node_tree.nodes if getattr(node, "type", "") == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is not None:
        principled.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = roughness
    return material


def _tag(obj: Any, role: str, *, benchmark: str = "maintenance_bay") -> None:
    obj["vectra_role"] = role
    obj["semantic_role"] = role
    obj["vectra_benchmark"] = benchmark


def _link_to_collection(obj: Any, collection_name: str = "Vectra_MaintenanceBay") -> None:
    collection = _ensure_collection(collection_name)
    if not any(existing == obj for existing in collection.objects):
        collection.objects.link(obj)
    for linked_collection in list(obj.users_collection):
        if linked_collection != collection:
            linked_collection.objects.unlink(obj)


def _cube(
    context: Any,
    *,
    name: str,
    role: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: Any,
    rotation: tuple[float, float, float] | None = None,
) -> Any:
    api = _require_bpy()
    existing = api.data.objects.get(name)
    if existing is None:
        result = api.ops.mesh.primitive_cube_add(size=1.0, location=location)
        if isinstance(result, set) and "FINISHED" not in result:
            raise ToolExecutionError(f"Failed to create cube '{name}': {result}")
        obj = api.context.active_object
        if obj is None:
            raise ToolExecutionError(f"Cube creation did not return object '{name}'")
        obj.name = name
    else:
        obj = existing
        obj.location = location
    obj.dimensions = dimensions
    if rotation is not None:
        obj.rotation_euler = rotation
    obj.data.materials.clear()
    obj.data.materials.append(material)
    _tag(obj, role)
    _link_to_collection(obj)
    scene = getattr(context, "scene", None) or api.context.scene
    if scene is not None:
        scene.update()
    return obj


def _output(objects: list[Any], message: str) -> ToolExecutionResult:
    names = [obj.name for obj in objects]
    return ToolExecutionResult(outputs={"object_names": names, "object_name": names[0] if names else ""}, message=message)


@register_tool
class BuildFloorTool(BaseTool):
    name = "skill.build_floor"
    description = "Build the deterministic Z-up maintenance-bay floor slab."
    input_schema = {
        "name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
        "dimensions": {"type": "vector3", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        location = validate_vector3(params.get("location", (0.0, 0.0, -0.05)), "location")
        dimensions = validate_vector3(params.get("dimensions", (10.0, 6.0, 0.1)), "dimensions")
        if dimensions[2] > min(dimensions[0], dimensions[1]) * 0.2:
            raise ToolExecutionError("Floor must be thin on Blender Z, not X or Y")
        floor = _cube(
            context,
            name=str(params.get("name") or "MaintenanceBay_Floor"),
            role="floor",
            location=location,
            dimensions=dimensions,
            material=_material("Vectra_Floor_DarkRubber", (0.08, 0.09, 0.09), roughness=0.9),
        )
        return _output([floor], "Built horizontal Z-up maintenance-bay floor")


@register_tool
class BuildRaisedCatwalkTool(BaseTool):
    name = "skill.build_raised_catwalk"
    description = "Build a raised catwalk and rails above the maintenance bay."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        deck_material = _material("Vectra_Catwalk_Gunmetal", (0.20, 0.23, 0.24), metallic=0.25)
        rail_material = _material("Vectra_Catwalk_Rails", (0.35, 0.38, 0.36), metallic=0.35)
        objects = [
            _cube(
                context,
                name="MaintenanceBay_Catwalk",
                role="catwalk",
                location=(0.0, 0.0, 2.2),
                dimensions=(7.2, 1.2, 0.18),
                material=deck_material,
            ),
            _cube(
                context,
                name="MaintenanceBay_Catwalk_Rail_North",
                role="catwalk",
                location=(0.0, 0.72, 2.55),
                dimensions=(7.2, 0.08, 0.42),
                material=rail_material,
            ),
            _cube(
                context,
                name="MaintenanceBay_Catwalk_Rail_South",
                role="catwalk",
                location=(0.0, -0.72, 2.55),
                dimensions=(7.2, 0.08, 0.42),
                material=rail_material,
            ),
        ]
        return _output(objects, "Built raised catwalk with rails")


@register_tool
class BuildWorkstationRowTool(BaseTool):
    name = "skill.build_workstation_row"
    description = "Build three deterministic workstations beneath the raised catwalk."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        body_material = _material("Vectra_Workstation_Body", (0.12, 0.16, 0.18), metallic=0.15)
        screen_material = _material("Vectra_Workstation_Glow", (0.05, 0.55, 0.65), roughness=0.25)
        objects: list[Any] = []
        for index, x_location in enumerate((-2.4, 0.0, 2.4), start=1):
            objects.append(
                _cube(
                    context,
                    name=f"MaintenanceBay_Workstation_{index}",
                    role="workstation",
                    location=(x_location, -0.25, 0.45),
                    dimensions=(0.82, 0.58, 0.9),
                    material=body_material,
                )
            )
            objects.append(
                _cube(
                    context,
                    name=f"MaintenanceBay_Workstation_{index}_Panel",
                    role="workstation",
                    location=(x_location, -0.58, 0.88),
                    dimensions=(0.58, 0.05, 0.32),
                    material=screen_material,
                    rotation=(math.radians(10.0), 0.0, 0.0),
                )
            )
        return _output(objects, "Built three maintenance workstations beneath the catwalk")


@register_tool
class BuildCableBundleTool(BaseTool):
    name = "skill.build_cable_bundle"
    description = "Route deterministic cable bundles across the maintenance bay floor."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        material = _material("Vectra_Cable_Black", (0.01, 0.01, 0.012), roughness=0.8)
        objects = [
            _cube(
                context,
                name="MaintenanceBay_CableBundle_Main",
                role="cable",
                location=(0.0, -1.2, 0.06),
                dimensions=(6.2, 0.08, 0.08),
                material=material,
            ),
            _cube(
                context,
                name="MaintenanceBay_CableBundle_Return",
                role="cable",
                location=(0.0, 1.15, 0.06),
                dimensions=(4.8, 0.06, 0.06),
                material=material,
            ),
        ]
        return _output(objects, "Routed visible cable bundles")


@register_tool
class BuildHazardStripesTool(BaseTool):
    name = "skill.build_hazard_stripes"
    description = "Build deterministic yellow hazard stripe markings along bay edges."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        material = _material("Vectra_Hazard_Yellow", (1.0, 0.72, 0.05), roughness=0.55)
        objects: list[Any] = []
        for index, x_location in enumerate((-3.2, -1.6, 0.0, 1.6, 3.2), start=1):
            objects.append(
                _cube(
                    context,
                    name=f"MaintenanceBay_HazardStripe_Floor_{index}",
                    role="hazard_stripe",
                    location=(x_location, -2.15, 0.025),
                    dimensions=(0.95, 0.08, 0.035),
                    material=material,
                    rotation=(0.0, 0.0, math.radians(25.0)),
                )
            )
        for index, x_location in enumerate((-2.5, 0.0, 2.5), start=1):
            objects.append(
                _cube(
                    context,
                    name=f"MaintenanceBay_HazardStripe_Catwalk_{index}",
                    role="hazard_stripe",
                    location=(x_location, -0.62, 2.32),
                    dimensions=(0.85, 0.06, 0.035),
                    material=material,
                    rotation=(0.0, 0.0, math.radians(-25.0)),
                )
            )
        return _output(objects, "Added floor and catwalk hazard stripes")


@register_tool
class BuildOverheadLightsTool(BaseTool):
    name = "skill.build_overhead_lights"
    description = "Place deterministic overhead area lights for the maintenance bay."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        api = _require_bpy()
        self.validate_params(params)
        ensure_object_mode(context)
        objects: list[Any] = []
        for index, x_location in enumerate((-2.5, 0.0, 2.5), start=1):
            name = f"MaintenanceBay_OverheadLight_{index}"
            light = api.data.objects.get(name)
            if light is None:
                light_data = api.data.lights.new(name=f"{name}_Data", type="AREA")
                light = api.data.objects.new(name, light_data)
                api.context.scene.collection.objects.link(light)
            light.location = (x_location, -1.1, 4.2)
            light.rotation_euler = (math.radians(70.0), 0.0, 0.0)
            light.data.energy = 650.0
            light.data.size = 1.4
            _tag(light, "overhead_light")
            _link_to_collection(light)
            objects.append(light)
        return _output(objects, "Placed overhead area lights")


@register_tool
class FrameCorridorCameraTool(BaseTool):
    name = "skill.frame_corridor_camera"
    description = "Create or adjust the camera to frame the maintenance-bay corridor."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        api = _require_bpy()
        self.validate_params(params)
        ensure_object_mode(context)
        camera = getattr(context.scene, "camera", None)
        if camera is None:
            data = api.data.cameras.new("MaintenanceBay_Camera_Data")
            camera = api.data.objects.new("MaintenanceBay_Camera", data)
            api.context.scene.collection.objects.link(camera)
            context.scene.camera = camera
        camera.name = "MaintenanceBay_Camera"
        camera.location = (6.2, -6.0, 3.1)
        target = resolve_object(context, "MaintenanceBay_Catwalk") or resolve_object(context, "MaintenanceBay_Workstation_2")
        target_location = list(getattr(target, "location", (0.0, 0.0, 1.1))) if target is not None else [0.0, 0.0, 1.1]
        camera.rotation_euler = look_at_rotation(list(camera.location), target_location)
        camera.data.lens = 28.0
        _tag(camera, "camera")
        _link_to_collection(camera)
        return _output([camera], "Framed corridor camera on the maintenance bay")
