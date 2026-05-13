from __future__ import annotations

import math
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, normalize_optional_string, validate_vector3
from .registry import register_tool


def _require_bpy() -> Any:
    if bpy is None:
        raise ToolExecutionError("Blender Python API is unavailable")
    return bpy


def _material(name: str, color: tuple[float, float, float], *, roughness: float = 0.7) -> Any:
    api = _require_bpy()
    material = api.data.materials.get(name)
    if material is None:
        material = api.data.materials.new(name=name)
    material.diffuse_color = (color[0], color[1], color[2], 1.0)
    material.use_nodes = True
    principled = next(
        (node for node in material.node_tree.nodes if getattr(node, "type", "") == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is not None:
        principled.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        principled.inputs["Roughness"].default_value = roughness
    return material


def _cube(
    context: Any,
    *,
    name: str,
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
    obj["vectra_role"] = "composition"
    obj["semantic_role"] = "composition"
    return obj


def _output(objects: list[Any], message: str) -> ToolExecutionResult:
    names = [obj.name for obj in objects]
    return ToolExecutionResult(
        outputs={"object_name": names[0] if names else "", "object_names": names},
        message=message,
    )


@register_tool
class BuildRoomShellTool(BaseTool):
    name = "scene.build_room_shell"
    description = "Build a generic floor and surrounding wall shell for a composed interior or vignette."
    input_schema = {
        "name": {"type": "string", "required": False},
        "width": {"type": "number", "required": False},
        "depth": {"type": "number", "required": False},
        "height": {"type": "number", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        width = float(params.get("width", 8.0))
        depth = float(params.get("depth", 6.0))
        height = float(params.get("height", 3.2))
        if width <= 1.0 or depth <= 1.0 or height <= 1.0:
            raise ToolValidationError("Room shell width, depth, and height must be greater than 1.0")
        return {
            "name": normalize_optional_string(params.get("name", "VectraRoom"), "name") or "VectraRoom",
            "width": width,
            "depth": depth,
            "height": height,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        self.validate_params(params)
        ensure_object_mode(context)
        values = self.validate_params(params)
        name = values["name"]
        width = values["width"]
        depth = values["depth"]
        height = values["height"]
        floor_material = _material("Vectra_Composition_Floor", (0.18, 0.18, 0.17), roughness=0.85)
        wall_material = _material("Vectra_Composition_Walls", (0.32, 0.34, 0.34), roughness=0.8)
        objects = [
            _cube(
                context,
                name=f"{name}_Floor",
                location=(0.0, 0.0, -0.05),
                dimensions=(width, depth, 0.1),
                material=floor_material,
            ),
            _cube(
                context,
                name=f"{name}_BackWall",
                location=(0.0, depth / 2.0, height / 2.0),
                dimensions=(width, 0.12, height),
                material=wall_material,
            ),
            _cube(
                context,
                name=f"{name}_LeftWall",
                location=(-width / 2.0, 0.0, height / 2.0),
                dimensions=(0.12, depth, height),
                material=wall_material,
            ),
            _cube(
                context,
                name=f"{name}_RightWall",
                location=(width / 2.0, 0.0, height / 2.0),
                dimensions=(0.12, depth, height),
                material=wall_material,
            ),
        ]
        return _output(objects, "Built a structured room shell")


@register_tool
class BuildFocalFurnitureTool(BaseTool):
    name = "scene.build_focal_furniture"
    description = "Build a generic multi-part focal furniture/decor object from aligned primitive parts."
    input_schema = {
        "name": {"type": "string", "required": False},
        "style": {"type": "string", "required": False, "enum": ["sofa", "console", "sculpture"]},
        "location": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        style = str(params.get("style", "sofa")).strip().lower() or "sofa"
        if style not in {"sofa", "console", "sculpture"}:
            raise ToolValidationError("'style' must be one of sofa, console, or sculpture")
        return {
            "name": normalize_optional_string(params.get("name", "FocalFurniture"), "name") or "FocalFurniture",
            "style": style,
            "location": validate_vector3(params.get("location", (0.0, -0.35, 0.0)), "location"),
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        ensure_object_mode(context)
        values = self.validate_params(params)
        name = values["name"]
        x, y, z = values["location"]
        body = _material("Vectra_Focal_Body", (0.08, 0.16, 0.22), roughness=0.55)
        accent = _material("Vectra_Focal_Accent", (0.95, 0.62, 0.16), roughness=0.45)
        dark = _material("Vectra_Focal_Dark", (0.03, 0.035, 0.04), roughness=0.75)
        style = values["style"]
        if style == "console":
            objects = [
                _cube(context, name=f"{name}_Top", location=(x, y, z + 0.82), dimensions=(2.4, 0.62, 0.16), material=body),
                _cube(context, name=f"{name}_LeftLeg", location=(x - 0.92, y, z + 0.38), dimensions=(0.18, 0.18, 0.76), material=dark),
                _cube(context, name=f"{name}_RightLeg", location=(x + 0.92, y, z + 0.38), dimensions=(0.18, 0.18, 0.76), material=dark),
                _cube(context, name=f"{name}_AccentPanel", location=(x, y - 0.34, z + 0.62), dimensions=(1.15, 0.08, 0.34), material=accent),
            ]
        elif style == "sculpture":
            objects = [
                _cube(context, name=f"{name}_Plinth", location=(x, y, z + 0.25), dimensions=(1.2, 1.2, 0.5), material=dark),
                _cube(context, name=f"{name}_ColumnA", location=(x - 0.25, y, z + 0.9), dimensions=(0.22, 0.22, 1.25), material=body, rotation=(0.0, 0.0, math.radians(12))),
                _cube(context, name=f"{name}_ColumnB", location=(x + 0.25, y, z + 1.05), dimensions=(0.2, 0.2, 1.55), material=accent, rotation=(0.0, 0.0, math.radians(-15))),
            ]
        else:
            objects = [
                _cube(context, name=f"{name}_Seat", location=(x, y, z + 0.16), dimensions=(2.6, 0.92, 0.32), material=body),
                _cube(context, name=f"{name}_Back", location=(x, y + 0.42, z + 0.9), dimensions=(2.75, 0.2, 1.0), material=body),
                _cube(context, name=f"{name}_LeftArm", location=(x - 1.45, y, z + 0.36), dimensions=(0.22, 0.98, 0.72), material=dark),
                _cube(context, name=f"{name}_RightArm", location=(x + 1.45, y, z + 0.36), dimensions=(0.22, 0.98, 0.72), material=dark),
                _cube(context, name=f"{name}_AccentBlock", location=(x - 0.45, y - 0.53, z + 0.15), dimensions=(0.55, 0.08, 0.3), material=accent),
            ]
        return _output(objects, f"Built a multi-part focal {style}")
