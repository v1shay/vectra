from __future__ import annotations

CREATE_CUBE_ACTIONS = [
    {
        "action_id": "create_cube",
        "tool": "mesh.create_primitive",
        "params": {
            "primitive_type": "cube",
            "name": "VectraCube",
            "location": [0.0, 0.0, 0.0],
        },
    }
]

CREATE_TWO_CUBES_ACTIONS = [
    {
        "action_id": "create_cube_one",
        "tool": "mesh.create_primitive",
        "params": {
            "primitive_type": "cube",
            "name": "VectraCubeOne",
            "location": [0.0, 0.0, 0.0],
        },
    },
    {
        "action_id": "create_cube_two",
        "tool": "mesh.create_primitive",
        "params": {
            "primitive_type": "cube",
            "name": "VectraCubeTwo",
            "location": [2.0, 0.0, 0.0],
        },
    },
]

MOVE_CUBE_ACTIONS = [
    {
        "action_id": "move_cube",
        "tool": "object.transform",
        "params": {
            "object_name": "Cube",
            "location": [2.0, 0.0, 0.0],
        },
    }
]
