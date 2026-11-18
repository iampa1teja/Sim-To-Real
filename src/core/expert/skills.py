"""Skill descriptors shared by the parser and state machine."""

from dataclasses import dataclass as _dataclass
from collections.abc import Mapping as _Mapping
from core.tasks.base import _freeze_value


@_dataclass(frozen=True)
class Skill:
    name: str
    required: tuple[str, ...]
    defaults: _Mapping
    budget: int | None

    def __post_init__(self):
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "defaults", _freeze_value(self.defaults))


SKILLS = {name: Skill(name, required, defaults, budget) for name, required, defaults, budget in (
    ("move_joint", ("q",), {"tol_rad": 0.01}, 150),
    ("set_gripper", ("gap_m",), {"tol_m": 0.002}, 60),
    ("reach", ("target",), {"offset_m": (0.0, 0.0, 0.0), "orientation": "down", "yaw": "keep", "tol_m": 0.005}, 200),
    ("approach", ("target",), {"offset_m": (0.0, 0.0, 0.06), "orientation": "down", "yaw": "object", "tol_m": 0.005}, 200),
    ("grasp", ("target",), {"z_offset_m": 0.0, "close_gap_m": 0.0, "dwell_steps": 10}, 150),
    ("lift", (), {"height_m": 0.08}, 120),
    ("transport", ("target",), {"height_m": 0.06, "align_held": False}, 250),
    ("place", ("target",), {"offset_m": (0.0, 0.0, 0.0), "clearance_m": 0.005, "align_held": False}, 150),
    ("release", (), {"gap_m": "open", "dwell_steps": 10}, 60),
    ("retreat", (), {"dz_m": 0.08}, 120),
    ("push", ("target", "goal"), {"standoff_m": 0.03, "depth_m": 0.0}, 300),
    ("touch", ("target",), {"contact_tol_m": 0.004}, 200),
    ("sweep", ("joint", "angles"), {"tol_rad": 0.01}, 300),
    ("wait", ("steps",), {}, None),
)}
