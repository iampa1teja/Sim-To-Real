"""Synthetic task and scene folders (contracts §4.6, §13). Standard library only at import time."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from .toml_writer import dump_toml

#: ``__init__.py`` of a GenericTask folder (contracts §4.6); ``{name}`` is substituted.
GENERIC_INIT = (
    "from core.tasks.base import load_task_spec\n"
    "from core.tasks.generic import GenericTask\n\n"
    "TASK = GenericTask(load_task_spec({name!r}))\n"
)
#: ``env_cfg.py`` of a generic task folder (contracts §4.6).
GENERIC_ENV_CFG = "from sim.envs.generic_task_cfg import apply_teleop, apply_generate  # noqa: F401\n"
#: Minimal valid USDA text for stub object files (``ObjectSpec.from_toml`` only checks existence).
STUB_USDA = '#usda 1.0\n(\n    defaultPrim = "Root"\n)\n\ndef Xform "Root"\n{\n}\n'


def write_task(tasks_dir: str | Path, name: str, task_table: Mapping, *, init_py: str | None = None,
               env_cfg_py: str | None = GENERIC_ENV_CFG, files: Mapping[str, str | bytes] | None = None) -> Path:
    """Write ``tasks_dir/<name>/`` with ``task.toml`` (``[task]`` = ``task_table``), ``__init__.py`` and ``env_cfg.py``.

    ``task_table["name"]`` defaults to ``name``. ``init_py`` defaults to ``GENERIC_INIT`` (needs WP1's
    ``core.tasks.generic``); pass your own text for a custom ``Task`` subclass. ``env_cfg_py=None``
    omits ``env_cfg.py``. ``files`` maps relative paths (e.g. ``"usd/box.usda"``) to text or bytes.
    Use an ``fx_``-style unique name per content: ``load_task`` caches by name (``clear_task_cache``).
    """
    folder = Path(tasks_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    table = dict(task_table)
    table.setdefault("name", name)
    (folder / "task.toml").write_text(dump_toml({"task": table}))
    (folder / "__init__.py").write_text(init_py if init_py is not None else GENERIC_INIT.format(name=name))
    if env_cfg_py is not None:
        (folder / "env_cfg.py").write_text(env_cfg_py)
    for rel, content in (files or {}).items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return folder


def write_scene(scene_dir: str | Path, setup: Mapping, *, scene_table: Mapping | None = None) -> Path:
    """Write ``scene_dir/real_setup.json`` (and ``scene.toml`` when ``scene_table`` is given); return ``scene_dir``.

    Point ``S2R_SCENE_DIR`` at ``scene_dir`` itself (``core.scenes.scene_dir`` semantics); ``load_setup()``
    then returns ``setup``. Combine with ``fx_arms.fx_setup()`` for the per-arm ``"arms"`` block.
    """
    folder = Path(scene_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "real_setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    if scene_table is not None:
        (folder / "scene.toml").write_text(dump_toml({"scene": dict(scene_table)}))
    return folder


_REACH = {
    "variants": {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"},
    "instruction_template": {"texts": ["Move the gripper to the point"]},
    "start": {"goal": {"point": {"type": "point", "region": {"r_m": [0.15, 0.32], "bearing_deg": [-60.0, 60.0]},
                                 "z_m": [0.03, 0.20]}}},
    "stages": {"order": ["reached", "success"]},
    "checks": {"reached": {"check": "ee_near", "point": "@goal.point", "tol_m": 0.01},
               "success": {"check": "all", "of": ["reached"]}},
    "failures": {"rules": [{"name": "never_reached", "unless": ["reached"]}, {"name": "left_target"}]},
    "expert": {"steps": [{"skill": "move_joint", "q": "home"},
                         {"skill": "reach", "target": "@goal.point", "stage": "reached"},
                         {"skill": "wait", "steps": 10, "stage": "success"}]},
    "perturbations": {"failure": [{"type": "target_offset", "xy_m": [0.03, 0.08]},
                                  {"type": "stall", "stages": ["reached"], "steps": [600, 600]}],
                      "recovery": [{"type": "target_offset", "xy_m": [0.03, 0.08]}]},
}

_COLOR = {
    "default_variant": "teleop",
    "variants": {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"},
    "objects": {"small_box": {"usd": "usd/Small-Box-Physics.usda", "origin": "bottom", "size_outer": [0.05, 0.05, 0.035],
                              "inner_half_size": [0.02, 0.02], "floor_z": 0.002}},
    "object_sets": {"cubes": {"prefix": "cube", "count_max": 6, "size": [0.02, 0.02, 0.02], "mass": 0.008}},
    "palette": {"red": [0.8, 0.05, 0.05], "green": [0.05, 0.6, 0.1], "blue": [0.05, 0.1, 0.8],
                "yellow": [0.9, 0.8, 0.05], "orange": [0.95, 0.4, 0.02], "purple": [0.4, 0.1, 0.6],
                "pink": [0.95, 0.45, 0.65]},
    "roles": {"target": "cubes", "container": "small_box"},
    "holdout": {"colors": ["purple"], "counts": [4]},
    "instruction_template": {"texts": ["Pick up the {target.color} cube and place it in the box"]},
    "start": {"count": [2, 6], "min_separation_m": 0.035, "clearance_m": 0.01, "reachable": ["@target", "@container"],
              "place": {"small_box": {"region": {"xy_min": [0.18, -0.12], "xy_max": [0.28, 0.12]}},
                        "cubes": {"region": {"xy_min": [0.12, -0.18], "xy_max": [0.30, 0.18]}, "yaw_deg": [0.0, 90.0]}}},
    "stages": {"order": ["reached", "grasped", "lifted", "over_box", "placed", "success"],
               "hold_until": {"reached": "grasped", "grasped": "placed", "lifted": "placed", "over_box": "placed"}},
    "checks": {
        "reached": {"check": "ee_near", "object": "@target", "tol_m": 0.02},
        "holding": {"check": "held", "object": "@target"},
        "grasped": {"check": "all", "of": ["holding"]},
        "raised": {"check": "object_lifted", "object": "@target", "min_lift_m": 0.02},
        "lifted": {"check": "all", "of": ["holding", "raised"]},
        "above_box": {"check": "object_in_region", "object": "@target", "region": "@container.inner", "check_z": False},
        "over_box": {"check": "all", "of": ["holding", "above_box"]},
        "in_box": {"check": "object_in_region", "object": "@target", "region": "@container.inner"},
        "released": {"check": "held", "object": "@target", "negate": True},
        "placed": {"check": "all", "of": ["in_box", "released"]},
        "ee_clear": {"check": "ee_near", "object": "@target", "tol_m": 0.04, "negate": True},
        "success": {"check": "all", "of": ["placed", "ee_clear"]},
        "distractor_in_box": {"check": "objects_in_region", "set": "cubes", "exclude": ["@target"],
                              "region": "@container.inner"},
    },
    "failures": {"rules": [
        {"name": "wrong_object_placed", "if": ["distractor_in_box"]},
        {"name": "placed_not_confirmed", "if": ["placed"]},
        {"name": "never_reached", "unless": ["reached"]},
        {"name": "missed_grasp", "unless": ["grasped"]},
        {"name": "timeout_holding", "if": ["holding"]},
        {"name": "dropped_before_box", "unless": ["over_box"]},
        {"name": "dropped_outside"},
    ]},
    "expert": {"max_steps": 600, "steps": [
        {"skill": "move_joint", "q": "home"},
        {"skill": "set_gripper", "gap_m": "open"},
        {"skill": "approach", "target": "@target", "offset_m": [0, 0, 0.06], "stage": "reached"},
        {"skill": "grasp", "target": "@target", "stage": "grasped"},
        {"skill": "lift", "height_m": 0.08, "stage": "lifted"},
        {"skill": "transport", "target": "@container", "height_m": 0.06, "stage": "over_box"},
        {"skill": "place", "target": "@container"},
        {"skill": "release", "stage": "placed"},
        {"skill": "retreat", "dz_m": 0.08, "stage": "success"},
    ]},
    "perturbations": {
        "failure": [
            {"type": "grasp_offset", "weight": 2, "xy_m": [0.008, 0.02]},
            {"type": "early_release", "stages": ["lifted"], "delay_steps": [5, 40]},
            {"type": "place_offset", "xy_m": [0.025, 0.05]},
            {"type": "wrong_target"},
            {"type": "object_drop", "stages": ["lifted", "over_box"], "xy_jitter_m": [0.0, 0.02]},
        ],
        "recovery": [
            {"type": "grasp_offset", "xy_m": [0.008, 0.02]},
            {"type": "object_shift", "stages": ["reached"], "xy_m": [0.02, 0.05], "yaw_rad": [-0.5, 0.5]},
            {"type": "object_drop", "stages": ["lifted"], "xy_jitter_m": [0.0, 0.02]},
            {"type": "container_shift", "stages": ["lifted"], "xy_m": [0.02, 0.05]},
        ],
    },
}


def contract_reach_task_table() -> dict:
    """``[task]`` table transcribed from the contracts §4.6 ``prim_reach_point`` example (without ``name``)."""
    return copy.deepcopy(_REACH)


def contract_color_task_table() -> dict:
    """``[task]`` table transcribed from the contracts §4.6 ``pick_place_color_cube`` example (without ``name``).

    Deviation: ``palette.blue`` is the literal ``[0.05, 0.1, 0.8]`` instead of ``"from_setup:blue_cube"``,
    so the table needs no scene. Write it with ``files={"usd/Small-Box-Physics.usda": STUB_USDA}``.
    """
    return copy.deepcopy(_COLOR)


def write_contract_tasks(tasks_dir: str | Path, *, reach: str = "fx_contract_reach", color: str = "fx_contract_color",
                         init_py: str | None = None) -> tuple[Path, Path]:
    """Write both §4.6 example tasks (reach, colour-cube with a stub box USDA) and return their folders."""
    return (write_task(tasks_dir, reach, contract_reach_task_table(), init_py=init_py),
            write_task(tasks_dir, color, contract_color_task_table(), init_py=init_py,
                       files={"usd/Small-Box-Physics.usda": STUB_USDA}))
