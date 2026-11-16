"""Shared harness for the task.py port oracles (tests/workspace/tasks/test_<task>.py).

The legacy task files are kept verbatim in tests/fixtures/legacy_tasks/<task>/. This module imports
them (and the env modules, without gym registration) against a torch stand-in for Isaac Lab's XYZW
quaternion math, and builds the legacy Task class from its legacy task.toml.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib
from pathlib import Path
import re
import sys
import tomllib
import types
from types import SimpleNamespace

import numpy as np
import torch

from utils.workspace import load_file_module
from core.tasks.task import load_task

ROOT = Path(__file__).resolve().parents[3]
LEGACY_TASKS = ROOT / "tests" / "fixtures" / "legacy_tasks"
ENVS = ROOT / "src" / "sim" / "workspace" / "envs"


def _math_module():
    """Isaac Lab 3.0 quaternion math (XYZW) in torch."""
    math = types.ModuleType("isaaclab.utils.math")

    def quat_mul(a, b):
        x1, y1, z1, w1 = a.unbind(-1)
        x2, y2, z2, w2 = b.unbind(-1)
        return torch.stack((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2), dim=-1)

    def quat_inv(q):
        conjugate = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
        return conjugate / q.pow(2).sum(-1, keepdim=True).clamp(min=1e-9)

    def quat_apply(q, v):
        xyz, w = q[..., :3], q[..., 3:]
        t = 2 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    def matrix_from_quat(q):
        x, y, z, w = q.unbind(-1)
        s = 2.0 / (q * q).sum(-1)
        return torch.stack((1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
                            s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
                            s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)),
                           dim=-1).reshape(q.shape[:-1] + (3, 3))

    math.quat_mul, math.quat_inv, math.quat_apply, math.matrix_from_quat = quat_mul, quat_inv, quat_apply, matrix_from_quat
    return math


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def legacy_base():
    return load_file_module(LEGACY_TASKS / "base.py", "_s2r_legacy_task_base")


@contextmanager
def legacy_base_import():
    name = "core.tasks.base"
    previous = sys.modules.get(name)
    sys.modules[name] = legacy_base()
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@contextmanager
def isolated_modules(task, legacy_task=None):
    """Install Isaac stubs and the path-loaded legacy package ``legacy_<task>``; restore sys.modules after.

    ``legacy_task`` becomes the package's ``TASK`` (legacy mdp modules read it with ``from . import TASK``).
    """
    package = "legacy_" + task
    stub_names = ("pxr", "isaaclab", "isaaclab.utils", "isaaclab.utils.math", "isaaclab.assets", "isaaclab.envs",
                  "isaaclab.managers", "isaaclab.sensors", "isaaclab.sim", "sim.workspace.envs",
                  "sim.workspace.envs.mdp", "sim.workspace.scene.geometry", package, "core.tasks.expert", "core.tasks.base")
    prefixes = ("sim.workspace.envs.", package + ".", "isaaclab.", "pxr.")
    saved = {name: module for name, module in sys.modules.items()
             if name in stub_names or name.startswith(prefixes)}
    import sim.actions.action as action

    def not_used(*args, **kwargs):
        raise AssertionError("a parallel gripper must not need the USD tip mesh")

    sys.modules.update({
        "pxr": _module("pxr", Gf=None, Usd=None, UsdShade=None),
        "isaaclab": _module("isaaclab"),
        "isaaclab.utils": _module("isaaclab.utils"),
        "isaaclab.utils.math": _math_module(),
        "isaaclab.assets": _module("isaaclab.assets", RigidObject=object),
        "isaaclab.envs": _module("isaaclab.envs", ManagerBasedRLEnv=object),
        "isaaclab.managers": _module("isaaclab.managers", SceneEntityCfg=lambda name: SimpleNamespace(name=name)),
        "isaaclab.sensors": _module("isaaclab.sensors", ContactSensor=object),
        "isaaclab.sim": _module("isaaclab.sim", get_current_stage=lambda: None),
        "sim.workspace.envs": _package("sim.workspace.envs", ENVS),
        "sim.workspace.envs.mdp": _package("sim.workspace.envs.mdp", ENVS / "mdp"),
        "sim.workspace.scene.geometry": _module("sim.workspace.scene.geometry", gripper_tip_local_offset=not_used),
        package: _package(package, LEGACY_TASKS / task),
        # The legacy expert predates the move of its data classes to sim.actions.action.
        "core.tasks.expert": action,
        "core.tasks.base": legacy_base(),
    })
    sys.modules["isaaclab"].utils = sys.modules["isaaclab.utils"]
    sys.modules["isaaclab.utils"].math = sys.modules["isaaclab.utils.math"]
    # The mdp package's own __init__ imports Isaac; expose the two helpers it re-exports.
    common = importlib.import_module("sim.workspace.envs.mdp._common")
    sys.modules["sim.workspace.envs.mdp"].as_torch = common.as_torch
    sys.modules["sim.workspace.envs.mdp"].env_indices = common.env_indices
    sys.modules[package].TASK = legacy_task
    try:
        yield SimpleNamespace(
            mdp=importlib.import_module(package + ".mdp"),
            expert=importlib.import_module(package + ".expert"),
            task_env=importlib.import_module("sim.workspace.envs.task_env"),
        )
    finally:
        for name in [name for name in sys.modules if name in stub_names or name.startswith(prefixes)]:
            del sys.modules[name]
        sys.modules.update(saved)


def legacy_raw(task):
    return tomllib.loads((LEGACY_TASKS / task / "task.toml").read_text())["task"]


def legacy_task(task, class_name, **changes):
    """The legacy Task class from the fixture __init__.py, built from the legacy task.toml fixture."""
    spec = load_task(task).spec
    raw = legacy_raw(task)
    path = LEGACY_TASKS / task / "__init__.py"
    source = re.sub(r"^TASK = .*$", "", path.read_text(), flags=re.M)
    namespace = {}
    with legacy_base_import():
        exec(compile(source, str(path), "exec"), namespace)
    legacy_spec = replace(spec, raw=raw, task_py=None,
                          variants={key: f"{legacy_base().TASK_PACKAGE}.{task}.{value}" for key, value in raw["variants"].items()})
    return namespace[class_name](replace(legacy_spec, **changes))


def yaw(angle):
    return [0., 0., float(np.sin(angle / 2)), float(np.cos(angle / 2))]


def same(left, right, label):
    assert left.dtype == right.dtype and left.shape == right.shape, (label, left, right)
    assert torch.equal(left, right), (label, left, right)


def plan(expert, context):
    try:
        return expert.plan(context)
    except Exception as error:  # the same refusal is part of the behaviour
        return type(error), str(error)


class Untouchable:
    def __init__(self, label):
        self._label = label

    def __getattr__(self, name):
        raise AssertionError(f"{self._label}.{name} must not be read by this term")
