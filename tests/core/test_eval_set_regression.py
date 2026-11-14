"""Frozen eval sets keep their digests and stored starts through the v3 restructure.

Needs only numpy and the stdlib: no torch, pyarrow or Isaac. The eval-set module still imports torch
indirectly (utils.arms.mapping), so when torch is absent it is imported once with a stand-in ``torch``.
"""
import functools
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

# --- _locate: the only place later restructure phases should need to edit --------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVAL_SET_MODULES = ("core.eval.eval_set",)# first importable wins
TASK_LOADER_MODULES = ("core.tasks.loader",)                          # provides load_task(name)
POSE_MODULES = ("core.tasks.base",)                                   # provides Pose.from_lists(pos, xyzw)
TASK_DIRS = ("workspace/tasks", "tasks")                              # first existing wins
LEGACY_FIXTURE = "legacy_eval_set_v1.json"                            # schema-1 set, searched under tests/
SKIP_DIRS = {"node_modules", "__pycache__"}                           # plus dot-dirs and top-level tests/
# ----------------------------------------------------------------------------------------------------------

# Pinned when v3 was branched from v2. "rejected" lists stored starts that the task's own start check
# already refuses on v2: train-059 is a recorded training demo whose cube footprint reaches about 4 mm past
# the stored table outline. Pinning it exposes both new rejections and silent changes to the rule.
PINNED = {
    "pick_place_v1_75ep_reach150_seed1984.json": dict(
        task="pick_place_cube", schema=2, starts=150, rejected={"train-059"},
        digest="4102164c55500b043c6cab4697a96c1c7d429decce138e1b6fd4e0eff9e99084",
        legacy="233254427402237a00ade1db7c182e79fca1dc7054b42de78b637f040342fa4b"),
    "pick_place_v1_75ep_seed1984.json": dict(
        task="pick_place_cube", schema=2, starts=200, rejected={"train-059"},
        digest="81e9e9472846750875c81880918a15156e8cdd46a01d746e6f575bcfff525e23",
        legacy="779f5e18b147939309efb5638560c963f6d6aa5ca1d6271e1c186714ed576e14"),
    "pick_place_v1_seed1984.json": dict(
        task="pick_place_cube", schema=2, starts=150, rejected=set(),
        digest="dddd4db4d7f9092db3f218d9f7ef42e60844e28bf2e7bc062bea80e054e5444f",
        legacy="91dde5ea0243ca56c62040bfe66df6e171f00b80f2663a2f3a457880bad7ff08"),
}
PINNED_LEGACY = dict(task="pick_place_cube", schema=1, starts=4, rejected=set(), legacy=None,
                     digest="a13e9cc9e0e245072e4bdbf9ed13d4ee99468d49b9a0b8775971ca8d89e064c2")
EXPECTED_TASKS = {"pick_place_cube", "stack_bowls", "stack_two_cubes"}
ISAAC_PREFIXES = ("isaaclab", "isaacsim", "omni", "pxr", "carb", "warp", "newton", "mujoco_warp")


def _import_first(names):
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name is None or not (name == error.name or name.startswith(error.name + ".")):
                raise
    raise ModuleNotFoundError(f"None of {names} is importable; update the _locate section")


def _stand_in_torch():
    module = types.ModuleType("torch")
    module.__getattr__ = lambda name: type(name, (), {})   # class-level annotations such as torch.Tensor
    return module


@functools.cache
def eval_set_module():
    try:
        return _import_first(EVAL_SET_MODULES)
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
    missing = object()
    saved = sys.modules.get("torch", missing)
    sys.modules["torch"] = _stand_in_torch()
    try:
        return _import_first(EVAL_SET_MODULES)
    finally:
        if saved is missing:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def load_task(name):
    return _import_first(TASK_LOADER_MODULES).load_task(name)


def _visible_dirs(names, top):
    skip = SKIP_DIRS | {"tests"} if top else SKIP_DIRS
    return [name for name in names if not name.startswith(".") and name not in skip]


def find_eval_sets():
    """Every JSON under an eval_sets/ folder that is an eval set (has eval_set_id and starts), outside tests/."""
    found = {}
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(_visible_dirs(dirs, Path(current) == ROOT))
        if Path(current).name != "eval_sets":
            continue
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix != ".json":
                continue
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "eval_set_id" in data and "starts" in data:
                assert name not in found, f"Duplicate eval set filename {name}: {found[name]} and {path}"
                found[name] = path
    return found


def find_legacy_fixture():
    paths = sorted(p for p in (ROOT / "tests").rglob(LEGACY_FIXTURE) if "__pycache__" not in p.parts)
    assert len(paths) == 1, f"Expected one {LEGACY_FIXTURE} under tests/, found {paths}"
    return paths[0]


def tasks_dir():
    for relative in TASK_DIRS:
        if (ROOT / relative).is_dir():
            return ROOT / relative
    raise FileNotFoundError(f"None of {TASK_DIRS} exists; update the _locate section")


def _cases():
    sets = find_eval_sets()
    cases = [pytest.param(sets[name], PINNED[name], id=name) for name in sorted(sets) if name in PINNED]
    return cases + [pytest.param(find_legacy_fixture(), PINNED_LEGACY, id=LEGACY_FIXTURE)]


def test_discovered_eval_sets_are_exactly_the_pinned_ones():
    assert set(find_eval_sets()) == set(PINNED)


@pytest.mark.parametrize("path,pinned", _cases())
def test_digest_matches_file_and_pinned_literal(path, pinned):
    data = json.loads(path.read_text())
    digest = eval_set_module().eval_set_digest(data)
    assert data["schema_version"] == pinned["schema"]
    assert digest == data["eval_set_id"], "eval-set digest function disagrees with the stored eval_set_id"
    assert digest == pinned["digest"], "stored eval_set_id or the digest function changed"
    # Schema-2 conversions keep the schema-1 identity; it is part of their digest.
    assert data.get("legacy_eval_set_id") == pinned["legacy"]
    assert len(data["starts"]) == pinned["starts"]


def _rejected_starts(module, data, task, scene):
    pose_type = _import_first(POSE_MODULES).Pose
    rejected = set()
    for start in data["starts"]:
        # As reset_from_starts: EvalSetStarts.sample -> WXYZ objects -> XYZW Pose -> task.start_is_valid.
        objects = module.start_objects(start, task=task)
        poses = {}
        for name, row in objects.items():
            w, x, y, z = row["orientation_wxyz"]
            poses[name] = pose_type.from_lists(row["position"], (x, y, z, w))
        if not task.start_is_valid(poses, scene):
            rejected.add(start["id"])
    return rejected


@pytest.mark.parametrize("path,pinned", _cases())
def test_stored_starts_pass_the_task_start_check(path, pinned):
    module = eval_set_module()
    data = json.loads(path.read_text())
    task = load_task(data.get("task", pinned["task"]))
    assert task.name == pinned["task"]
    # Live resets pass a captured scene; check both the stored snapshot and its named-object normalization.
    assert _rejected_starts(module, data, task, data["scene"]) == pinned["rejected"]
    assert _rejected_starts(module, data, task, module.normalize_scene(data["scene"], task)) == pinned["rejected"]


LOAD_TASKS_WITHOUT_TORCH = """
import importlib, json, sys
loader = None
for name in json.loads(sys.argv[1]):
    try:
        loader = importlib.import_module(name)
        break
    except ModuleNotFoundError as error:
        if error.name is None or not (name == error.name or name.startswith(error.name + ".")):
            raise
for task in json.loads(sys.argv[2]):
    assert loader.load_task(task).name == task, task
prefixes = tuple(json.loads(sys.argv[3]))
heavy = sorted(m for m in sys.modules if m.split(".")[0] == "torch" or m.split(".")[0].startswith(prefixes))
assert not heavy, heavy
print("loaded", len(json.loads(sys.argv[2])))
"""


def test_every_task_loads_without_torch_or_isaac():
    # Folders starting with "_" are scaffolding templates (the CLI listing skips them too).
    names = sorted(p.parent.name for p in tasks_dir().glob("*/task.toml") if not p.parent.name.startswith("_"))
    assert EXPECTED_TASKS <= set(names), f"Task folders missing from {tasks_dir()}: {EXPECTED_TASKS - set(names)}"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])))
    env.pop("S2R_TASKS_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", LOAD_TASKS_WITHOUT_TORCH, json.dumps(TASK_LOADER_MODULES), json.dumps(names),
         json.dumps(ISAAC_PREFIXES)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
