"""The task.py port of pick_place_cube reproduces the legacy task exactly.

Oracles are the verbatim legacy files in tests/fixtures/legacy_tasks/pick_place_cube/:

- signals: the legacy mdp.py (with Isaac stubbed by a torch XYZW math module) against the new checks,
  driven through the env entry points in Isaac Lab step order on synthetic float32 multi-env histories;
- expert: legacy PickPlaceExpert against the composed moves, across seeds and option overrides;
- core: observe, failure_mode, progress_label, sample_start, start_is_valid (stored eval sets) and
  episode_metadata of the legacy PickPlaceTask against the generic Task.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.tasks.base import Pose, StepState  # noqa: E402
from core.tasks.loader import load_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "tests" / "fixtures" / "legacy_tasks" / "pick_place_cube"
ENVS = ROOT / "src" / "sim" / "workspace" / "envs"
TASK = "pick_place_cube"


# ---------------------------------------------------------------------------
# Isolated imports: Isaac stubs, the legacy fixture package and the env modules without gym registration
# ---------------------------------------------------------------------------


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


@contextmanager
def isolated_modules():
    """Install stubs and path-loaded packages; restore every touched sys.modules entry afterwards."""
    stub_names = ("pxr", "isaaclab", "isaaclab.utils", "isaaclab.utils.math", "isaaclab.assets", "isaaclab.envs",
                  "isaaclab.managers", "isaaclab.sensors", "isaaclab.sim", "sim.workspace.envs",
                  "sim.workspace.envs.mdp", "sim.workspace.scene.geometry", "legacy_pick_place", "core.tasks.expert")
    prefixes = ("sim.workspace.envs.", "legacy_pick_place.", "isaaclab.", "pxr.")
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
        "legacy_pick_place": _package("legacy_pick_place", LEGACY),
        # The legacy expert predates the move of its data classes to sim.actions.action.
        "core.tasks.expert": action,
    })
    sys.modules["isaaclab"].utils = sys.modules["isaaclab.utils"]
    sys.modules["isaaclab.utils"].math = sys.modules["isaaclab.utils.math"]
    try:
        yield SimpleNamespace(
            mdp=importlib.import_module("legacy_pick_place.mdp"),
            expert=importlib.import_module("legacy_pick_place.expert"),
            task_env=importlib.import_module("sim.workspace.envs.task_env"),
        )
    finally:
        for name in [name for name in sys.modules if name in stub_names or name.startswith(prefixes)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def legacy():
    with isolated_modules() as modules:
        yield modules


@pytest.fixture
def legacy_task():
    """The legacy PickPlaceTask class, built from the legacy task.toml fixture."""
    import tomllib
    from core.tasks import base

    spec = load_task(TASK).spec
    raw = tomllib.loads((LEGACY / "task.toml").read_text())["task"]
    namespace = {}
    source = (LEGACY / "__init__.py").read_text().replace('TASK = PickPlaceTask(load_task_spec("pick_place_cube"))', "")
    exec(compile(source, str(LEGACY / "__init__.py"), "exec"), namespace)
    legacy_spec = replace(spec, raw=raw, task_py=None,
                          variants={key: f"{base.TASK_PACKAGE}.{TASK}.{value}" for key, value in raw["variants"].items()})
    return namespace["PickPlaceTask"](legacy_spec)


# ---------------------------------------------------------------------------
# Signals oracle
# ---------------------------------------------------------------------------


def _legacy_params(task):
    """legacy env_cfg._place_params(TASK) plus confirm_steps, from the legacy task.toml fixture."""
    import tomllib

    raw = tomllib.loads((LEGACY / "task.toml").read_text())["task"]
    box, success = task.spec.objects["white_box"], raw["success"]
    return {
        "contact_sensor_cfg": SimpleNamespace(name="contact_grasp"),
        "object_name": "blue_cube",
        "container_name": "white_box",
        "container_size": tuple(box.params["size_outer"]),
        "container_floor_z": box.params["floor_z"],
        "container_inner_half_size": tuple(box.params["inner_half_size"]),
        "min_lift": success["min_lift"],
        "warmup_steps": success["warmup_steps"],
        "force_threshold": success["force_threshold"],
        "confirm_steps": success["confirm_steps"],
    }


def _yaw(angle):
    return [0., 0., float(np.sin(angle / 2)), float(np.cos(angle / 2))]


class History:
    """Synthetic per-env cube histories that cross every stage boundary (float32, like the simulator)."""

    TABLE = .76

    def __init__(self, env, task, rng):
        self.env, self.rng, self.n = env, rng, env.num_envs
        self.size = task.spec.objects["blue_cube"].size
        box = task.spec.objects["white_box"].params
        self.box, self.floor = box, box["floor_z"]
        self.mode, self.left = ["table"] * self.n, [0] * self.n

    def _cube(self, index, mode):
        rng, half = self.rng, self.size[2] / 2
        box = self.env.scene["white_box"].data
        box_position = box.root_pos_w[index].double().numpy()
        box_yaw = 2 * np.arctan2(float(box.root_quat_w[index, 2]), float(box.root_quat_w[index, 3]))
        rotation = np.array([[np.cos(box_yaw), -np.sin(box_yaw), 0.], [np.sin(box_yaw), np.cos(box_yaw), 0.], [0., 0., 1.]])
        yaw = rng.choice([0., np.pi / 2, rng.uniform(-np.pi, np.pi)])
        if mode in ("inside", "settled", "inside_edge", "rim"):
            inner = np.asarray(self.box["inner_half_size"])
            local = {"inside": rng.uniform(-.5, .5, 2) * (inner - .015), "settled": np.zeros(2),
                     "inside_edge": np.sign(rng.uniform(-1, 1, 2)) * (inner - half + rng.choice([-1e-4, 0., 1e-4])),
                     "rim": np.sign(rng.uniform(-1, 1, 2)) * inner}[mode]
            height = self.floor + half + rng.choice([0., 0., -1e-4, 2e-4])
            if mode == "inside_edge":
                yaw = box_yaw
            return box_position + rotation @ np.r_[local, height], yaw
        position = np.r_[rng.uniform(.05, .25), rng.uniform(-.2, .05), self.TABLE + half]
        lift = {"table": rng.choice([0., 0., -1e-4, 5e-4]), "lifted": rng.choice([.0099, .01, .0101, .03, .08]),
                "over": rng.uniform(.02, .1)}[mode]
        if mode == "over":
            position = box_position + rotation @ np.r_[rng.uniform(-.03, .03, 2), self.box["size_outer"][2] + half]
        return position + np.r_[0., 0., lift], yaw

    def place(self, index, mode=None):
        mode = mode or self.rng.choice(["table", "table", "lifted", "lifted", "over", "inside", "inside", "inside_edge",
                                        "rim", "settled"])
        position, yaw = self._cube(index, mode)
        data = self.env.scene["blue_cube"].data
        data.root_pos_w[index] = torch.tensor(position, dtype=torch.float32)
        data.root_quat_w[index] = torch.tensor(_yaw(yaw), dtype=torch.float32)
        return mode

    def contact(self, index, mode):
        choices = {"table": [0., 0., .5, 2., 2.5, 6.], "lifted": [0., 1., 2., 2.0001, 5.],
                   "over": [0., .3, 4.], "inside": [0., 0., 0., .2, 3.], "settled": [0.], "inside_edge": [0., 0., 3.],
                   "rim": [0., 0., 3.]}[mode]
        force = self.rng.choice(choices)
        # Split the norm over the jaw bodies; the sum over bodies is what both sides read.
        matrix = self.env.scene["contact_grasp"].data.force_matrix_w
        matrix[index] = 0.
        matrix[index, 0, 0, 2] = force * .25
        matrix[index, 1, 0, 0] = force * .75

    def physics(self):
        """One simulator step: hold or switch each env's phase, with small jitter."""
        data = self.env.scene["blue_cube"].data
        for index in range(self.n):
            if self.left[index] <= 0:
                self.mode[index] = self.place(index)
                self.left[index] = self.rng.choice([1, 2, 5, 20, 30, 45] if self.mode[index] != "settled" else [24, 26, 40])
                self.contact(index, self.mode[index])
            elif self.rng.random() < .1:
                data.root_pos_w[index, :2] += torch.tensor(self.rng.normal(0, 2e-4, 2), dtype=torch.float32)
            self.left[index] -= 1
            if self.rng.random() < .3 or self.left[index] == 0:
                self.contact(index, self.mode[index])
        robot = self.env.scene["robot"].data
        robot.body_link_pos_w[:, 1] = data.root_pos_w + torch.tensor(self.rng.normal(0, .03, (self.n, 3)), dtype=torch.float32)
        robot.body_link_quat_w[:, 1] = torch.nn.functional.normalize(
            torch.tensor(self.rng.normal(0, 1, (self.n, 4)), dtype=torch.float32), dim=-1)

    def reset(self, ids):
        """reset_from_starts: new start poses on the table (contact data stays from the last physics step)."""
        for index in ids.tolist():
            self.mode[index] = self.place(index, "table")
            self.left[index] = self.rng.choice([1, 10, 40])


def _env(task, rng, n):
    arm = SimpleNamespace(gripper=SimpleNamespace(kind="parallel", tip_mesh=None),
                          ik=SimpleNamespace(position_offset_m=(0., 0., -.0981)), links=SimpleNamespace(ee="gripper"))
    box_yaw = [rng.choice([0., np.pi / 2, rng.uniform(-np.pi, np.pi)]) for _ in range(n)]
    box_position = torch.tensor([[-.25 + rng.uniform(-.02, .02), -.04, History.TABLE] for _ in range(n)], dtype=torch.float32)
    scene = {
        "white_box": SimpleNamespace(cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/WhiteBox"),
                                     data=SimpleNamespace(root_pos_w=box_position,
                                                          root_quat_w=torch.tensor([_yaw(a) for a in box_yaw]))),
        "blue_cube": SimpleNamespace(cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/CUBE",
                                                         spawn=SimpleNamespace(size=task.spec.objects["blue_cube"].size)),
                                     data=SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.zeros(n, 4))),
        "robot": SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"),
                                 find_bodies=lambda name: ([1], [name]),
                                 data=SimpleNamespace(body_link_pos_w=torch.zeros(n, 3, 3),
                                                      body_link_quat_w=torch.zeros(n, 3, 4))),
        "contact_grasp": SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n, 2, 1, 3)),
                                         reset=lambda env_ids=None: None),
    }
    return SimpleNamespace(cfg=SimpleNamespace(arm_spec=arm), scene=scene, num_envs=n, device="cpu",
                           common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long))


def _same(left, right, label):
    assert left.dtype == right.dtype and left.shape == right.shape, (label, left, right)
    assert torch.equal(left, right), (label, left, right)


@pytest.mark.parametrize("seed", range(6))
def test_signals_match_the_legacy_mdp_in_isaac_step_order(legacy, seed):
    rng = np.random.default_rng(seed)
    task = load_task(TASK)
    params = _legacy_params(task)
    observation = {key: value for key, value in params.items() if key != "confirm_steps"}
    grasp = {key: value for key, value in observation.items() if not key.startswith("container_")}
    env = _env(task, rng, n=8)
    env.cfg.terminations = SimpleNamespace(success=SimpleNamespace(params=params))
    history = History(env, task, rng)
    for index in range(env.num_envs):
        history.place(index, "table")
    mdp, task_env = legacy.mdp, legacy.task_env
    all_ids = torch.arange(env.num_envs)

    def reset(ids):
        mdp.reset_pick_place_state(env, ids)
        task_env.reset_task_tracking(env, ids, TASK)

    # Isaac sizes the observation terms once at construction, before the startup reset.
    _same(task_env.stage_observation(env, TASK, "grasped"), mdp.object_grasped(env, **grasp), "sizing")
    _same(task_env.stage_observation(env, TASK, "placed"), mdp.object_placed_in_container(env, **observation), "sizing")
    reset(None if seed % 2 else all_ids)  # the startup reset of every environment
    seen = {key: 0 for key in ("grasped", "lifted", "placed", "success", "holding", "reached")}
    for step in range(1, 700):
        history.physics()
        env.common_step_counter += 1
        env.episode_length_buf += 1
        # Termination manager: time_out then success.
        time_out = env.episode_length_buf >= 450
        success = mdp.object_placed_in_container_termination(env, **params)
        _same(task_env.task_success(env, TASK), success, ("termination", step))
        # The stage tracker reads the signals; repeated calls within a step are idempotent.
        for _ in range(2):
            old, new = mdp.stage_signals(env), task_env.task_signals(env, TASK)
            assert set(new) == set(old) == {"grasped", "placed", "success", "holding", "lifted", "reach_distance"}
            for key in old:
                _same(new[key], old[key], (key, step))
        seen["reached"] += int((old["reach_distance"] <= .03).sum())
        for key in ("grasped", "lifted", "placed", "success", "holding"):
            seen[key] += int(old[key].sum())
        # Reset terminated environments, plus a few others, after writing their new start poses.
        extra = torch.tensor(rng.random(env.num_envs) < .01)
        ids = all_ids[success | time_out | extra]
        if len(ids):
            history.reset(ids)
            env.episode_length_buf[ids] = 0
            reset(ids if rng.random() < .8 else ids.tolist())
        elif rng.random() < .005:
            # An event manager may also pass a slice for every environment.
            history.reset(all_ids)
            env.episode_length_buf[:] = 0
            reset(slice(None))
        # Observation manager: the subtask terms.
        _same(task_env.stage_observation(env, TASK, "grasped"), mdp.object_grasped(env, **grasp), ("cube_grasped", step))
        _same(task_env.stage_observation(env, TASK, "placed"), mdp.object_placed_in_container(env, **observation),
              ("cube_placed", step))
        # A tracker reading after the step (reset envs re-evaluated on their new poses) still agrees.
        if rng.random() < .3:
            old, new = mdp.stage_signals(env), task_env.task_signals(env, TASK)
            for key in old:
                _same(new[key], old[key], (key, step, "after reset"))
    # The synthetic histories exercised every stage, including confirmed successes.
    assert all(count > 0 for count in seen.values()), seen


class _Untouchable:
    def __init__(self, label):
        self._label = label

    def __getattr__(self, name):
        raise AssertionError(f"{self._label}.{name} must not be read by this term")


def test_termination_observations_and_reset_read_no_robot_or_velocity(legacy):
    """As in the legacy terms, only the stage tracer reads the end effector; nothing reads velocities."""
    rng = np.random.default_rng(5)
    task = load_task(TASK)
    env = _env(task, rng, n=2)
    env.scene["robot"] = _Untouchable("robot")
    for name in ("blue_cube", "white_box"):
        data = env.scene[name].data
        env.scene[name].data = type("Data", (), {
            "root_pos_w": data.root_pos_w, "root_quat_w": data.root_quat_w,
            "root_vel_w": property(lambda self: (_ for _ in ()).throw(AssertionError("root_vel_w read")))})()
    env.scene["blue_cube"].data.root_pos_w[:] = torch.tensor([.1, -.1, .77])
    env.scene["blue_cube"].data.root_quat_w[:] = torch.tensor(_yaw(0.))
    task_env = legacy.task_env
    task_env.stage_observation(env, TASK, "grasped")
    task_env.reset_task_tracking(env, None, TASK)
    for step in range(1, 4):
        env.common_step_counter = step
        task_env.task_success(env, TASK)
        task_env.stage_observation(env, TASK, "grasped")
        task_env.stage_observation(env, TASK, "placed")
    with pytest.raises(AssertionError, match="robot"):
        task_env.task_signals(env, TASK)


def test_stage_hyperparameters_are_the_legacy_success_parameters():
    """Teleop has no success termination; the legacy stage_signals fallback read these same values."""
    task = load_task(TASK)
    params = _legacy_params(task)
    module = task.module
    assert params["min_lift"] == module.lift.min_lift
    assert params["force_threshold"] == module.grasp.force
    assert params["warmup_steps"] == module.grasp.warmup_steps == module.placed_in_box.warmup_steps
    assert params["confirm_steps"] == module.stages["success"].steps
    assert module.stages["reached"].dist == pytest.approx(.03)
    assert module.placed_in_box.grasp is module.grasp and module.grasp.lift is module.lift


# ---------------------------------------------------------------------------
# Expert oracle
# ---------------------------------------------------------------------------


def _expert_case(seed):
    from sim.actions.action import ExpertContext

    rng = random.Random(seed)
    yaw = rng.uniform(-np.pi, np.pi)
    table = np.array([0., 0., 1.])
    if seed % 5 == 4:  # a tilted table
        angle = rng.uniform(-.2, .2)
        table = np.array([np.sin(angle), 0., np.cos(angle)])
    cube = Pose.from_lists((rng.uniform(.05, .3), rng.uniform(-.3, .1), .76 + rng.uniform(0, .01)), _yaw(yaw))
    box = Pose.from_lists((rng.uniform(-.35, -.15), rng.uniform(-.1, .1), .75), _yaw(rng.choice([0., rng.uniform(-3, 3)])))
    ee = Pose.from_lists((0., 0., 1.), _yaw(rng.uniform(-1, 1)))
    state = StepState({"blue_cube": cube, "white_box": box}, ee, 0.)
    options = {}
    for key, values in {
        "approach_height_m": [.04, .08], "lift_height_m": [.07, .12], "release_height_m": [0., .03, "lift"],
        "move_seconds": [1., 3.], "transport_seconds": [.5, 2.5], "final_transport_seconds": [1.2],
        "grasp_seconds": [.5], "release_seconds": [.3], "settle_seconds": [2.], "place_margin_m": [.001, .005],
        "place_xy": [[.01, -.01], [-.02, .005]], "transport_arc_margin_m": [.005, .02],
        "transport_interpolate_radius": [True, False], "position_tolerance_m": [.002],
        "waypoint_timeout_seconds": [12.], "object_feedback": [True, False],
        "gripper_open_fraction": [.6], "gripper_closed_fraction": [.05], "orientation_xyzw": [(0., 1., 0., 0.)],
        "gripper_open_gap_m": [.035], "gripper_closed_gap_m": [.001],
    }.items():
        if rng.random() < .3:
            options[key] = rng.choice(values)
    if options.get("release_height_m") == "lift":
        options["release_height_m"] = options.get("lift_height_m", .09)
    arm = None
    if seed % 3:
        from utils.arms.arms import load_arm_spec

        arm = load_arm_spec(rng.choice(["so101", "sp7"]))[0]
    base = Pose.from_lists((rng.uniform(-.05, .05), rng.uniform(-.05, .05), .75))
    return ExpertContext(state, ee, tuple(table), options, base, arm)


def _plan(expert, context):
    try:
        return expert.plan(context)
    except Exception as error:  # the same refusal is part of the behaviour
        return type(error), str(error)


@pytest.mark.parametrize("seed", range(50))
def test_expert_matches_the_legacy_waypoints(legacy, legacy_task, seed):
    task = load_task(TASK)
    context = _expert_case(seed)
    planned = _plan(task.expert(), context)
    assert planned == _plan(legacy.expert.create(legacy_task), context)


def test_expert_cases_mostly_plan(legacy):
    task = load_task(TASK)
    planned = [_plan(task.expert(), _expert_case(seed)) for seed in range(50)]
    assert sum(not isinstance(plan[0], type) for plan in planned) >= 40


@pytest.mark.parametrize("options", [
    {"transport_roll_degrees": 90.}, {"transport_orientation_xyzw": (0., 0., 0., 1.)},
    {"place_margin_m": .05}, {"place_xy": [.2, 0.]}, {"place_xy": [0., 0., 0.]},
    {"release_height_m": -.01}, {"release_height_m": .2}, {"transport_arc_margin_m": 0.},
    {"move_seconds": -1.}, {"position_tolerance_m": 0.},
])
def test_expert_refusals_match_the_legacy_expert(legacy, legacy_task, options):
    context = replace(_expert_case(0), options=options)
    planned = _plan(load_task(TASK).expert(), context)
    assert isinstance(planned[0], type) and planned == _plan(legacy.expert.create(legacy_task), context)
    without_base = replace(_expert_case(3), arm_base=None)
    assert _plan(load_task(TASK).expert(), without_base) == _plan(legacy.expert.create(legacy_task), without_base)


# ---------------------------------------------------------------------------
# Core oracle
# ---------------------------------------------------------------------------


def _states(rng, count):
    names = ["reached", "grasped", "lifted", "over_box", "placed", "success"]
    for _ in range(count):
        cube = Pose.from_lists((rng.uniform(-.35, .3), rng.uniform(-.2, .1), .76), _yaw(rng.uniform(-3, 3)))
        box = Pose.from_lists((rng.uniform(-.3, -.2), rng.uniform(-.06, -.02), .75), _yaw(rng.choice([0., rng.uniform(-3, 3)])))
        ee = Pose.from_lists(np.asarray(cube.position) + rng.normal(0, .03, 3))
        signals = {}
        for key in ("grasped", "lifted", "placed", "success", "holding"):
            if rng.random() < .9:
                signals[key] = bool(rng.random() < .5)
        if rng.random() < .8:
            signals["reach_distance"] = float(rng.choice([.03, np.nextafter(.03, 1), rng.uniform(0, .06)]))
        state = StepState({"blue_cube": cube, "white_box": box}, ee, 0., signals)
        flags = {name: bool(rng.random() < .5) for name in names if rng.random() < .9}
        yield state, flags


def test_core_rules_match_the_legacy_task(legacy_task):
    task = load_task(TASK)
    rng = np.random.default_rng(3)
    for state, flags in _states(rng, 3000):
        assert task.observe(state) == legacy_task.observe(state)
        for current in (state, None):
            assert task.failure_mode(flags, current) == legacy_task.failure_mode(flags, current)
            assert task.progress_label(flags, current) == legacy_task.progress_label(flags, current)
    assert task.stages == legacy_task.stages


def test_sampled_starts_match_the_legacy_task(legacy_task):
    task = load_task(TASK)
    for seed in range(100):
        assert task.sample_start(np.random.default_rng(seed)) == legacy_task.sample_start(np.random.default_rng(seed))


def test_stored_eval_set_starts_get_the_legacy_verdicts(legacy_task):
    from core.eval import eval_set

    task = load_task(TASK)
    paths = sorted(p for p in ROOT.rglob("eval_sets/*.json") if "tests" not in p.parts)
    paths.append(ROOT / "tests" / "fixtures" / "legacy_eval_set_v1.json")
    checked = rejected = 0
    rng = np.random.default_rng(0)
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("task", TASK) != TASK or "starts" not in data:
            continue
        for scene in (data["scene"], eval_set.normalize_scene(data["scene"], task)):
            for start in data["starts"]:
                objects = eval_set.start_objects(start, task=task)
                poses = {}
                for name, row in objects.items():
                    w, x, y, z = row["orientation_wxyz"]
                    poses[name] = Pose.from_lists(row["position"], (x, y, z, w))
                verdict = task.start_is_valid(poses, scene)
                assert verdict == legacy_task.start_is_valid(poses, scene), (path.name, start["id"])
                checked, rejected = checked + 1, rejected + (not verdict)
                # Nearby starts, some overlapping the box, get the same verdict too.
                cube = poses["blue_cube"]
                moved = dict(poses, blue_cube=Pose.from_lists(
                    np.asarray(cube.position) + np.r_[rng.uniform(-.3, .1), rng.uniform(-.1, .1), 0.], cube.quat_xyzw))
                assert task.start_is_valid(moved, scene) == legacy_task.start_is_valid(moved, scene)
    assert checked > 500 and rejected > 0


def test_episode_metadata_matches_the_legacy_task(legacy_task):
    task = load_task(TASK)
    assert task.episode_metadata(SimpleNamespace()) == legacy_task.episode_metadata(SimpleNamespace())
    for names in (["red", "blue"], ["blue", "red"]):
        new, old = SimpleNamespace(_object_colors={"blue_cube": names}), SimpleNamespace(_pick_place_cube_color=names)
        for index in range(2):
            assert task.episode_metadata(new, index) == legacy_task.episode_metadata(old, index)


def test_object_colour_names_match_the_legacy_palette():
    import tomllib
    from utils.scene import load_scene_spec
    with isolated_modules() as modules:
        task = load_task(TASK)
        scene = load_scene_spec()
        raw = tomllib.loads((LEGACY / "task.toml").read_text())["task"]
        base = scene.appearance.get("object_materials", {}).get("blue_cube", {}).get("color", (0, 0, 1))
        legacy = {name: tuple(base) if value == "from_setup" else tuple(value)
                  for name, value in raw["dr"]["cube_colors"].items()}
        new = modules.task_env.object_colors(task.spec, scene)
        assert new == {"blue_cube": {key: tuple(float(v) for v in value) for key, value in legacy.items()}}
        assert list(new["blue_cube"]) == list(legacy)
