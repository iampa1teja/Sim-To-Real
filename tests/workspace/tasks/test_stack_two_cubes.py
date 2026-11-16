"""Stack port against verbatim legacy signals, core rules, expert and cfg expectations."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.tasks.task import Pose, StepState
from core.tasks.task import load_task
from sim.actions.action import ExpertContext
from utils.arms.arms import load_arm_spec
from utils.geometry import rotation_matrix_xyzw
from task_oracle import isolated_modules, legacy_task as make_legacy_task, yaw, same, plan

TASK = "stack_two_cubes"


@pytest.fixture
def legacy_task():
    return make_legacy_task(TASK, "StackTwoCubesTask")


@pytest.fixture
def legacy(legacy_task):
    with isolated_modules(TASK, legacy_task) as modules:
        yield modules


def environment(n=8):
    arm = SimpleNamespace(gripper=SimpleNamespace(kind="parallel", tip_mesh=None),
                          ik=SimpleNamespace(position_offset_m=(0., 0., -.0981)), links=SimpleNamespace(ee="gripper"))
    scene = {}
    for name in ("cube_a", "cube_b"):
        scene[name] = SimpleNamespace(
            cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/" + name, spawn=SimpleNamespace(size=(.02, .02, .02))),
            data=SimpleNamespace(root_pos_w=torch.tensor([[.1, -.1, .77]] * n),
                                 root_quat_w=torch.tensor([yaw(0)] * n), root_vel_w=torch.zeros(n, 6)))
    scene["robot"] = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"), find_bodies=lambda name: ([1], [name]),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(n, 3, 3), body_link_quat_w=torch.zeros(n, 3, 4)))
    scene["contact_grasp"] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n, 2, 1, 3)),
                                             reset=lambda *args, **kwargs: None)
    return SimpleNamespace(cfg=SimpleNamespace(arm_spec=arm), scene=scene, num_envs=n, device="cpu",
                           common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long))


@pytest.mark.parametrize("seed", range(6))
def test_signals_match_legacy_in_step_order(legacy, seed):
    """Cross warmup, hysteresis, footprint/height/speed boundaries, confirmation and partial resets."""
    rng = np.random.default_rng(seed)
    env = environment()
    old, new = legacy.mdp, legacy.task_env
    ids = torch.arange(env.num_envs)
    seen = dict.fromkeys(("holding", "grasped", "lifted", "over_target", "stacked", "success"), 0)

    def reset(selected):
        old.reset_tracking(env, selected)
        new.reset_task_tracking(env, selected, TASK)

    def compare(label):
        expected, actual = old.stage_signals(env), new.task_signals(env, TASK)
        assert set(actual) == set(expected)
        for key in expected:
            same(actual[key], expected[key], (label, key))
        return expected

    compare("before startup reset")
    reset(None if seed % 2 else ids)
    for step in range(1, 701):
        a, b = env.scene["cube_a"].data, env.scene["cube_b"].data
        # A repeated deterministic episode ensures all stages; other rows cross exact thresholds.
        phase = step % 100
        for index in range(env.num_envs):
            angle = 0. if index == 0 else rng.choice([0., np.pi / 2, rng.uniform(-3, 3)])
            quaternion = yaw(angle) if index < 6 else np.array([.1, .2, np.sin(angle / 2), np.cos(angle / 2)])
            quaternion = np.asarray(quaternion) / np.linalg.norm(quaternion)
            b.root_quat_w[index] = torch.tensor(quaternion)
            a.root_quat_w[index] = torch.tensor(yaw(angle if index % 2 == 0 else rng.uniform(-3, 3)))
            offset = np.zeros(3)
            force, speed = 0., 0.
            if phase < 10:
                offset = np.array([.08, 0., 0.])
            elif phase < 40:
                offset[2] = .03 if index == 0 else rng.choice([.009999, .01, .010001, .02, .04])
                force = 3. if index == 0 else rng.choice([0., .1, 2., 2.0001, 5.])
            else:
                offset[2] = .02 if index == 0 else .02 + rng.choice([-.004001, -.004, 0., .004, .004001])
                if index:
                    offset[:2] = rng.choice([0., .003999, .004, .004001, .01], 2)
                    force = rng.choice([0., 0., .001, 2.])
                    speed = rng.choice([0., .049999, .05, .050001])
            rotation = rotation_matrix_xyzw(quaternion)
            a.root_pos_w[index] = b.root_pos_w[index] + torch.tensor(rotation @ offset, dtype=torch.float32)
            a.root_vel_w[index] = 0.
            a.root_vel_w[index, 0] = speed
            matrix = env.scene["contact_grasp"].data.force_matrix_w
            matrix[index] = 0.
            matrix[index, 0, 0, 2] = force * .25
            matrix[index, 1, 0, 0] = force * .75
        robot = env.scene["robot"].data
        robot.body_link_pos_w[:, 1] = a.root_pos_w + torch.tensor(rng.normal(0., .03, (env.num_envs, 3)), dtype=torch.float32)
        robot.body_link_quat_w[:, 1] = torch.nn.functional.normalize(
            torch.tensor(rng.normal(size=(env.num_envs, 4)), dtype=torch.float32), dim=-1)
        env.common_step_counter = step
        env.episode_length_buf += 1
        same(new.task_success(env, TASK), old.success(env), (step, "termination"))
        for repeat in range(2):
            signals = compare((step, repeat))
        for key in seen:
            seen[key] += int(signals[key].sum())
        for name in ("grasped", "stacked"):
            same(new.stage_observation(env, TASK, name), old.signal(env, name), (step, name))
        # Reset only some rows inside the same step, leaving cached rows untouched.
        selected = ids[torch.tensor(rng.random(env.num_envs) < .025) & (ids != 0)]
        if phase == 0:
            selected = ids
        if len(selected):
            a.root_pos_w[selected] = b.root_pos_w[selected] + torch.tensor([.08, 0., 0.])
            env.episode_length_buf[selected] = 0
            reset(selected if step % 2 else selected.tolist())
            compare((step, "after partial reset"))
    assert all(value > 0 for value in seen.values()), seen


def expert_case(seed):
    rng = np.random.default_rng(seed)
    def pose():
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        return Pose.from_lists(rng.uniform([.05, -.3, .75], [.3, .1, .8]), q)
    state = StepState({"cube_a": pose(), "cube_b": pose()}, pose(), 0.)
    options = {}
    for key, values in {
        "move_seconds": [1., 3.], "transport_seconds": [.2, 1.], "grasp_seconds": [.4, 1.2],
        "release_seconds": [.3, 1.], "settle_seconds": [.1, 2.], "control_fps": [10., 60.],
        "approach_height_m": [.04, .08], "lift_height_m": [.03, .12], "stack_clearance_m": [0., .002],
        "position_tolerance_m": [.0001, .002, .01], "waypoint_timeout_seconds": [1., 12.],
        "gripper_open_fraction": [.6, .9], "gripper_closed_fraction": [0., .05],
        "gripper_open_gap_m": [.035, .04], "gripper_closed_gap_m": [.001, .002],
        "object_feedback": [True, False], "orientation_xyzw": [(0., 1., 0., 0.)],
    }.items():
        if rng.random() < .4:
            options[key] = values[int(rng.integers(len(values)))]
    arm = None if seed % 3 == 0 else load_arm_spec("sp7" if seed % 3 == 1 else "so101")[0]
    normal = rng.normal(size=3)
    return ExpertContext(state, state.ee, tuple(normal), options, Pose((0., 0., .75)), arm)


@pytest.mark.parametrize("seed", range(50))
def test_expert_matches_legacy_waypoints(legacy, legacy_task, seed):
    context = expert_case(seed)
    expected = legacy.expert.create(legacy_task).plan(context)
    assert len(expected) == 9  # Never count matching exceptions as waypoint equality.
    assert load_task(TASK).expert().plan(context) == expected


@pytest.mark.parametrize("changes", [
    {"options": {"transport_roll_degrees": 90.}}, {"options": {"transport_orientation_xyzw": (0., 0., 0., 1.)}},
    {"table_normal": (0., 0., 0.)}, {"table_normal": (float("nan"), 0., 1.)},
    {"options": {"stack_clearance_m": -.001}}, {"options": {"stack_clearance_m": float("inf")}},
    {"options": {"approach_height_m": 0.}}, {"options": {"lift_height_m": -1.}},
    {"options": {"control_fps": 0.}}, {"options": {"control_fps": float("nan")}},
])
def test_expert_refusals_match(legacy, legacy_task, changes):
    ctx = replace(expert_case(0), **changes)
    expected = plan(legacy.expert.create(legacy_task), ctx)
    assert isinstance(expected[0], type)
    assert plan(load_task(TASK).expert(), ctx) == expected


def test_core_and_starts_match(legacy_task):
    task = load_task(TASK)
    rng = np.random.default_rng(10)
    assert task.stages == legacy_task.stages
    for _ in range(3000):
        state = expert_case(int(rng.integers(50))).state
        signals = {key: bool(rng.integers(2)) for key in (*task.stages, "holding") if rng.random() < .8}
        if rng.random() < .8:
            signals["reach_distance"] = float(rng.choice([.03, np.nextafter(.03, 1.), rng.uniform(0., .06)]))
        state = replace(state, signals=signals)
        flags = {name: bool(rng.integers(2)) for name in task.stages if rng.random() < .8}
        assert task.observe(state) == legacy_task.observe(state)
        for current in (state, None):
            # Phase 15 intentionally replaces legacy failure names; flag/progress oracles remain.
            assert task.progress_label(flags, current) == legacy_task.progress_label(flags, current)
    for seed in range(100):
        assert task.sample_start(np.random.default_rng(seed)) == legacy_task.sample_start(np.random.default_rng(seed))
    assert task.episode_metadata(SimpleNamespace()) == legacy_task.episode_metadata(SimpleNamespace())


@pytest.mark.parametrize("selection", [None, [1, 3], slice(None)])
def test_target_reset_matches_legacy(legacy, selection):
    class Target:
        def __init__(self):
            self.data = SimpleNamespace(default_root_state=torch.arange(52, dtype=torch.float32).reshape(4, 13))
            self.writes = []
        def write_root_pose_to_sim_index(self, **kwargs):
            self.writes.append(kwargs)
        def write_root_velocity_to_sim_index(self, **kwargs):
            self.writes.append(kwargs)
    class Scene(dict):
        env_origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    old, new = Target(), Target()
    env = SimpleNamespace(num_envs=4, device="cpu", scene=Scene(cube_b=old))
    legacy.mdp.reset_target(env, selection)
    env.scene["cube_b"] = new
    legacy.task_env.reset_to_default(env, selection, ("cube_b",))
    for a, b in zip(old.writes, new.writes):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key], key)


@pytest.mark.parametrize("variant", ["teleop", "eval"])
@pytest.mark.parametrize("bodies", [("jaw",), ("gripper_right_link", "gripper_left_link")])
def test_env_cfg_plan_matches_legacy(legacy, variant, bodies):
    """Explicit expectation transcribed from the verbatim env_cfg; keep physics and event order."""
    m = legacy.task_env
    S, O, T, E = m.SensorPlan, m.ObjectPlan, m.TermPlan, m.Entity
    prefix = "sim.workspace.envs.task_env:"
    tracking = T("reset_tracking", prefix + "reset_task_tracking", {"task": TASK}, mode="reset")
    if variant == "teleop":
        events = (T("reset_target", prefix + "reset_to_default", {"names": ("cube_b",)}, mode="reset"),
                  T("reset_from_starts", "sim.workspace.envs.mdp:park_object", {"asset_cfg": E("cube_a")}, mode="reset"),
                  tracking)
    else:
        events = (T("reset_from_starts", "sim.workspace.envs.mdp.starts:reset_from_starts",
                    {"task": TASK, "starts_dir": None, "mode": "cycle", "random_fraction": 0., "seed": None,
                     "reset_robot": True}, mode="reset"), tracking,
                  T("check_eval_order", "sim.workspace.envs.mdp.starts:assert_reset_event_order", mode="startup"))
    import re
    expression = re.escape(bodies[0]) if len(bodies) == 1 else "(" + "|".join(map(re.escape, bodies)) + ")"
    scene = (S("contact_grasp", "{ENV_REGEX_NS}/Robot/" + expression, ("{ENV_REGEX_NS}/cube_a",)),
             O("cube_a", "{ENV_REGEX_NS}/cube_a", None, (.02, .02, .02), .008, (.02, .06, .9), .5, True,
               (.14, -.12, .05), (0., 0., 0.), "arm_base"),
             O("cube_b", "{ENV_REGEX_NS}/cube_b", None, (.02, .02, .02), .008, (.8, .04, .03), .5, True,
               (.20, -.08, .05), (0., 0., 0.), "arm_base"))
    terminations = () if variant == "teleop" else (
        T("time_out", "sim.workspace.envs.mdp:time_out", time_out=True),
        T("success", prefix + "task_success", {"task": TASK}, time_out=False))
    expected = m.TaskCfgPlan(scene, events, (), terminations, 4, None if variant == "teleop" else 450, variant != "teleop")
    task = load_task(TASK)
    actual = m.plan_task_cfg(task.spec, variant, SimpleNamespace(gripper=SimpleNamespace(contact_bodies=bodies)),
                             SimpleNamespace(appearance={}))
    assert actual == expected
    assert task.gym_id(variant) == "S2R-stack_two_cubes-" + variant


def test_exact_float32_lift_threshold_matches_legacy(legacy):
    """At z=0.01 the float32 >= rule differs from pick-place's strict acquisition rule."""
    env = environment(3)
    env.scene["cube_a"].data.root_pos_w[:, 2] = 0.
    legacy.mdp.reset_tracking(env, None)
    legacy.task_env.reset_task_tracking(env, None, TASK)
    threshold = torch.tensor(.01, dtype=torch.float32)
    env.scene["cube_a"].data.root_pos_w[:, 2] = torch.stack((
        torch.nextafter(threshold, torch.tensor(0.)), threshold,
        torch.nextafter(threshold, torch.tensor(1.))))
    env.scene["contact_grasp"].data.force_matrix_w[:, 0, 0, 2] = 3.
    env.common_step_counter = 1
    env.episode_length_buf[:] = 30
    old, new = legacy.mdp.stage_signals(env), legacy.task_env.task_signals(env, TASK)
    assert old["lifted"].tolist() == old["grasped"].tolist() == [False, True, True]
    for key in old:
        same(old[key], new[key], key)
