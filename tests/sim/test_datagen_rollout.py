"""Accepted data must have real final task success and one shared frame timeline."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.datasets.task_meta import load_task_meta
from core.tasks.base import Pose, StepState, Task
from core.tasks.loader import load_task
from sim.datagen.rollout import LiveRollout, SuccessRecorder
from sim.datagen.scripted import GenerationContext, parse_args


class ToyTask(Task):
    def signals(self, env):
        return env.signals

    def observe(self, state):
        return {name: bool(state.signals.get(name, False)) for name in self.stages}


class Writer:
    def __init__(self, root):
        self.dataset_root = root
        self.cameras = {"view": {"width": 2, "height": 2}}
        self.frames, self.saved = [], []
        self.cleared = 0

    @property
    def frame_count(self):
        return len(self.frames)

    @property
    def episode_index(self):
        return len(self.saved)

    def make_sim_frame(self, action, state, images):
        return {"action": np.asarray(action).copy(), "observation.state": np.asarray(state).copy(),
                "image": np.asarray(images["view"]).copy()}

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        index = self.episode_index
        self.saved.append(self.frames)
        self.frames = []
        return index

    def clear_episode_buffer(self):
        self.cleared += 1
        self.frames = []


@pytest.fixture
def attempt(tmp_path):
    original = load_task("stack_two_cubes").spec
    task = ToyTask(replace(original, stages=("positioned", "success")))
    arm = SimpleNamespace(name="four_axis", links=SimpleNamespace(base="mount"))
    writer = Writer(tmp_path)
    recorder = SuccessRecorder(writer, task, arm, {name: (.02,) * 3 for name in task.spec.tracked_objects}, 30,
                               control_offset_m=(.01, 0., 0.))
    return recorder, writer, task, arm


def frame(recorder, success, number=0):
    poses = {name: [number / 100, 0., .01, 1., 0., 0., 0.] for name in recorder.poses}
    recorder.append(np.arange(4) + number, np.arange(4) - number,
                    {"view": np.full((2, 2, 3), number, dtype=np.uint8)}, poses,
                    annotation={"phase": "move"})
    state = StepState({name: Pose((number / 100, 0, .01)) for name in recorder.poses},
                      Pose((0, 0, .1)), 0., {"positioned": number > 0, "success": success})
    recorder.observe(state)


def test_failed_attempt_is_excluded_without_metadata_or_saved_frames(attempt):
    recorder, writer, *_ = attempt
    frame(recorder, False)
    row = recorder.finish(attempt=4, seed=7)
    assert row["accepted"] is False and row["index"] is None
    assert not writer.saved and writer.cleared == 1
    assert not list(Path(writer.dataset_root).rglob("*.json"))


def test_success_commits_every_object_and_camera_on_the_action_timeline(attempt):
    recorder, writer, task, arm = attempt
    frame(recorder, False)
    frame(recorder, True, 1)
    row = recorder.finish(attempt=4, seed=7)
    assert row["accepted"] and row["index"] == 0 and row["first_success_step"] == 2
    assert row["frames"] == row["steps"] == len(writer.saved[0]) == 2
    assert row["camera_frames"] == {"view": 2}
    assert row["object_frames"] == {"cube_a": 2, "cube_b": 2}
    sidecar = load_task_meta(writer.dataset_root, 0, task=task, arm=arm, tracked_objects=task.spec.tracked_objects,
                            num_frames=2, fps=30, reference_frame="robot base link (mount)")
    assert sidecar["objects"]["cube_b"]["position"][1] == [.01, 0., .01]
    metadata = json.loads(Path(row["metadata_path"]).read_text())
    assert metadata["control_offset_m"] == [.01, 0., 0.]
    assert metadata["frames"][1]["subtask_term_signals"] == {"positioned": True, "success": True}
    assert metadata["last_state"]["signals"]["success"] is True
    np.testing.assert_array_equal(writer.saved[0][1]["action"], np.arange(4) + 1)


def test_late_loss_is_rejected_even_when_success_was_previously_latched(attempt):
    recorder, writer, *_ = attempt
    frame(recorder, True)
    frame(recorder, False, 1)
    row = recorder.finish()
    assert row["latched_success"] and not row["final_success"]
    assert not row["accepted"] and not row["success"] and not writer.saved
    assert row["reason"] == row["failure_mode"] == "success_lost_before_episode_end"


def test_unobserved_or_invalid_pose_frame_cannot_publish_an_episode(attempt):
    recorder, writer, *_ = attempt
    frame(recorder, True)
    recorder.poses["cube_b"][0][3:] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match="quaternion"):
        recorder.finish()
    assert not writer.saved
    recorder.abort()
    assert not writer.frames


def test_action_observation_count_mismatch_is_rejected_before_commit(attempt):
    recorder, writer, *_ = attempt
    frame(recorder, True)
    recorder.steps = 0
    with pytest.raises(ValueError, match="timelines"):
        recorder.finish()
    assert not writer.saved
    recorder.abort()


def test_numpy_adapter_supports_arrays_tensors_and_proxy_views():
    array = np.array([1., 2.])
    for value in (array, torch.tensor(array), SimpleNamespace(torch=torch.tensor(array))):
        np.testing.assert_array_equal(LiveRollout._numpy(value), array)


def test_gripper_interpolation_uses_selected_reversed_codec_endpoints():
    class Mapping:
        raw_mins = torch.tensor([-100., 120.])
        raw_maxs = torch.tensor([100., 20.])
        to_raw = staticmethod(lambda value: value.clone())
        to_sim = staticmethod(lambda value: value.clone())
    context = SimpleNamespace(mapping=Mapping(), arm=SimpleNamespace(joints=[SimpleNamespace(role="arm"), SimpleNamespace(role="gripper")]))
    result = GenerationContext.gripper_target(context, torch.tensor([[.25, 60.]]), .25)
    torch.testing.assert_close(result, torch.tensor([[.25, 95.]]))


def test_supplied_scripted_yaml_parses_without_launching_kit(tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs/datagen/scripted.yaml"
    args = parse_args(["--config_path", str(path), "--destination", str(tmp_path / "generated"), "--attempts", "2"])
    assert (args.arm, args.scene, args.task, args.variant) == ("so101", "rpd_meeting_room", "pick_place_cube", "teleop")
    assert args.headless and args.attempts == 2 and args.destination == tmp_path / "generated"
    assert args.expert_options["gripper_open_fraction"] == .45


def test_waypoint_requires_declared_orientation_convergence(monkeypatch):
    from core.tasks.expert import Waypoint
    from sim.datagen.scripted import execute_expert
    import sim.eval.scene
    monkeypatch.setattr(sim.eval.scene, "capture_scene", lambda *args: {"table_plane_world": [0., 0., 0.]})
    initial = StepState({}, Pose((0., 0., .1)), 0.)
    progress = SimpleNamespace(steps=0)
    controller_advances = []
    class IK:
        options = SimpleNamespace(orientation_tolerance_rad=.03)
        def solve(self, *args, **kwargs):
            controller_advances.append(kwargs.get("advance", True))
            return SimpleNamespace(joint_positions=torch.zeros(1, 4), position_error=[0.],
                                   orientation_error=[.2 if progress.steps < 2 else 0.])
    class Rollout:
        last_state = None
        recorder = SimpleNamespace(trace=SimpleNamespace(stages={}))
        snapshot = staticmethod(lambda: initial)
        control_pose = staticmethod(lambda state: state.ee)
        def step(self, *args, **kwargs):
            progress.steps += 1
            self.last_state = initial
    point = Waypoint("aligned", initial.ee, .5, duration_s=1., timeout_s=3.)
    expert = SimpleNamespace(plan=lambda context: [point])
    context = SimpleNamespace(raw=None, task=SimpleNamespace(is_success=lambda flags: False),
        args=SimpleNamespace(expert_options={}, fps=1, max_steps=10), ik=IK(), gripper_target=lambda target, fraction: target)
    assert execute_expert(context, Rollout(), expert) == "expert_finished_without_success"
    assert progress.steps == 2  # Position was satisfied on tick one; orientation was not.
    assert controller_advances == [True, False, True, False]


def test_live_step_records_before_action_and_observes_success_after_it(tmp_path):
    from core.arms.mapping import JointMapping
    from core.arms.spec import load_arm_spec
    original = load_arm_spec()[0]
    arm = replace(original, name="permuted_four_axis", joints=(
        original.joints[-1],
        replace(original.joints[1], name="slide", usd="Slide", type="prismatic", limits=(-.2, .2), initial=0.),
        original.joints[0], original.joints[4],
    ))
    mapping = JointMapping.from_arm(arm, device="cpu")
    task_spec = load_task("stack_two_cubes").spec
    task_spec = replace(task_spec, stages=("positioned", "success"))
    signal_calls = []
    class PhysicalStepTask(ToyTask):
        def signals(self, env):
            signal_calls.append(True)
            return {"positioned": torch.tensor([env.succeeded]), "success": torch.tensor([env.succeeded])}
    task = PhysicalStepTask(task_spec)
    writer = Writer(tmp_path)
    recorder = SuccessRecorder(writer, task, arm, {name: (.02,) * 3 for name in task.spec.tracked_objects}, 30)
    # USD order differs from the manifest, with an interleaved prismatic axis.
    order = [2, 0, 3, 1]
    usd_names = [arm.joints[index].usd for index in order]
    manifest_ids = [usd_names.index(joint.usd) for joint in arm.joints]
    robot = SimpleNamespace(
        find_joints=lambda names, **kwargs: ([usd_names.index(name) for name in names], list(names)),
        find_bodies=lambda name: ([0 if name == arm.links.base else 1], [name]),
        data=SimpleNamespace(joint_pos=torch.tensor([[.1, .2, -.3, .04]]),
            body_link_pos_w=torch.tensor([[[10., 0., 0.], [10., 0., .2]]]),
            body_link_quat_w=torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])),
    )
    objects = {name: SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[10. + (index + 1) / 10, (index + 1) / 10, .01]]),
        root_quat_w=torch.tensor([[0., 0., 0., 1.]]))) for index, name in enumerate(task.spec.objects)}
    pixels = torch.full((1, 2, 2, 3), 7, dtype=torch.uint8)
    class Env:
        num_envs = 1
        device = "cpu"
        succeeded = False
        def __init__(self):
            self.scene = {"robot": robot, **objects}
            self.observation_manager = SimpleNamespace(compute=lambda: {"visual": {"rgb_view": pixels}})
        def step(self, target):
            self.received_target = target.clone()
            robot.data.joint_pos[0, manifest_ids] = target[0]
            robot.data.body_link_pos_w.add_(1.)
            for obj in objects.values():
                obj.data.root_pos_w.add_(2.)
            pixels.fill_(255)
            self.succeeded = True
            return None, None, torch.tensor([False]), torch.tensor([False]), {}
    env = Env()
    target = torch.tensor([[.25, .07, .15, -.1]])
    raw_action = mapping.to_raw(target[0]).numpy()
    expected_state = mapping.to_raw(robot.data.joint_pos[0, manifest_ids]).numpy().copy()
    def target_fk(actual):
        torch.testing.assert_close(actual, target)
        return Pose((10.4, .3, .2))
    rollout = LiveRollout(env, task, arm, mapping, recorder, control_offset=(.01, 0., 0.), target_pose_fn=target_fk)
    with pytest.raises(ValueError, match="does not decode"):
        rollout.step(target, action_raw=raw_action + 1)
    assert writer.frame_count == 0 and not signal_calls
    rollout.step(target, action_raw=raw_action, target_eef_pose=Pose((11., 1., 1.)), phase="transport")
    torch.testing.assert_close(env.received_target, target)
    np.testing.assert_array_equal(writer.frames[0]["action"], raw_action)
    np.testing.assert_array_equal(writer.frames[0]["observation.state"], expected_state)
    np.testing.assert_array_equal(writer.frames[0]["image"], np.full((2, 2, 3), 7))
    np.testing.assert_allclose(recorder.poses["cube_a"][0][:3], [.1, .1, .01], atol=1e-6)
    assert signal_calls == [True] and recorder.trace.stages["success"]
    assert rollout.last_state.objects["cube_a"].position[0] > 12.
    annotation = recorder.annotations[0]
    np.testing.assert_allclose(np.array(annotation["eef_pose"])[:3, 3], [.01, 0., .2], atol=1e-6)
    np.testing.assert_allclose(np.array(annotation["target_eef_pose"])[:3, 3], [.4, .3, .2], atol=1e-6)
    np.testing.assert_allclose(np.array(annotation["requested_eef_pose"])[:3, 3], [1., 1., 1.], atol=1e-6)
    assert annotation["subtask_term_signals"] == {"positioned": True, "success": True}
    assert recorder.finish()["accepted"]


def test_controller_bias_resets_after_warmup_and_for_audit_replay(tmp_path, monkeypatch):
    from core.arms.mapping import JointMapping
    from core.arms.spec import load_arm_spec
    import sim.datagen.scripted
    arm = load_arm_spec()[0]
    events = []
    robot = SimpleNamespace(
        find_joints=lambda names, **kwargs: (list(range(len(names))), list(names)),
        find_bodies=lambda name: ([0 if name == arm.links.base else 1], [name]),
        data=SimpleNamespace(joint_pos=torch.zeros(1, len(arm.joints))),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", cfg=SimpleNamespace(seed=0), scene={"robot": robot},
                          reset=lambda: events.append("reset"), step=lambda action: events.append("warmup"))
    ik = SimpleNamespace(position_offset_m=np.zeros(3), reset=lambda: events.append("controller_reset"))
    task = load_task("stack_two_cubes")
    args = SimpleNamespace(expert_options={}, warmup_steps=2, fps=30)
    context = GenerationContext(args, env, task, arm, JointMapping.from_arm(arm, device="cpu"), None,
                                {}, Writer(tmp_path), ik, None)
    monkeypatch.setattr(sim.datagen.scripted, "reset_task_tracking", lambda raw: events.append("tracking_reset"))
    context.begin_attempt(101)
    assert events == ["reset", "warmup", "warmup", "tracking_reset", "controller_reset"]
    context.begin_attempt(102, reset=False, warmup=False)
    assert events[-1] == "controller_reset" and events.count("controller_reset") == 2
