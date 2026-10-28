"""Synthetic angle checks and guarded operator sessions; no hardware is opened."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.calibration_check import (
    CalibrationGapLogger, GAPS, LIMITATION, Thresholds, pose_measurement, reference_poses, summarize,
)
from sim_to_real_so101.utils.joint_calibration import (
    FOLLOWER_JOINTS, SETUP_PATH, file_sha256, follower_mapping_from_calibration, resolve_calibration, sim_mapping_parameters,
)
from sim_to_real_so101.utils.recording_alignment import (
    mark_recording_root, recording_contract, validate_recording_root,
)

BOUNDS = np.array([[-500., 500.]] * 6)
SETUP = json.loads(SETUP_PATH.read_text())
FIXTURE = ROOT / "tests/fixtures/rpd_robot.json"


def logger(path, **kwargs):
    return CalibrationGapLogger(path, joint_order=FOLLOWER_JOINTS, runtime_limits=BOUNDS,
                                provenance={"git_commit": "synthetic"}, **kwargs)


def record(trace, i, *, real=None, sim=None, command=None, target=None):
    real = np.zeros(6) if real is None else np.asarray(real)
    sim = real if sim is None else np.asarray(sim)
    return trace.record(commanded_deg=real if command is None else command, real_deg=real, sim_deg=sim,
                        sim_target_deg=sim if target is None else target, timestamp_ns=1_000_000_000 + i * 100_000_000)


def test_known_offsets_tracking_breakdown_verdicts_and_markdown(tmp_path):
    path = tmp_path / "offsets.jsonl"
    offsets = np.array([.4, 1.5, 3.5, -.6, -2., 4.5])
    with logger(path) as trace:
        for i in range(30):
            record(trace, i, real=np.ones(6) * 5, sim=5 + offsets, command=np.ones(6) * 4, target=4 + offsets)
    result = summarize(path)
    for i, name in enumerate(FOLLOWER_JOINTS):
        joint = result["gripper"] if name == "gripper" else result["joints"][name]
        for scope in ("all_frames", "static_not_at_limit"):
            stats = joint[scope][GAPS[0]]
            assert stats["mean_signed_deg"] == pytest.approx(offsets[i])
            assert stats["mean_abs_deg"] == pytest.approx(abs(offsets[i]))
            assert stats["median_deg"] == pytest.approx(offsets[i])
            assert stats["std_deg"] == pytest.approx(0)
            assert stats["p95_abs_deg"] == pytest.approx(abs(offsets[i]))
            assert stats["max_abs_deg"] == pytest.approx(abs(offsets[i]))
            assert stats["slope"] is None
            assert joint[scope][GAPS[1]]["mean_signed_deg"] == pytest.approx(1.)
            assert joint[scope][GAPS[2]]["mean_signed_deg"] == pytest.approx(1.)
        assert joint["counts"]["used"] == 27
    assert [j["verdict"] for j in result["joints"].values()] == ["PASS", "WARN", "FAIL", "PASS", "WARN"]
    assert "gripper" not in result["joints"] and result["gripper"]["verdict"] == "FAIL"
    assert json.loads(Path(str(path) + ".summary.json").read_text()) == result == trace.summary
    markdown = Path(str(path) + ".summary.md").read_text()
    assert LIMITATION in markdown and "Gripper (closed anchor unresolved)" in markdown
    assert "Slope" in markdown and "all frames" in markdown and "static not at limit" in markdown


def test_inverted_sign_is_detected_by_slope_and_correlation(tmp_path):
    path = tmp_path / "sign.jsonl"
    with logger(path) as trace:
        for i, angle in enumerate(np.linspace(-1., 1., 60)):
            record(trace, i, real=np.ones(6) * angle, sim=np.ones(6) * -angle)
    stats = trace.summary["joints"]["elbow_flex"]["static_not_at_limit"][GAPS[0]]
    assert stats["slope"] == pytest.approx(-2)
    assert stats["correlation"] == pytest.approx(-1)


def test_excludes_limits_and_motion_per_joint_and_uses_both_velocities(tmp_path):
    with logger(tmp_path / "filter.jsonl") as trace:
        for i in range(30):
            real = np.array([499.98, i * 2, 0, 0, 0, 0])
            sim = np.array([499.98, 0, i * 2, .5, 0, 0])
            record(trace, i, real=real, sim=sim)
    joints = trace.summary["joints"]
    assert joints["shoulder_pan"]["counts"]["excluded_at_limit"] == 30
    for name in FOLLOWER_JOINTS[:3]:
        assert joints[name]["counts"]["used"] == 0
        assert joints[name]["verdict"] == "NO_DATA"
    assert joints["wrist_flex"]["counts"]["used"] == 27
    assert joints["wrist_flex"]["verdict"] == "PASS"


def test_a_clamped_target_is_flagged_even_if_readback_has_not_reached_it(tmp_path):
    with logger(tmp_path / "clamp.jsonl") as trace:
        row = record(trace, 0, target=np.array([500, 0, 0, 0, 0, 0]))
    assert row["at_limit"] == [True, False, False, False, False, False]


def test_large_gaps_are_flagged_without_wrapping(tmp_path):
    path = tmp_path / "branch.jsonl"
    with logger(path) as trace:
        for i in range(10):
            row = record(trace, i, sim=[192, -360, 180, -181, 88, 0])
    assert row[GAPS[0]] == [192., -360., 180., -181., 88., 0.]
    assert row["suspected_branch_error"] == [True, True, False, True, False, False]
    assert trace.summary["joints"]["elbow_flex"]["counts"]["suspected_branch_error"] == 0
    assert trace.summary["joints"]["shoulder_pan"]["all_frames"][GAPS[0]]["mean_signed_deg"] == 192


@pytest.mark.parametrize("error", [RuntimeError("test exception"), KeyboardInterrupt()])
def test_summary_is_written_when_the_session_raises(tmp_path, error):
    path = tmp_path / "exception.jsonl"
    with pytest.raises(type(error)):
        with logger(path) as trace:
            for i in range(10):
                record(trace, i)
            raise error
    assert Path(str(path) + ".summary.json").is_file()
    assert Path(str(path) + ".summary.md").is_file()
    assert trace.summary["frames"] == 10


def test_header_only_finalize_but_offline_empty_log_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    with logger(path):
        pass
    with pytest.raises(ValueError, match="No calibration-check frames"):
        summarize(path)
    assert json.loads(Path(str(path) + ".summary.json").read_text())["frames"] == 0
    assert json.loads(path.read_text())["type"] == "provenance"


def test_existing_log_cannot_be_overwritten(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("preserved bytes\n")
    with pytest.raises(FileExistsError):
        logger(path)
    assert path.read_text() == "preserved bytes\n"


def test_unknown_previous_command_is_null_and_nonfinite_inputs_are_rejected(tmp_path):
    with logger(tmp_path / "startup.jsonl") as trace:
        row = trace.record(commanded_deg=None, real_deg=np.zeros(6), sim_deg=np.zeros(6), sim_target_deg=None)
        assert row[GAPS[1]] is None and row[GAPS[2]] is None
        with pytest.raises(ValueError, match="finite"):
            record(trace, 1, sim=[np.nan] * 6)
    assert trace.summary["joints"]["wrist_roll"]["all_frames"][GAPS[1]]["count"] == 0


def test_offline_threshold_override_recomputes_stillness_and_verdict(tmp_path):
    path = tmp_path / "thresholds.jsonl"
    with logger(path) as trace:
        for i in range(30):
            record(trace, i, real=np.ones(6) * i * .02, sim=np.ones(6) * (i * .02 + 2))
    assert trace.summary["joints"]["shoulder_pan"]["verdict"] == "WARN"
    result = summarize(path, Thresholds(static_velocity_deg_s=.1))
    assert result["joints"]["shoulder_pan"]["verdict"] == "NO_DATA"
    result = summarize(path, Thresholds(pass_deg=2.1))
    assert result["joints"]["shoulder_pan"]["verdict"] == "PASS"


def test_unset_environment_retains_current_mapping_and_stale_check():
    selection = resolve_calibration(environ={})
    assert selection.mapping == SETUP["sim_joint_mapping"]
    assert selection.follower_path is selection.leader_path is None
    stale = json.loads(json.dumps(SETUP))
    stale["sim_joint_mapping"]["elbow_flex"]["joint_min"] += 1
    with pytest.raises(ValueError, match="Stale"):
        resolve_calibration(environ={}, setup=stale)


def test_environment_override_fallback_provenance_and_loaded_mismatch(tmp_path):
    follower = tmp_path / "robots/so101_follower/f.json"
    leader = tmp_path / "teleoperators/so101_leader/l.json"
    for path in (follower, leader):
        path.parent.mkdir(parents=True)
        path.write_bytes(FIXTURE.read_bytes())
    env = {"HF_LEROBOT_CALIBRATION": str(tmp_path), "ROBOT_ID": "f", "TELEOP_ID": "l"}
    selection = resolve_calibration(environ=env)
    assert selection.follower_path == follower and selection.leader_path == leader
    assert selection.mapping == follower_mapping_from_calibration(follower, **sim_mapping_parameters())
    assert selection.provenance["follower_calibration"]["sha256"] == file_sha256(follower)
    assert selection.provenance["real_setup"]["sha256"] == file_sha256(SETUP_PATH)
    for role, path in (("follower", follower), ("leader", leader)):
        selection.verify_loaded(SimpleNamespace(calibration_fpath=path), role)
        with pytest.raises(ValueError, match="mismatch"):
            selection.verify_loaded(SimpleNamespace(calibration_fpath=tmp_path / "different.json"), role)
    other = tmp_path / "arbitrary-name.json"
    data = json.loads(FIXTURE.read_text())
    data["elbow_flex"]["range_min"] += 20
    other.write_text(json.dumps(data))
    selection = resolve_calibration(environ={**env, "SO101_FOLLOWER_CALIBRATION": str(other),
                                             "SO101_LEADER_CALIBRATION": str(other)})
    assert selection.follower_path == selection.leader_path == other
    assert selection.mapping != SETUP["sim_joint_mapping"]
    other.write_text(other.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        selection.verify_loaded(SimpleNamespace(calibration_fpath=other), "follower")


def test_selected_hash_guards_resume_even_when_angle_ranges_are_unchanged(tmp_path):
    path = tmp_path / "file.json"
    path.write_bytes(FIXTURE.read_bytes())
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path)})
    first = recording_contract(follower_id="f", leader_id="l", calibration_selection=selection)
    root = tmp_path / "dataset"
    mark_recording_root(root, first)
    path.write_text(path.read_text() + "\n")
    changed = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path)})
    assert changed.mapping == selection.mapping
    second = recording_contract(follower_id="f", leader_id="l", calibration_selection=changed)
    assert first["calibration_sha256"] != second["calibration_sha256"]
    with pytest.raises(ValueError, match="NEW dataset root"):
        validate_recording_root(root, second)


def load_cli():
    spec = importlib.util.spec_from_file_location("calibration_cli", ROOT / "scripts/calibration_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_offline_summary_watch_partial_lines_and_no_hardware_imports(tmp_path):
    cli = load_cli()
    path = tmp_path / "cli.jsonl"
    with logger(path) as trace:
        assert cli.latest_frame(path) is None
        for i in range(10):
            record(trace, i)
    assert cli.latest_frame(path)["frame"] == 9
    with path.open("a") as stream:
        stream.write('{"type":"frame","frame":10')
    assert cli.latest_frame(path)["frame"] == 9
    # Restore the complete synthetic trace for strict offline summarization.
    path.write_text(path.read_text().rsplit("\n", 1)[0] + "\n")
    output = tmp_path / "report.json"
    cli.main(["summarize", "--log", str(path), "--output", str(output)])
    assert json.loads(output.read_text())["frames"] == 10
    with pytest.raises(SystemExit):
        cli.main(["summarize", "--log", str(path), "--output", str(path)])
    result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'source'); "
                             "import sim_to_real_so101.utils.calibration_check; "
                             "assert not any(x in sys.modules for x in ('torch', 'isaaclab', 'lerobot', 'serial'))"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reference_poses_and_additive_candidates_do_not_wrap():
    from sim_to_real_so101.utils.joint_angle_check import summarize_candidates

    limits = [[-100, 100]] * 6
    poses = reference_poses(FOLLOWER_JOINTS, limits)
    assert len(poses) == 6 and not poses[0][1].any()
    mapping = {name: {"joint_min": -100, "joint_max": 100} for name in FOLLOWER_JOINTS}
    results = []
    for name, target in poses:
        units = target + 2
        units[-1] = 50
        samples = {"observations": [units.tolist()] * 30}
        result = pose_measurement(name, target, samples, mapping, FOLLOWER_JOINTS, limits)
        assert result["real_minus_target_deg"][0] == 2
        assert result["offsets_degrees"]["shoulder_pan"] == -2
        results.append(result)
    candidates = summarize_candidates(results, pan_reference="visual reference")
    assert all(j["status"] == "CONSISTENT" and j["offset_degrees"] == -2 for j in candidates["joints"].values())
    duplicates = summarize_candidates([results[0]] * 6, pan_reference="visual reference")
    assert all(j["status"] == "UNRESOLVED" and j["poses"] == 1 for j in duplicates["joints"].values())
    samples = {"observations": [[360, 0, 0, 0, 0, 50]] * 30}
    branch = pose_measurement("branch", np.zeros(6), samples, mapping, FOLLOWER_JOINTS, BOUNDS)
    assert branch["real_minus_target_deg"][0] == pytest.approx(360) and branch["suspected_branch_error"][0]
    samples["observations"][0] = [0] * 6
    assert not pose_measurement("moving", np.zeros(6), samples, mapping, FOLLOWER_JOINTS, BOUNDS)["accepted"]


def test_both_agents_observe_completed_tick_before_resets_and_new_commands():
    for name in ("lerobot_agent", "pick_place_agent"):
        text = (ROOT / f"source/sim_to_real_so101/scripts/{name}.py").read_text()
        tree = ast.parse(text)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.While))
        observation = next(node for node in ast.walk(loop) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "observe")
        mutations = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr in {"send_action", "step", "reset", "write_joint_state_to_sim"}]
        assert mutations and all(observation.lineno < node.lineno for node in mutations)
        assert "previous_command=previous_leader_action" in text
        assert "joint_angle_log.close" in text and "--calibration_check_log" in text


def test_pose_session_uses_write_guard_and_finalizes_on_interrupt(tmp_path):
    from test_joint_angle_check import FakeRobot

    cli = load_cli()
    path = tmp_path / "f.json"
    path.write_bytes(FIXTURE.read_bytes())
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path)})
    robot = FakeRobot()
    robot.calibration_fpath = path

    def interrupt():
        raise KeyboardInterrupt

    renderer = SimpleNamespace(show=lambda name, target: target.tolist(), wait_for_match=interrupt,
                               render_for=lambda seconds: None)
    log = tmp_path / "poses.jsonl"
    with log.open("x", buffering=1) as stream, pytest.raises(KeyboardInterrupt):
        cli.run_pose_session(robot, selection, renderer, stream, supported=True)
    result = json.loads(Path(str(log) + ".candidates.json").read_text())
    assert result["applied"] is False and not result["measurements"]
    assert all(value == 0 for value in result["connection_evidence"]["torque_enable_final"].values())
    assert not robot.bus.is_connected and path.read_bytes() == FIXTURE.read_bytes()


def test_pose_mode_requires_operator_support_before_rendering_or_connecting(tmp_path, monkeypatch):
    cli = load_cli()

    def forbid(*args, **kwargs):
        raise AssertionError("No sim or hardware session may start without support confirmation")

    monkeypatch.setattr(cli, "PoseRenderer", forbid)
    log = tmp_path / "not-started.jsonl"
    with pytest.raises(SystemExit):
        cli.main(["poses", "--log", str(log), "--follower-port", "unused", "--follower-id", "f"])
    assert not log.exists()


def test_pose_session_collects_six_poses_without_motor_or_calibration_writes(tmp_path):
    from test_joint_angle_check import FakeRobot

    cli = load_cli()
    path = tmp_path / "custom-name.json"
    path.write_bytes(FIXTURE.read_bytes())
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path)})
    robot = FakeRobot()
    robot.calibration_fpath = path
    renderer = SimpleNamespace(show=lambda name, target: target.tolist(), wait_for_match=lambda: None,
                               render_for=lambda seconds: None)
    log = tmp_path / "complete-poses.jsonl"
    with log.open("x", buffering=1) as stream:
        result = cli.run_pose_session(robot, selection, renderer, stream, supported=True)
    assert len(result["measurements"]) == 6 and robot.count == 180
    assert result["applied"] is False and not robot.bus.is_connected
    assert result["connection_evidence"]["calibration_unchanged"]
    assert path.read_bytes() == FIXTURE.read_bytes()


def test_selected_filename_is_used_by_both_lerobot_configs(tmp_path):
    path = tmp_path / "alternate-name.json"
    path.write_bytes(FIXTURE.read_bytes())
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path),
                                             "SO101_LEADER_CALIBRATION": str(path)})
    source = ast.parse((ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "LeRobotSO101Interface")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "make_cfg")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(SO101LeaderConfig=lambda **kwargs: kwargs, SO101FollowerConfig=lambda **kwargs: kwargs)
    exec(compile(module, "production_make_cfg", "exec"), namespace)
    for kind in ("leader", "follower"):
        interface = SimpleNamespace(kind=kind, id="logical-id", port="unused", calibration_selection=selection,
                                    make_cameras_cfg=lambda: {}, follower_options={})
        cfg = namespace["make_cfg"](interface)
        assert cfg["calibration_dir"] / (cfg["id"] + ".json") == path
        assert interface.id == "logical-id"


def test_telemetry_keeps_unclipped_angles_and_previous_command_and_reads_physx(tmp_path):
    torch = pytest.importorskip("torch")
    from sim_to_real_so101.utils.joint_angle_check import JointAngleTelemetry, ORDER
    from sim_to_real_so101.utils.joint_limits import runtime_joint_limits, USD_JOINTS
    from sim_to_real_so101.utils.calibration_check import units_to_degrees

    selection = resolve_calibration(environ={})
    limits = runtime_joint_limits()
    observation = dict.fromkeys(ORDER, 100.)
    previous_command = dict.fromkeys(ORDER, 50.)
    mapped = units_to_degrees([100] * 6, selection.mapping, FOLLOWER_JOINTS)
    clipped = np.clip(mapped, [limits[n]["joint_min"] for n in FOLLOWER_JOINTS],
                      [limits[n]["joint_max"] for n in FOLLOWER_JOINTS])
    interface = SimpleNamespace(get_raw_actions_tensor=lambda values: torch.tensor([values[n] for n in ORDER]),
                                get_mapped_actions_vectorized=lambda values: torch.deg2rad(torch.tensor(clipped)),
                                joint_mins=torch.tensor([limits[n]["joint_min"] for n in FOLLOWER_JOINTS]),
                                joint_maxs=torch.tensor([limits[n]["joint_max"] for n in FOLLOWER_JOINTS]))
    sim = SimpleNamespace(joint_names=list(USD_JOINTS),
                          root_physx_view=SimpleNamespace(get_dof_positions=lambda: torch.deg2rad(torch.tensor(clipped).unsqueeze(0))),
                          data=SimpleNamespace(joint_pos=torch.ones(1, 6) * 123))
    trace = JointAngleTelemetry(tmp_path / "adapter.jsonl", selection=selection)
    try:
        row = trace.observe(observation, interface, sim, torch.zeros(6), previous_command=previous_command)
    finally:
        trace.close()
    np.testing.assert_allclose(row["real_deg"], mapped)
    np.testing.assert_allclose(row["commanded_deg"], units_to_degrees([50] * 6, selection.mapping, FOLLOWER_JOINTS))
    np.testing.assert_allclose(row["sim_deg"], clipped)
    assert row["sim_target_deg"] == [0.] * 6
    assert row["real_deg"][0] > row["mapped_follower_degrees"][0]
