"""CPU episode execution and crash-resumable dataset generation."""

from copy import copy as _copy
from dataclasses import dataclass as _dataclass, replace as _replace
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType
import math as _math
import time as _time

import numpy as _np

from core.arms.gripper import Gripper as _Gripper
from core.arms.kinematics import KinematicChain as _KinematicChain
from core.arms.spec import load_arm_spec as _load_arm_spec
from core.datasets.task_meta import write_sidecar as _write_sidecar
from core.expert.machine import ScriptedExpert as _ScriptedExpert
from core.expert.motion import MotionLimits as _MotionLimits, expert_joint_limits as _expert_joint_limits
from core.expert.program import parse_program as _parse_program
from core.generate.errors import (AttemptBudgetExceeded as _AttemptBudgetExceeded,
                                  ResumeError as _ResumeError,
                                  ScenarioSpaceExhausted as _ScenarioSpaceExhausted)
from core.generate.fake_env import SimulatedCrash as _SimulatedCrash
from core.generate.fingerprint import (scenario_key as _scenario_key, spec_fp as _spec_fp, traj_fp as _traj_fp,
                                       appearance_fp as _appearance_fp, episode_fp as _episode_fp)
from core.generate.labels import (accept as _accept, build_phases as _build_phases, build_recovery as _build_recovery,
                                  build_segments as _build_segments, frame_labels as _frame_labels)
from core.generate.manifest import (Manifest as _Manifest, ManifestLine as _ManifestLine, reconcile as _reconcile,
                                    write_plan_json as _write_plan_json)
from core.generate.perturbations import WorldPerturber as _WorldPerturber, ou_noise as _ou_noise
from core.generate.plan import KINDS as _KINDS, iter_plan as _iter_plan, shard_attempts as _shard_attempts, shard_quota as _shard_quota
from core.generate.protocols import (AttemptStatus as _Status, EpisodeResult as _EpisodeResult,
                                     Fingerprints as _Fingerprints, Kind as _Kind, PerturbationGroup as _Group,
                                     RejectionReason as _Reason, StepRecord as _StepRecord)
from core.generate.seeds import rng_for as _rng_for
from core.generate.specs import build_spec as _build_spec, validate_physics as _validate_physics
from core.scenes import _generation_table_height
from core.tasks.base import CheckContext as _CheckContext, episode_objects as _episode_objects
from core.tasks.loader import load_task as _load_task
from core.tasks.sampling import validate_workspace as _validate_workspace
from core.utils.hashing import file_sha256 as _file_sha256


def home_q(arm, *, setup=None):
    spec, _ = _load_arm_spec(arm)
    values = {**spec["initial_state"]["joint_positions"], **spec.get("expert", {}).get("home_rad", {})}
    return tuple(float(values[name]) for name in spec["joints"]["names"])


def make_expert_factory(task, arm, *, setup=None):
    arm_spec, _ = _load_arm_spec(arm)
    lower, upper = _expert_joint_limits(arm, setup=setup)
    program = _parse_program(task.spec.raw["expert"], task=task, arm_spec=arm_spec, joint_limits=(lower, upper))
    chain, grip, limits = _KinematicChain.from_arm(arm), _Gripper.from_arm(arm), _MotionLimits.from_arm(arm)
    return lambda spec: _ScriptedExpert(program, chain, grip, lower=lower, upper=upper, limits=limits,
                                        fps=spec.fps, stages=task.stages)


def _task_for_scene(task, scene, arm):
    height = _generation_table_height(scene, arm)
    if height is None:
        return task
    raw = task.spec.raw
    placed = _copy(task)
    placed.spec = _replace(task.spec, raw={**raw, "start": {**raw.get("start", {}), "table_z_m": height}})
    return placed


def make_spec_fn(task, arm, *, scene, noise=None, weights_override=None, fps=30, setup=None, reach_fn=None):
    task = _task_for_scene(task, scene, arm)
    spec, _ = _load_arm_spec(arm)
    workspace = _validate_workspace(spec["workspace"]) if "workspace" in spec else None
    _validate_physics(task.spec)
    initial, names = home_q(arm, setup=setup), tuple(spec["joints"]["names"])
    limits, joints = _expert_joint_limits(arm, setup=setup), tuple(spec["expert"]["arm_joints"])
    start_pose = task.spec.raw.get("start", {}).get("initial_q")
    tcp_fn = (_KinematicChain.from_arm(arm).fk_pose
              if isinstance(start_pose, dict) and "around" in start_pose else None)
    return lambda kind, index, seed: _build_spec(task, kind, index, seed, arm=arm, scene=scene,
                                                home_q=initial, reach_fn=reach_fn, noise=noise,
                                                weights_override=weights_override, fps=fps,
                                                joint_names=names, joint_limits=limits, arm_joints=joints,
                                                workspace=workspace, tcp_fn=tcp_fn)


def _finite_obs(obs):
    values = [obs.q, obs.ee.position, obs.ee.quat_xyzw, obs.tcp.position, obs.tcp.quat_xyzw, obs.gripper_gap_m]
    values.extend(v for state in obs.objects.values() for v in (state.pose.position, state.pose.quat_xyzw))
    return all(_np.all(_np.isfinite(v)) for v in values)


def run_episode(spec, task, env, expert, writer, mapping, *, noise_rng=None, perturber_rng=None):
    tracker, ctx = task.make_tracker(), _CheckContext.from_spec(spec, task.spec)
    perturber = _WorldPerturber(spec, task.spec, perturber_rng if perturber_rng is not None else _rng_for(spec.seed, "perturbation"))
    noise = (_ou_noise(spec.noise, ~mapping.gripper_mask, spec.max_steps,
                       noise_rng if noise_rng is not None else _rng_for(spec.seed, "noise"))
             if spec.noise is not None else _np.zeros((spec.max_steps, len(spec.initial_q))))
    records, reentries, error, obs = [], [], None, None
    last_checks, last_state = {}, None
    done_frames = 0
    settle = max(1, task.spec.raw.get("expert", {}).get("settle_steps", 10) or 10)
    try:
        obs = env.reset(spec)
    except _SimulatedCrash:
        raise
    except Exception:
        error = _Reason.ENV_ERROR
    if error is None:
        expert.reset(spec, obs)
        for t in range(spec.max_steps):
            if not _finite_obs(obs):
                error = _Reason.INVALID_STATE
                break
            state = obs.step_state(env.joint_names)
            checks = task.observe_checks(state, ctx)
            progress = tracker.update(checks)
            action = expert.act(obs, progress)
            if action.reentered:
                tracker.rewind_from(action.resumed_stage)
            desired = action.q_target + noise[t]
            target = mapping.clamp_radians(desired)
            if not _np.all(_np.isfinite(desired)) or not _np.all(_np.isfinite(target)):
                error = _Reason.INVALID_STATE
                break
            try:
                images = env.render()
            except _SimulatedCrash:
                raise
            except Exception:
                error = _Reason.ENV_ERROR
                break
            labels = _frame_labels(tracker.stage_index, action.injected, obs.ee)
            raw_action, raw_state = mapping.to_raw(target), mapping.to_raw(obs.q)
            writer.add_frame(raw_action, raw_state, images, spec.instruction, labels)
            records.append(_StepRecord(t, raw_state, raw_action, labels, action.phase, action.injected,
                                       action.reentered, {name: obj.pose for name, obj in obs.objects.items() if obj.active}))
            last_checks, last_state = checks, state
            if action.reentered:
                reentries.append((t, action.resumed_stage))
            done_frames = done_frames+1 if action.done else 0
            if done_frames >= settle or t == spec.max_steps-1:
                break
            try:
                obs = env.step(target)
                if perturber.due(progress, t):
                    perturber.apply(env, obs)
                    obs = env.observe()
            except _SimulatedCrash:
                raise
            except Exception:
                error = _Reason.ENV_ERROR
                break
    n = len(records)
    injected = [r.frame for r in records if r.injected]
    is_world = spec.perturbation is not None and spec.perturbation.group is _Group.WORLD
    onset = perturber.onset_frame if is_world else (injected[0] if injected else None)
    applied = perturber.applied if is_world else bool(injected)
    end = injected[-1]+1 if injected else None
    outcome = task.outcome(tracker, last_checks, last_state) if last_state is not None else None
    reason = error or _accept(spec.kind, success=outcome.success if outcome else False, applied=applied,
                               reentered=bool(reentries), reentered_before_onset=onset is not None and any(t < onset for t, _ in reentries))
    stages = tuple(r.labels.stage for r in records)
    recovery = _build_recovery(reentries, stages, task.stages) if spec.kind is _Kind.RECOVERY and reentries else None
    segments = (_build_segments(n, spec.kind, onset=onset, injected_end=end,
                                detected=recovery.detected_frame if recovery else None,
                                back_on_plan=recovery.back_on_plan_frame if recovery else None,
                                phases=[r.phase for r in records]) if n else ())
    traj = appearance = episode = None
    if n and error is None:
        objects0 = {name: tuple(round(float(v), 6) for v in pose.position) for name, pose in records[0].objects.items()}
        traj = _traj_fp(_np.stack([r.action_raw for r in records]), _np.stack([r.state_raw for r in records]), objects0)
        appearance = _appearance_fp({name: obj.color for name, obj in spec.objects.items() if obj.active}, spec.instruction, None, None)
        episode = _episode_fp(traj, appearance)
    fp = _Fingerprints(_scenario_key(spec), _spec_fp(spec), traj, appearance, episode)
    result = _EpisodeResult(spec.task, spec.kind, spec.attempt_index, spec.seed,
                            _Status.ACCEPTED if reason is None else _Status.REJECTED, reason, None, n,
                            outcome.success if outcome else None, outcome.failure_mode if outcome else None,
                            outcome.furthest_stage if outcome else None, stages,
                            tuple(r.labels.imitate for r in records), segments, onset, applied, recovery, fp)
    return result, records


@_dataclass(frozen=True)
class GenerationSummary:
    accepted: object
    rejected: object
    skipped: int
    attempts: int

    def __post_init__(self):
        object.__setattr__(self, "accepted", _MappingProxyType(dict(self.accepted)))
        object.__setattr__(self, "rejected", _MappingProxyType(dict(self.rejected)))


def _sidecar(spec, task, env, result, records, index, shard, base, git_commit):
    objects = {}
    definitions = _episode_objects(task.spec, spec)
    for name, state in spec.objects.items():
        if not state.active:
            continue
        obj = definitions[name]
        dims = obj.size if obj.size is not None else obj.params.get("size_outer", (0., 0., 0.))
        roles = sorted(role for role, value in spec.roles.items() if value == name)
        object_set = next((key for key, value in task.spec.object_sets.items() if name in value.slot_names()), None)
        poses = [r.objects[name] for r in records]
        objects[name] = dict(set=object_set, size_m=[float(v) for v in dims], color=state.color, active=True,
                             role=roles[0] if roles else None, position=[[float(v) for v in p.position] for p in poses],
                             orientation_wxyz=[[float(v) for v in (p.quat_xyzw[3], *p.quat_xyzw[:3])] for p in poses])
    perturbation = (None if spec.perturbation is None else
                    dict(spec.perturbation.to_json(), onset_frame=result.perturbation_onset_frame, applied=result.perturbation_applied))
    attempt = dict(index=spec.attempt_index, seed=spec.seed, shard=list(shard), spec_fp=result.fingerprints.spec_fp)
    if spec.physics or spec.expert_params:
        payload = spec.to_json()
        attempt.update({key: payload[key] for key in ("physics", "expert_params") if key in payload})
    return dict(schema_version=2, episode_index=index, num_frames=result.num_frames, fps=spec.fps,
                task=spec.task, arm=spec.arm, scene=spec.scene, reference_frame=f"robot base link ({base})",
                units="m", angle_units="rad", joint_names=list(env.joint_names), objects=objects,
                ee_pose=dict(position=[r.labels.ee_pose_wxyz[:3].tolist() for r in records],
                             orientation_wxyz=[r.labels.ee_pose_wxyz[3:].tolist() for r in records]),
                instruction=spec.instruction, roles=dict(spec.roles), kind=spec.kind.value,
                success=result.success, failure_mode=result.failure_mode, furthest_stage=result.furthest_stage,
                stages=list(task.stages), stage=list(result.stage), imitate=list(result.imitate),
                phases=_build_phases([r.phase for r in records]), segments=[s.to_json() for s in result.segments],
                perturbation=perturbation, recovery=result.recovery.to_json() if result.recovery else None,
                noise=spec.noise.to_json() if spec.noise else None,
                attempt=attempt,
                fingerprints=result.fingerprints.to_json(), appearance=dict(light_exposure=None, robot_color=None),
                source=None, generator=dict(name="scripted_expert", git_commit=git_commit), legacy_v1=False)


def _summary(lines):
    accepted, rejected, skipped = dict.fromkeys(_KINDS, 0), {}, 0
    for line in lines:
        if line.status is _Status.ACCEPTED:
            accepted[line.kind] += 1
        elif line.status is _Status.REJECTED:
            rejected[line.reason.value] = rejected.get(line.reason.value, 0)+1
        else:
            skipped += 1
    return GenerationSummary(accepted, rejected, skipped, len(lines))


def run_generation(cfg, task_name, env, writer, *, expert_factory, mapping, spec_fn, git_commit="unknown"):
    task = _task_for_scene(_load_task(task_name), cfg.scene, cfg.arm)
    root = _Path(writer.root)
    arm, arm_dir = _load_arm_spec(cfg.arm)
    totals = {kind: getattr(cfg.quotas[task_name], kind.value) for kind in _KINDS}
    shard, n = cfg.shard
    quotas = {kind: _shard_quota(value, shard, n) for kind, value in totals.items()}
    _write_plan_json(root, dict(v=1, config_sha256=cfg.sha256, config=cfg.to_json(), master_seed=cfg.master_seed,
                               git_commit=git_commit, shard=list(cfg.shard), task=task_name,
                               task_toml_sha256=_file_sha256(task.spec.task_dir / "task.toml"), arm=cfg.arm,
                               arm_files_sha256={name: _file_sha256(arm_dir/name) for name in ("arm.toml", "kinematics.json")},
                               scene=cfg.scene, quotas={k.value: v for k, v in totals.items()},
                               holdout=dict(colors=list(task.spec.holdout.colors), counts=list(task.spec.holdout.counts)),
                               created_at=_datetime.now(_timezone.utc).isoformat()))
    manifest = _Manifest(root / "meta" / "generation_manifest.jsonl")
    _reconcile(root, writer, manifest)
    old_lines = manifest.lines()
    old = {(line.kind, line.attempt_index): line for line in old_lines}
    trajectories = {line.traj_fp for line in old_lines if line.status is _Status.ACCEPTED}
    accepted, executed, skips = (dict.fromkeys(_KINDS, 0) for _ in range(3))
    attempts = _shard_attempts(_iter_plan(task_name, cfg.master_seed, spec_fn,
                                          unique_across_kinds=cfg.scenario_unique_across_kinds,
                                          near_duplicates=True), shard, n)
    for planned in attempts:
        if all(accepted[k] >= quotas[k] for k in _KINDS):
            break
        kind = planned.kind
        if accepted[kind] >= quotas[kind]:
            continue
        line = old.get((kind, planned.index))
        if line is not None and line.spec_fp != (_spec_fp(planned.spec) if planned.spec is not None else None):
            raise _ResumeError(f"{task_name}: stored attempt {kind.value}/{planned.index} has a different episode spec")
        if line is None:
            started = _time.perf_counter()
            if planned.skip is not None:
                fp = _Fingerprints(planned.scenario_key, _spec_fp(planned.spec) if planned.spec else None, None, None, None)
                result = _EpisodeResult(task_name, kind, planned.index, planned.seed, _Status.SKIPPED,
                                        planned.skip, None, 0, None, None, None, (), (), (), None, False, None, fp)
                writer.discard_episode()
            else:
                if executed[kind] >= _math.ceil(cfg.max_attempt_factor*quotas[kind]):
                    raise _AttemptBudgetExceeded(f"{task_name}: {kind.value} attempt budget exhausted")
                spec = planned.spec
                result, records = run_episode(spec, task, env, expert_factory(spec), writer, mapping)
                if result.status is _Status.ACCEPTED and result.fingerprints.traj_fp in trajectories:
                    result = _replace(result, status=_Status.REJECTED, reason=_Reason.DUPLICATE_TRAJECTORY)
                if result.status is _Status.ACCEPTED:
                    index = writer.episode_count()
                    _write_sidecar(root, _sidecar(spec, task, env, result, records, index, cfg.shard, arm["links"]["base"], git_commit))
                    saved = writer.save_episode()
                    if saved != index:
                        raise RuntimeError("Writer committed an unexpected episode index")
                    result = _replace(result, episode_index=index)
                    trajectories.add(result.fingerprints.traj_fp)
                else:
                    writer.discard_episode()
            line = _ManifestLine.from_result(result, shard=cfg.shard,
                                              perturbation=planned.spec.perturbation.type if planned.spec and planned.spec.perturbation else None,
                                              wall_s=_time.perf_counter()-started)
            manifest.append(line)
        if line.status is _Status.SKIPPED:
            skips[kind] += 1
            if skips[kind] >= cfg.max_consecutive_skips:
                raise _ScenarioSpaceExhausted(f"{task_name}: {kind.value} consecutive skips exhausted")
        else:
            executed[kind] += 1
            skips[kind] = 0
            if line.status is _Status.ACCEPTED:
                accepted[kind] += 1
    writer.finalize()
    return _summary(manifest.lines())
