"""Deterministic, speed-limited expert skills with observed-state termination."""

from collections.abc import Mapping as _Mapping
from dataclasses import replace as _replace
import math as _math

import numpy as _np
from scipy.spatial.transform import Rotation as _Rotation

from core.arms.kinematics import IKWeights as _IKWeights
from core.expert.motion import MotionLimits, step_toward_joint as _step_toward_joint, cartesian_waypoint as _cartesian_waypoint
from core.generate.protocols import ExpertAction as _ExpertAction, PerturbationGroup as _PerturbationGroup
from core.generate.seeds import rng_for as _rng_for
from core.tasks.base import episode_objects as _episode_objects


def _rotation(pose):
    return _Rotation.from_quat(pose.quat_xyzw).as_matrix()


def _skew(v):
    x, y, z = v
    return _np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _yaw(pose):
    x, y, z, w = pose.quat_xyzw
    return _math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


class ScriptedExpert:
    def __init__(self, program, chain, gripper, *, lower, upper, limits, fps, stages, weights=None):
        self._base_program = self._program = program
        self._chain, self._grip, self._limits = chain, gripper, limits
        self._lower, self._upper = _np.array(lower, dtype=float, copy=True), _np.array(upper, dtype=float, copy=True)
        size = (len(chain.q_names),)
        if (self._lower.shape != size or self._upper.shape != size or _np.any(self._lower > self._upper)
                or _np.any(_np.isnan(self._lower)) or _np.any(_np.isnan(self._upper))
                or limits.max_joint_speed_rad_s.shape != size or type(fps) is not int or fps <= 0):
            raise ValueError("Invalid expert limits, speed shape or fps")
        self._fps, self._stages = fps, tuple(stages)
        self._weights = weights or _IKWeights()
        self._base_max_step = _np.array(limits.max_joint_speed_rad_s) / fps
        self._base_max_step[gripper.index] = limits.max_gripper_speed_rad_s / fps
        self._indices = {name: i for i, name in enumerate(chain.q_names)}
        self._arm_indices = _np.array([self._indices[name] for name in chain.chain_joints], dtype=int)
        self._approach, self._closing = _np.array(program.approach_axis), _np.array(program.closing_axis)
        self._local_basis = _np.column_stack((self._closing, _np.cross(self._approach, self._closing), self._approach))
        self._ready, self._done = False, False
        self._index = 0
        self.reentries: list[tuple[int, str]] = []

    @property
    def phase(self) -> str:
        return self._program.steps[self._index].name

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, spec, obs):
        if obs.q.shape != self._lower.shape:
            raise ValueError("Observation joint vector does not match the arm")
        base = self._base_program
        overrides = spec.expert_params.get("steps", {})
        steps = list(base.steps)
        for index, params in overrides.items():
            i = int(index)
            if params.get("q") == "home":
                from core.arms.spec import load_arm_spec
                arm, _ = load_arm_spec(spec.arm)
                home = {**arm["initial_state"]["joint_positions"], **arm["expert"].get("home_rad", {})}
                params = {**params, "q": {name: home[name] for name in arm["expert"]["arm_joints"]}}
            steps[i] = _replace(steps[i], params={**steps[i].params, **params})
        self._program = _replace(base, steps=tuple(steps), objects=_episode_objects(base, spec))
        self._speed_scale = spec.expert_params.get("speed_scale", base.speed_scale)
        self._max_step = self._base_max_step * self._speed_scale
        self._spec, self._q = spec, _np.array(obs.q, dtype=float, copy=True)
        self._blind = spec.expert_mode == "blind"
        self._poses = {name: state.pose for name, state in obs.objects.items() if state.active}
        self._reset_poses = dict(self._poses)
        self._roles = dict(spec.roles)
        self._held = spec.held_object
        self._index, self._state, self._emitted = 0, None, 0
        self._done, self._ready = False, True
        self.reentries = []
        self._visits = [0] * len(self._program.steps)
        self._pert = spec.perturbation if spec.perturbation is not None and spec.perturbation.group is _PerturbationGroup.EXPERT else None
        self._trigger, self._consumed, self._affected = None, False, False
        self._release_offset_index = None
        self._wrong, self._stall_onset = False, None
        if self._pert is not None:
            type_ = self._pert.type
            skill = {"grasp_offset": "grasp", "partial_close": "grasp", "early_close": "approach",
                     "early_release": "transport", "place_offset": "place"}.get(type_)
            for i, step in enumerate(self._program.steps):
                if (skill is not None and step.skill == skill) or (type_ == "target_offset" and step.stage is not None
                        and (self._pert.onset_stage is None or step.stage == self._pert.onset_stage)
                        and any(key in step.params for key in ("target", "q", "gap_m"))):
                    self._trigger = i
                    break
            if type_ == "wrong_target" and "target" in self._roles:
                target = self._roles["target"]
                set_name = self._program.objects[target].params.get("set")
                candidates = sorted(name for name in self._poses if name != target and set_name is not None
                                    and self._program.objects[name].params.get("set") == set_name)
                if candidates:
                    self._roles["target"] = candidates[int(_rng_for(spec.seed, "expert").integers(len(candidates)))]
                    self._wrong = True

    def _resolve(self, value):
        if not isinstance(value, str):
            return value
        if value.startswith("@goal."):
            result = self._spec.goal
            for part in value.split(".")[1:]:
                result = result[part] if isinstance(result, _Mapping) else result[int(part)]
            return result
        if value.startswith("@"):
            parts = value[1:].split(".")
            name = self._roles[parts[0]]
            return {"inner": name} if parts[1:] == ["inner"] else name
        return value

    def _dims(self, name):
        obj = self._program.objects[name]
        return _np.asarray(obj.size if obj.size is not None else obj.params.get("size_outer", (0, 0, 0)), dtype=float)

    def _bottom(self, name):
        obj = self._program.objects[name]
        return obj.size is None and obj.params.get("origin", "center") == "bottom"

    def _centre(self, name):
        result = _np.array(self._poses[name].position, dtype=float)
        if self._bottom(name):
            result[2] += self._dims(name)[2]/2
        return result

    def _support(self):
        if self._held is not None and self._held in self._reset_poses:
            z = self._reset_poses[self._held].position[2]
            return z if self._bottom(self._held) else z - self._dims(self._held)[2]/2
        return self._state["start"][2]

    def _destination(self, ref):
        value = self._resolve(ref)
        if isinstance(value, str):
            pose = self._poses[value]
            p = _np.array(pose.position, dtype=float)
            p[2] += self._dims(value)[2] * (1 if self._bottom(value) else 0.5)
            return p, value
        if isinstance(value, _Mapping):
            if "inner" in value:
                return self._destination(value["inner"])
            xy = (_np.asarray(value["xy_min"]) + value["xy_max"]) / 2
            return _np.r_[xy, self._support()], None
        point = _np.array(value, dtype=float)
        return (_np.r_[point, self._support()] if len(point) == 2 else point), None

    def _offset(self):
        params = self._pert.params
        amount, direction = params.get("xy_m", 0), params.get("dir_rad", 0)
        return _np.array([amount * _math.cos(direction), amount * _math.sin(direction), params.get("z_m", 0)], dtype=float)

    def _gap(self, value):
        value = self._resolve(value)
        if value == "open":
            return self._grip.gap_from_angle(self._grip.open_rad)
        if value == "closed":
            return self._grip.gap_from_angle(self._grip.closed_rad)
        return float(value)

    def _enter(self, obs):
        step = self._program.steps[self._index]
        self._affected = self._index == self._trigger and not self._consumed
        if self._affected:
            self._consumed = True
            if self._pert.type == "place_offset" and self._index + 1 < len(self._program.steps) and self._program.steps[self._index + 1].skill == "release":
                self._release_offset_index = self._index + 1
        if self._index == self._release_offset_index and self._visits[self._index] == 0:
            self._affected = True
        self._visits[self._index] += 1
        rotation = _rotation(obs.tcp)
        c = rotation @ self._closing
        start = _np.array(obs.tcp.position, dtype=float)
        self._state = {"start": start, "rotation": rotation, "yaw": _math.atan2(c[1], c[0]),
                       "carrot": start.copy(), "sub": "up" if step.skill == "grasp" else "gap" if step.skill == "release" else "above",
                       "sub_count": 0, "last_gap": obs.gripper_gap_m, "blocked": 0, "sweep": 0}
        self._emitted = 0
        if step.skill == "grasp":
            name = self._resolve(step.params["target"])
            point = self._centre(name) + [0, 0, step.params["z_offset_m"]]
            if self._affected and self._pert.type in ("grasp_offset", "target_offset"):
                point += self._offset()
            self._state["up_z"] = max(start[2], point[2] + self._dims(name)[2])
        if step.skill == "push":
            obj = self._resolve(step.params["target"])
            goal, _ = self._destination(step.params["goal"])
            d = goal[:2] - self._centre(obj)[:2]
            norm = float(_np.linalg.norm(d))
            self._state["direction"] = d/norm if norm else _np.array([1.0, 0.0])

    def _subphase(self, sub, obs):
        self._state.update(sub=sub, sub_count=0, carrot=_np.array(obs.tcp.position, dtype=float),
                           last_gap=obs.gripper_gap_m, blocked=0)

    def _goal(self):
        step, state = self._program.steps[self._index], self._state
        kind, params = step.skill, step.params
        if kind in ("lift", "retreat"):
            return state["start"] + [0, 0, params["height_m"] if kind == "lift" else params["dz_m"]]
        if kind in ("reach", "approach"):
            target = self._resolve(params["target"])
            point = _np.array(self._poses[target].position if isinstance(target, str) else target, dtype=float)
            goal = point + params["offset_m"]
        elif kind == "grasp":
            name = self._resolve(params["target"])
            point = self._centre(name) + [0, 0, params["z_offset_m"]]
            if self._affected and self._pert.type in ("grasp_offset", "target_offset"):
                point += self._offset()
            if state["sub"] == "up":
                return _np.r_[state["start"][:2], state["up_z"]]
            if state["sub"] == "over":
                return _np.r_[point[:2], state["up_z"]]
            return point
        elif kind in ("transport", "place"):
            goal, _ = self._destination(params["target"])
            if kind == "transport":
                goal = goal + [0, 0, params["height_m"]]
            else:
                goal = goal + params["offset_m"] + _np.array([0, 0, (self._dims(self._held)[2]/2 if self._held else 0) + params["clearance_m"]])
        elif kind == "touch":
            goal, _ = self._destination(params["target"])
        elif kind == "push":
            name = self._resolve(params["target"])
            point, direction = self._centre(name), state["direction"]
            width = min(self._dims(name)[:2])
            pre = point[:2] - direction * (width/2 + params["standoff_m"])
            if state["sub"] in ("above", "pre"):
                goal = _np.r_[pre, point[2] + params["depth_m"] + (0.04 if state["sub"] == "above" else 0)]
            else:
                dest, _ = self._destination(params["goal"])
                goal = _np.r_[dest[:2] - direction * width/2, point[2] + params["depth_m"]]
        else:
            return state["start"].copy()
        if self._affected and self._pert.type in ("target_offset", "place_offset") and "target" in params:
            goal = goal + self._offset()
        return goal

    def _orientation(self):
        step, state = self._program.steps[self._index], self._state
        if step.skill in ("lift", "transport", "place", "retreat") or step.params.get("orientation") == "keep":
            return state["rotation"]
        yaw = step.params.get("yaw", "object" if step.skill in ("grasp", "touch") else "keep")
        target = self._resolve(step.params["target"]) if "target" in step.params else None
        if yaw == "object" and isinstance(target, str):
            angle = _yaw(self._poses[target])
            state["yaw"] = angle + round((state["yaw"] - angle) / (_math.pi/2)) * (_math.pi/2)
        elif isinstance(yaw, (int, float)):
            state["yaw"] = float(yaw)
        angle = state["yaw"]
        if self._affected and self._pert.type == "grasp_offset":
            angle += self._pert.params.get("yaw_rad", 0)
        a = _np.array([0.0, 0.0, -1.0])
        c = _np.array([_math.cos(angle), _math.sin(angle), 0.0])
        return _np.column_stack((c, _np.cross(a, c), a)) @ self._local_basis.T

    def _joint_targets(self):
        value = self._resolve(self._program.steps[self._index].params["q"])
        if isinstance(value, _Mapping):
            result = {self._indices[name]: float(q) for name, q in value.items()}
        else:
            values = _np.asarray(value, dtype=float)
            if values.shape != self._q.shape:
                raise ValueError("Goal joint vector must match the arm")
            result = dict(enumerate(values))
        if self._affected and self._pert.type == "target_offset":
            result = {index: q + self._pert.params.get("joint_rad", 0) for index, q in result.items()}
        return {i: float(_np.clip(q, self._lower[i], self._upper[i])) for i, q in result.items()}

    def _sweep(self):
        params = self._program.steps[self._index].params
        joint, angles = self._resolve(params["joint"]), self._resolve(params["angles"])
        if isinstance(joint, _Mapping):
            joint = joint["joint"]
        if isinstance(angles, _Mapping):
            angles = angles["angles"]
        return self._indices[joint], angles

    def _close_gap(self):
        step = self._program.steps[self._index]
        gap = self._gap(step.params["close_gap_m"] if step.skill == "grasp" else step.params["gap_m"])
        if self._affected and self._pert.type == "partial_close":
            gap = min(self._dims(self._resolve(step.params["target"]))[:2]) + self._pert.params.get("extra_gap_m", 0)
        if self._affected and self._pert.type == "target_offset":
            gap += self._pert.params.get("gap_m", 0)
        return gap

    def _gap_done(self, obs, target, tolerance):
        state = self._state
        state["blocked"] = state["blocked"] + 1 if abs(obs.gripper_gap_m - state["last_gap"]) < 1e-4 else 0
        state["last_gap"] = obs.gripper_gap_m
        return abs(obs.gripper_gap_m - target) <= tolerance or state["blocked"] >= 5

    def _inside(self):
        params = self._program.steps[self._index].params
        point = self._centre(self._resolve(params["target"]))[:2]
        value = self._resolve(params["goal"])
        if isinstance(value, _Mapping):
            if "inner" in value:
                name = value["inner"]
                pose = self._poses[name]
                yaw = _yaw(pose)
                delta = point - pose.position[:2]
                local = _np.array([_math.cos(yaw)*delta[0] + _math.sin(yaw)*delta[1], -_math.sin(yaw)*delta[0] + _math.cos(yaw)*delta[1]])
                return bool(_np.all(_np.abs(local) <= self._program.objects[name].params["inner_half_size"]))
            return bool(_np.all(point >= value["xy_min"]) and _np.all(point <= value["xy_max"]))
        return float(_np.linalg.norm(point - _np.asarray(value)[:2])) <= 0.01

    def _terminated(self, obs):
        step, state = self._program.steps[self._index], self._state
        kind, p = step.skill, step.params
        if self._emitted >= step.budget:
            return True
        if kind == "wait":
            return self._emitted >= p["steps"]
        if kind == "move_joint":
            return all(abs(obs.q[i] - q) <= p["tol_rad"] for i, q in self._joint_targets().items())
        if kind == "sweep":
            index, angles = self._sweep()
            if abs(obs.q[index] - angles[state["sweep"]]) <= p["tol_rad"]:
                state["sweep"] += 1
            return state["sweep"] == len(angles)
        if kind == "set_gripper":
            return self._gap_done(obs, self._close_gap(), p["tol_m"])
        if kind in ("grasp", "release"):
            while True:
                sub = state["sub"]
                if kind == "grasp" and sub in ("up", "over", "down"):
                    if _np.linalg.norm(_np.asarray(obs.tcp.position) - self._goal()) > 0.005:
                        return False
                    self._subphase({"up": "over", "over": "down", "down": "close"}[sub], obs)
                    if sub == "down":
                        return False
                elif sub in ("close", "gap"):
                    if not self._gap_done(obs, self._close_gap(), 0.002):
                        return False
                    self._subphase("dwell", obs)
                    if p["dwell_steps"] > 0:
                        return False
                else:
                    return state["sub_count"] >= p["dwell_steps"]
        if kind == "push":
            if self._inside():
                return True
            if state["sub"] in ("above", "pre") and _np.linalg.norm(_np.asarray(obs.tcp.position) - self._goal()) <= 0.005:
                self._subphase("pre" if state["sub"] == "above" else "push", obs)
            return False
        tolerance = p.get("tol_m", p.get("contact_tol_m", 0.003 if kind == "place" else 0.005))
        return float(_np.linalg.norm(_np.asarray(obs.tcp.position) - self._goal())) <= tolerance

    def _ik(self, point, rotation):
        q, weights = self._q.copy(), self._weights
        at, ct = rotation @ self._approach, rotation @ self._closing
        indices = self._arm_indices
        lower, upper = self._lower[indices], self._upper[indices]
        damping = 1e-4 * _np.eye(7)
        for _ in range(12):
            transform = self._chain.fk(q)
            a, c = transform[:3, :3] @ self._approach, transform[:3, :3] @ self._closing
            s, k = float(at @ _np.cross(ct, c)), float(ct @ c)
            residual = _np.r_[weights.position*(transform[:3, 3] - point), weights.axis*_np.cross(a, at),
                              weights.roll*_math.atan2(s, k)]
            if float(residual @ residual) < 1e-14:
                break
            geometric = self._chain.jacobian(q)[:, self._arm_indices]
            dc = -_skew(c) @ geometric[3:]
            ds, dk = at @ _skew(ct) @ dc, ct @ dc
            jac = _np.vstack((weights.position*geometric[:3], weights.axis*_skew(at) @ _skew(a) @ geometric[3:],
                              weights.roll*(k*ds - s*dk)/max(s*s + k*k, 1e-20)))
            delta = -jac.T @ _np.linalg.solve(jac @ jac.T + damping, residual)
            # Re-solve over the movable joints when the first direction
            # pushes a joint farther into a bound.
            angles = q[indices]
            blocked = ((angles <= lower + 1e-10) & (delta < 0)) | ((angles >= upper - 1e-10) & (delta > 0))
            if _np.any(blocked):
                jac[:, blocked] = 0
                delta = -jac.T @ _np.linalg.solve(jac @ jac.T + damping, residual)
            scale = float(_np.max(_np.abs(delta))) if len(delta) else 0
            if scale > 0.2:
                delta *= 0.2/scale
            new = _np.clip(angles + delta, lower, upper)
            if _np.max(_np.abs(new - angles), initial=0) < 1e-10:
                break
            q[indices] = new
        return q

    def _desired(self, obs):
        step, state = self._program.steps[self._index], self._state
        kind = step.skill
        desired = self._q.copy()
        if kind == "move_joint":
            for index, value in self._joint_targets().items():
                desired[index] = value
        elif kind == "sweep":
            index, angles = self._sweep()
            desired[index] = angles[state["sweep"]]
        elif kind == "set_gripper" or (kind in ("grasp", "release") and state["sub"] in ("close", "gap", "dwell")):
            desired[self._grip.index] = self._grip.angle_from_gap(self._close_gap())
        elif kind != "wait":
            speed = self._limits.max_ee_speed_m_s * self._speed_scale
            state["carrot"] = _cartesian_waypoint(state["carrot"], self._goal(), speed / self._fps)
            desired = self._ik(state["carrot"], self._orientation())
            if kind == "grasp" and not self._blind and self.reentries:
                desired[self._grip.index] = self._grip.open_rad
        return desired

    def _reenter(self, obs, progress):
        index = progress.stage_index + 1
        if index >= len(self._stages):
            return None
        resumed = self._stages[index]
        first = next((i for i, step in enumerate(self._program.steps) if step.stage is not None
                      and self._stages.index(step.stage) >= index), len(self._program.steps))
        previous = [i for i, step in enumerate(self._program.steps[:first]) if step.stage is not None]
        self._index = previous[-1] + 1 if previous else 0
        self._index = min(self._index, len(self._program.steps) - 1)
        self._roles = dict(self._spec.roles)
        self._wrong = False
        self._release_offset_index = None
        self._state = None
        self.reentries.append((obs.frame, resumed))
        return resumed

    def _action(self, injected=False, resumed=None):
        return _ExpertAction(self._q.copy(), self.phase, self._done, bool(injected), resumed is not None, resumed)

    def act(self, obs, progress):
        if not self._ready:
            raise RuntimeError("ScriptedExpert.act requires reset")
        if not self._blind:
            self._poses.update({name: state.pose for name, state in obs.objects.items() if state.active})
        if self._pert is not None and self._pert.type == "stall":
            stage = self._pert.onset_stage
            if self._stall_onset is None and (stage is None or progress.furthest_index >= self._stages.index(stage)):
                self._stall_onset = obs.frame + self._pert.delay_steps
            if self._stall_onset is not None and self._stall_onset <= obs.frame < self._stall_onset + self._pert.params["steps"]:
                return self._action(True)
        if self._done:
            return self._action(self._wrong)
        detect = not self._blind and len(self.reentries) < self._program.max_retries
        resumed = self._reenter(obs, progress) if detect and progress.regressed_stage is not None else None
        if self._state is None:
            self._enter(obs)
        if resumed is None and self._emitted > 0 and self._terminated(obs):
            step = self._program.steps[self._index]
            if step.skill == "grasp":
                self._held = self._resolve(step.params["target"])
            elif step.skill == "release":
                self._held = None
            failed_stage = step.stage is not None and step.skill != "approach" and progress.stage_index < self._stages.index(step.stage)
            if detect and (failed_stage or (self._wrong and step.skill == "grasp")):
                resumed = self._reenter(obs, progress)
            if resumed is None:
                if self._index == len(self._program.steps) - 1:
                    self._done = True
                    return self._action(self._wrong)
                self._index += 1
            self._enter(obs)
        injected = self._wrong or self._affected
        desired = self._desired(obs)
        if self._affected and self._pert.type in ("early_close", "early_release"):
            injected = self._emitted >= self._pert.delay_steps
            if self._pert.type == "early_release" and obs.gripper_gap_m >= self._grip.gap_from_angle(self._grip.open_rad) - 0.002:
                injected = False
            if injected:
                desired[self._grip.index] = self._grip.closed_rad if self._pert.type == "early_close" else self._grip.open_rad
        self._q = _np.clip(_step_toward_joint(self._q, desired, self._max_step), self._lower, self._upper)
        self._emitted += 1
        self._state["sub_count"] += 1
        return self._action(injected, resumed)
