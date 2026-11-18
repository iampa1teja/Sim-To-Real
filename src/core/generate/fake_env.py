"""Deterministic kinematic environment for the CPU generation pipeline."""

from dataclasses import replace as _replace
import math as _math

import numpy as _np

from core.generate.errors import EnvError as _EnvError
from core.generate.protocols import ExpertObs as _ExpertObs, ObjectState as _ObjectState
from core.scenes import _generation_table_height
from core.tasks.base import Pose as _Pose, episode_objects as _episode_objects


class SimulatedCrash(Exception):
    """An intentional process interruption, outside the generation error hierarchy."""


def _rotation(q):
    x, y, z, w = q
    return _np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _pose(matrix):
    r = matrix[:3, :3]
    # Select the largest quaternion component to avoid a division near zero.
    candidates = (1+r[0, 0]+r[1, 1]+r[2, 2], 1+r[0, 0]-r[1, 1]-r[2, 2],
                  1-r[0, 0]+r[1, 1]-r[2, 2], 1-r[0, 0]-r[1, 1]+r[2, 2])
    k = int(_np.argmax(candidates))
    a = _math.sqrt(max(0., candidates[k])) / 2
    b = 4*a
    if k == 0:
        q = ((r[2, 1]-r[1, 2])/b, (r[0, 2]-r[2, 0])/b, (r[1, 0]-r[0, 1])/b, a)
    elif k == 1:
        q = (a, (r[0, 1]+r[1, 0])/b, (r[0, 2]+r[2, 0])/b, (r[2, 1]-r[1, 2])/b)
    elif k == 2:
        q = ((r[0, 1]+r[1, 0])/b, a, (r[1, 2]+r[2, 1])/b, (r[0, 2]-r[2, 0])/b)
    else:
        q = ((r[0, 2]+r[2, 0])/b, (r[1, 2]+r[2, 1])/b, a, (r[1, 0]-r[0, 1])/b)
    return _Pose(tuple(float(v) for v in matrix[:3, 3]), tuple(float(v) for v in q))


def _matrix(pose):
    matrix = _np.eye(4)
    matrix[:3, :3] = _rotation(pose.quat_xyzw)
    matrix[:3, 3] = pose.position
    return matrix


def _yaw(pose):
    x, y, z, w = pose.quat_xyzw
    return _math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _dims(obj):
    return _np.asarray(obj.size if obj.size is not None else obj.params.get("size_outer", (0., 0., 0.)))


def _bottom(obj):
    return obj.size is None and obj.params.get("origin", "center") == "bottom"


def _local_xy(pose, xy):
    yaw = _yaw(pose)
    c, s = _math.cos(yaw), _math.sin(yaw)
    d = _np.asarray(xy) - pose.position[:2]
    return _np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1]])


class FakeEnv:
    def __init__(self, task, chain, gripper, *, fps=30, alpha=0.5, cameras={}, push_radius_m=0.01,
                 crash_at_step=None):
        self.fps = fps
        self.joint_names = tuple(chain.q_names)
        self.camera_names = tuple(cameras)
        self._task, self._chain, self._grip = task, chain, gripper
        self._alpha, self._cameras, self._push_radius = alpha, dict(cameras), push_radius_m
        self._crash_at, self._step_calls = crash_at_step, 0

    def reset(self, spec):
        if set(spec.objects) - self._task.spec.objects.keys() or len(spec.initial_q) != len(self.joint_names):
            raise _EnvError("Spec has an unknown object or incorrect initial joint count")
        self._definitions = _episode_objects(self._task.spec, spec)
        nominal_table = self._task.spec.raw.get("start", {}).get("table_z_m", 0.)
        self._table_z = _generation_table_height(spec.scene, spec.arm, nominal_table)
        self._spec, self._frame = spec, 0
        self._q = _np.array(spec.initial_q, dtype=_np.float64)
        self._objects = {name: spec.objects.get(name, _ObjectState(_Pose((0., 0., 0.)), active=False))
                         for name in self._task.spec.objects}
        self._start_z = {name: state.pose.position[2] for name, state in self._objects.items()}
        self._held, self._relative = None, None
        self._placed_streak, self._previous_streak, self._signals_frame = 0, 0, None
        self._update_frames()
        if spec.held_object is not None:
            name = spec.held_object
            if name not in self._objects or not self._objects[name].active:
                raise _EnvError("Held object is missing or inactive")
            obj = self._definitions[name]
            self._q[self._grip.index] = self._grip.angle_from_gap(float(min(_dims(obj)[:2])))
            self._update_frames()
            position = self._tcp[:3, 3].copy()
            if _bottom(obj):
                position[2] -= _dims(obj)[2]/2
            pose = _Pose(tuple(position), self._objects[name].pose.quat_xyzw)
            self._objects[name] = _replace(self._objects[name], pose=pose)
            self._start_z[name] = pose.position[2]
            self._attach(name)
        return self.observe()

    def _update_frames(self):
        self._tcp = self._chain.fk(self._q, "tcp")
        self._ee = self._chain.fk(self._q, "ee")

    def _attach(self, name):
        self._held = name
        self._relative = _np.linalg.inv(self._tcp) @ _matrix(self._objects[name].pose)

    def _drop(self, name):
        state = self._objects[name]
        x, y, _ = state.pose.position
        support = self._table_z
        for other, other_state in self._objects.items():
            if other == name or not other_state.active:
                continue
            obj = self._definitions[other]
            local = _local_xy(other_state.pose, (x, y))
            z = other_state.pose.position[2]
            inner = obj.params.get("inner_half_size")
            if inner is not None and _np.all(abs(local) <= inner):
                support = max(support, z + obj.params.get("floor_z", 0.))
            elif _np.all(abs(local) <= _dims(obj)[:2]/2):
                support = max(support, z + _dims(obj)[2]*(1 if _bottom(obj) else .5))
        obj = self._definitions[name]
        z = support + (0 if _bottom(obj) else _dims(obj)[2]/2)
        yaw = _yaw(state.pose)
        pose = _Pose((x, y, float(z)), (0., 0., _math.sin(yaw/2), _math.cos(yaw/2)))
        self._objects[name] = _replace(state, pose=pose)
        self._held = self._relative = None

    def step(self, q_target):
        call = self._step_calls
        self._step_calls += 1
        if call == self._crash_at:
            raise SimulatedCrash(f"Interrupted at environment step {call}")
        old_tcp = self._tcp.copy()
        old_gap = self._grip.gap_from_angle(self._q[self._grip.index])
        self._q += self._alpha * (_np.clip(q_target, self._chain.lower, self._chain.upper) - self._q)
        gap = self._grip.gap_from_angle(self._q[self._grip.index])
        if self._held is not None:
            width = min(_dims(self._definitions[self._held])[:2])
            if gap < width:
                self._q[self._grip.index] = self._grip.angle_from_gap(float(width))
                gap = float(width)
        self._update_frames()
        if self._held is not None:
            self._objects[self._held] = _replace(self._objects[self._held], pose=_pose(self._tcp @ self._relative))
        if self._held is None and gap < old_gap:
            candidates = []
            for name, state in self._objects.items():
                obj = self._definitions[name]
                if not state.active or "inner_half_size" in obj.params:
                    continue
                dims = _dims(obj)
                center = _np.array(state.pose.position)
                if _bottom(obj):
                    center[2] += dims[2]/2
                d = self._tcp[:3, :3].T @ (center-self._tcp[:3, 3])
                width = min(dims[:2])
                if gap <= width+.004 and _math.hypot(*d[:2]) <= width/2 and abs(d[2]) <= dims[2]/2:
                    candidates.append((float(_np.linalg.norm(d)), name, float(width)))
            if candidates:
                _, name, width = min(candidates)
                self._q[self._grip.index] = self._grip.angle_from_gap(width)
                gap = width
                self._update_frames()
                self._attach(name)
        if self._held is not None and gap > min(_dims(self._definitions[self._held])[:2])+.002:
            self._drop(self._held)
        delta = self._tcp[:2, 3]-old_tcp[:2, 3]
        for name, state in self._objects.items():
            obj = self._definitions[name]
            if not state.active or name == self._held or "inner_half_size" in obj.params:
                continue
            dims = _dims(obj)
            bottom = state.pose.position[2] - (0 if _bottom(obj) else dims[2]/2)
            if (bottom <= self._tcp[2, 3] <= bottom+dims[2]
                    and _np.all(abs(_local_xy(state.pose, self._tcp[:2, 3])) <= dims[:2]/2+self._push_radius)):
                p = _np.array(state.pose.position)
                p[:2] += delta
                self._objects[name] = _replace(state, pose=_Pose(tuple(p), state.pose.quat_xyzw))
        self._frame += 1
        return self.observe()

    def observe(self):
        target, container = self._spec.roles.get("target"), self._spec.roles.get("container")
        placed = False
        if self._held is None and target in self._objects and container in self._objects:
            obj, box = self._definitions[target], self._definitions[container]
            state, box_state = self._objects[target], self._objects[container]
            inner = box.params.get("inner_half_size")
            bottom = state.pose.position[2] - (0 if _bottom(obj) else _dims(obj)[2]/2)
            placed = bool(state.active and box_state.active and inner is not None
                          and _np.all(abs(_local_xy(box_state.pose, state.pose.position[:2])) <= inner)
                          and bottom-(box_state.pose.position[2]+box.params.get("floor_z", 0.)) <= .01)
        if self._signals_frame != self._frame:
            self._previous_streak = self._placed_streak
            self._signals_frame = self._frame
        self._placed_streak = self._previous_streak+1 if placed else 0
        success = self._task.spec.raw.get("success", {})
        holding = self._held is not None
        lifted = bool(holding and self._objects[self._held].pose.position[2]-self._start_z[self._held]
                      >= success.get("min_lift", .01))
        signals = dict(holding=holding, grasped=holding, lifted=lifted, placed=placed,
                       success=placed and self._placed_streak >= success.get("confirm_steps", 25))
        return _ExpertObs(self._frame, self._q, _pose(self._ee), _pose(self._tcp),
                          self._grip.gap_from_angle(self._q[self._grip.index]), self._objects, signals)

    def set_object_pose(self, name, pose):
        if name not in self._objects:
            raise _EnvError(f"Unknown object: {name}")
        self._objects[name] = _replace(self._objects[name], pose=pose)
        if self._held == name:
            self._held = self._relative = None

    def render(self):
        return {name: _np.zeros((*shape, 3), dtype=_np.uint8) for name, shape in self._cameras.items()}

    def close(self):
        pass
