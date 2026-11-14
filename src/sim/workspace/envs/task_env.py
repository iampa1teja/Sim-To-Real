"""Apply a two-file task (task.toml + task.py) to a scene env cfg, and run its stage checks.

``plan_task_cfg`` decides everything from the task, arm and scene specs and returns plain data:
objects (prims, spawn kind, materials), the jaw contact sensor and its filter order, reset events in
order, subtask observation terms, terminations, episode length and DR events. ``apply_variant`` is
the thin Isaac layer that turns the plan into Isaac Lab cfg objects. The runtime terms at the end
(reset, termination, observations, signals) evaluate the task's checks on a fresh ``Frame``.

Module top: stdlib, utils and core only, so the plan can be tested without Isaac.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

from core.tasks.loader import load_task, resolve
from sim.workspace.envs.frame import CONTACT_SENSOR, build_frame, contact_objects

ENV_NS = "{ENV_REGEX_NS}/"


@dataclass(frozen=True)
class Entity:
    """Stands for ``SceneEntityCfg(name)`` in plan params."""
    name: str


@dataclass(frozen=True)
class ObjectPlan:
    name: str
    prim_path: str
    usd: Any                     # Path for a USD object, None for a cuboid
    size: tuple | None
    mass: float | None
    color: tuple[float, float, float]
    roughness: float
    grasp_material: bool         # cuboids get the grasp contact material, as today
    position: tuple
    rotation_euler_degrees: tuple
    frame: str


@dataclass(frozen=True)
class SensorPlan:
    name: str
    prim_path: str
    filter_prim_paths: tuple[str, ...]


@dataclass(frozen=True)
class TermPlan:
    name: str
    func: str                    # "module:attribute"
    params: dict = field(default_factory=dict)
    mode: str | None = None      # events: startup/reset
    time_out: bool | None = None  # terminations


@dataclass(frozen=True)
class TaskCfgPlan:
    scene: tuple                 # SensorPlan and ObjectPlan in insertion order
    events: tuple[TermPlan, ...]
    subtask_terms: tuple[TermPlan, ...]
    terminations: tuple[TermPlan, ...]
    decimation: int
    episode_steps: int | None
    drop_unused_cameras: bool


def _material(name, spec, scene):
    values = scene.appearance.get("object_materials", {}).get(name, {})
    params = spec.params
    fallback = params.get("material", {})
    color = values.get("color", fallback.get("color", params.get("color", (1, 1, 1))))
    roughness = values.get("roughness", fallback.get("roughness", params.get("roughness", 0.5)))
    return tuple(float(value) for value in color), float(roughness)


def _prim(spec, name):
    return ENV_NS + spec.objects[name].params.get("prim", name)


def object_colors(spec, scene):
    """``[task.dr] object_colors`` with "from_setup" resolved to the object's scene colour."""
    colors = {}
    for name, palette in spec.raw.get("dr", {}).get("object_colors", {}).items():
        base = _material(name, spec.objects[name], scene)[0]
        colors[name] = {key: tuple(base) if value == "from_setup" else tuple(value) for key, value in palette.items()}
    return colors


def plan_task_cfg(spec, variant, arm, scene) -> TaskCfgPlan:
    if variant not in spec.variants:
        raise KeyError(f"Task {spec.name!r} has no variant {variant!r}; choices: {sorted(spec.variants)}")
    if arm.gripper is None or not arm.gripper.contact_bodies:
        raise ValueError(f"Task {spec.name!r} needs arm gripper contact_bodies")
    episode = spec.raw.get("episode", {})
    bodies = arm.gripper.contact_bodies
    pattern = bodies[0] if len(bodies) == 1 else "(" + "|".join(bodies) + ")"
    items = [SensorPlan(CONTACT_SENSOR, ENV_NS + "Robot/" + pattern,
                        tuple(_prim(spec, name) for name in contact_objects(spec)))]
    for name, item in spec.objects.items():
        color, roughness = _material(name, item, scene)
        items.append(ObjectPlan(
            name=name, prim_path=_prim(spec, name), usd=item.usd, size=item.size,
            mass=item.params.get("mass") if item.usd is None else None, color=color, roughness=roughness,
            grasp_material=item.usd is None, position=item.position,
            rotation_euler_degrees=item.rotation_euler_degrees, frame=item.params.get("frame", "world")))

    start_object = spec.start.get("object", spec.tracked_objects[0])
    tracking = TermPlan("reset_tracking", f"{__name__}:reset_task_tracking", {"task": spec.name}, mode="reset")
    if variant == "teleop":
        # The spawnable start object stays parked until sim.teleop.spawner places it at the jaw.
        events = [TermPlan("reset_from_starts", "sim.workspace.envs.mdp:park_object",
                           {"asset_cfg": Entity(start_object)}, mode="reset"), tracking]
    else:
        events = [TermPlan("reset_from_starts", "sim.workspace.envs.mdp.starts:reset_from_starts",
                           {"task": spec.name, "starts_dir": None, "mode": "cycle", "random_fraction": 0.0,
                            "seed": None, "reset_robot": True}, mode="reset"),
                  tracking,
                  TermPlan("check_eval_order", "sim.workspace.envs.mdp.starts:assert_reset_event_order", mode="startup")]
    if variant == "eval_dr":
        dr = spec.raw["dr"]
        events.append(TermPlan("eval_room_light", "sim.workspace.scene.appearance:randomize_scene_light_exposure",
                               {"exposure_range": tuple(dr.get("room_light_exposure_range", (-3.0, 1.0)))}, mode="reset"))
        # The Isaac layer fills color_names with every name in ROBOT_COLORS.
        events.append(TermPlan("eval_robot_color", "sim.workspace.envs.mdp:randomize_robot_color", mode="reset"))
        for name, colors in object_colors(spec, scene).items():
            events.append(TermPlan("eval_object_color", "sim.workspace.envs.mdp.object_color:randomize_object_color",
                                   {"colors": colors, "object_name": name}, mode="reset"))

    terms = tuple(TermPlan(key, f"{__name__}:stage_observation", {"task": spec.name, "stage": stage})
                  for key, stage in spec.raw.get("observations", {}).get("subtask_terms", {}).items())
    terminations = () if variant == "teleop" else (
        TermPlan("time_out", "sim.workspace.envs.mdp:time_out", time_out=True),
        TermPlan("success", f"{__name__}:task_success", {"task": spec.name}, time_out=False))
    return TaskCfgPlan(
        scene=tuple(items), events=tuple(events), subtask_terms=terms, terminations=terminations,
        decimation=episode.get("decimation", 4),
        episode_steps=None if variant == "teleop" else episode.get("eval_steps", 450),
        drop_unused_cameras=variant != "teleop",
    )


# ---------------------------------------------------------------------------
# Isaac layer
# ---------------------------------------------------------------------------


@cache
def _cfg_classes():
    from isaaclab.managers import ObservationGroupCfg as ObsGroup, TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    @configclass
    class SubtaskTermsCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class TaskTerminationsCfg:
        time_out: DoneTerm = None
        success: DoneTerm = None

    return SubtaskTermsCfg, TaskTerminationsCfg


def _params(params):
    from isaaclab.managers import SceneEntityCfg

    return {key: SceneEntityCfg(value.name) if isinstance(value, Entity) else value for key, value in params.items()}


def _object_cfg(item, scene, arm_pose):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from utils.geometry import Pose, transform_pose
    from sim.workspace.scene.cameras import euler_xyz_degrees_to_xyzw
    from sim.workspace.scene.contact_materials import (
        GRASP_FRICTION, GRASP_ROLLING_FRICTION, contact_material, contact_properties)

    material = sim_utils.PreviewSurfaceCfg(diffuse_color=item.color, roughness=item.roughness, metallic=0.0)
    if item.usd is not None:
        spawn = sim_utils.UsdFileCfg(usd_path=str(item.usd), visual_material=material)
    else:
        spawn = sim_utils.CuboidCfg(
            size=item.size,
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=item.mass),
            collision_props=contact_properties(),
            physics_material_path="physicsMaterial",
            physics_material=contact_material(GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION),
            visual_material=material,
        )
    pose = transform_pose(Pose(item.position, euler_xyz_degrees_to_xyzw(item.rotation_euler_degrees)),
                          item.frame, scene, arm_pose)
    return RigidObjectCfg(prim_path=item.prim_path, spawn=spawn,
                          init_state=RigidObjectCfg.InitialStateCfg(pos=pose.position, rot=pose.quat_xyzw))


def apply_variant(cfg, task, variant):
    from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.sensors import ContactSensorCfg
    from utils.arms.arms import load_arm_spec
    from utils.geometry import Pose
    from utils.scene import load_scene_spec
    from sim.workspace.envs.mdp import ROBOT_COLORS
    from sim.workspace.envs.scene_builder import drop_unused_camera_outputs

    scene = getattr(cfg, "scene_spec", None) or load_scene_spec()
    arm = getattr(cfg, "arm_spec", None) or load_arm_spec()[0]
    plan = plan_task_cfg(task.spec, variant, arm, scene)
    state = cfg.scene.robot.init_state
    arm_pose = Pose(state.pos, state.rot)
    for item in plan.scene:
        if isinstance(item, SensorPlan):
            setattr(cfg.scene, item.name, ContactSensorCfg(
                prim_path=item.prim_path, update_period=0.0, history_length=1, debug_vis=False,
                filter_prim_paths_expr=list(item.filter_prim_paths)))
        else:
            setattr(cfg.scene, item.name, _object_cfg(item, scene, arm_pose))
    for term in plan.events:
        params = _params(term.params)
        if term.name == "eval_robot_color":
            params = {"color_names": list(ROBOT_COLORS)}
        setattr(cfg.events, term.name, EventTerm(func=resolve(term.func), mode=term.mode, params=params))
    subtask_cfg, terminations_cfg = _cfg_classes()
    if plan.subtask_terms:
        group = subtask_cfg()
        for term in plan.subtask_terms:
            setattr(group, term.name, ObsTerm(func=resolve(term.func), params=dict(term.params)))
        cfg.observations.subtask_terms = group
    cfg.decimation = plan.decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.robot.spawn.activate_contact_sensors = True
    if plan.terminations:
        cfg.terminations = terminations_cfg(**{term.name: DoneTerm(
            func=resolve(term.func), time_out=term.time_out, params=_params(term.params)) for term in plan.terminations})
    if plan.episode_steps is not None:
        cfg.episode_length_s = plan.episode_steps * cfg.decimation * cfg.sim.dt
    if plan.drop_unused_cameras:
        drop_unused_camera_outputs(cfg)


def apply_teleop(cfg, task):
    apply_variant(cfg, task, "teleop")


def apply_eval(cfg, task):
    apply_variant(cfg, task, "eval")


def apply_eval_dr(cfg, task):
    apply_variant(cfg, task, "eval_dr")


# ---------------------------------------------------------------------------
# Runtime terms
# ---------------------------------------------------------------------------


def _task(task):
    return load_task(task) if isinstance(task, str) else task


def _ids(env, env_ids):
    import torch

    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def reset_task_tracking(env, env_ids, task):
    """Clear the reset environments' check state, after their new object poses were written."""
    task = _task(task)
    frame = build_frame(env, task, gripper=False)
    ids = _ids(env, env_ids)
    for check in task.checks:
        check.reset(ids, frame)
    if CONTACT_SENSOR in env.scene.keys():
        env.scene[CONTACT_SENSOR].reset(env_ids)


def task_signals(env, task):
    """Every stage's published signals (stage flags, ``holding``, ``reach_distance``) for core."""
    task = _task(task)
    frame = build_frame(env, task)
    signals = {}
    for name, check in task.stage_checks.items():
        signals.update(check.signals(frame, name))
    return signals


def task_success(env, task):
    """The ``success`` termination: the task's last stage."""
    task = _task(task)
    return task.stage_checks[task.stages[-1]].evaluate(build_frame(env, task))


def stage_observation(env, task, stage):
    task = _task(task)
    return task.stage_checks[stage].evaluate(build_frame(env, task)).float().unsqueeze(-1)
