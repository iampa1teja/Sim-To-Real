"""Build the scene objects of a TOML task before Newton initializes its model."""

import os as _os

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.tasks.sampling import park_positions as _park_positions


def apply_teleop(cfg, task):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from sim.envs.mdp.resets import park_object
    from sim.scene.contact_materials import (GRASP_FRICTION, GRASP_ROLLING_FRICTION,
                                             contact_material, contact_properties)
    links = _load_arm_spec(_os.environ.get("S2R_ARM"))[0]["links"]
    movable = []
    for name, position in _park_positions(task.spec.objects).items():
        obj = task.spec.objects[name]
        path = "{ENV_REGEX_NS}/" + name
        properties = dict(rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(kinematic_enabled=not obj.rigid),
                          collision_props=contact_properties(),
                          physics_material_path="physicsMaterial",
                          physics_material=contact_material(GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION),
                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obj.params.get("color", (.5, .5, .5)))))
        if obj.params.get("mass") is not None:
            properties["mass_props"] = sim_utils.MassPropertiesCfg(mass=obj.params["mass"])
        spawn = (sim_utils.CuboidCfg(size=obj.size, **properties) if obj.size is not None
                 else sim_utils.UsdFileCfg(usd_path=str(obj.usd), **properties))
        setattr(cfg.scene, name, RigidObjectCfg(prim_path=path, spawn=spawn,
                                               init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(0., 0., 0., 1.))))
        setattr(cfg.events, f"park_{name}", EventTermCfg(func=park_object, mode="reset",
                                                       params=dict(asset_cfg=SceneEntityCfg(name), park_offset=position)))
        if obj.rigid:
            movable.append(path)
    if movable:
        cfg.scene.contact_grasp = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + links["moving_jaw"],
                                                   update_period=0., history_length=1, debug_vis=False,
                                                   filter_prim_paths_expr=movable)
        cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.decimation = task.spec.raw.get("episode", {}).get("decimation", 4)
    cfg.sim.render_interval = cfg.decimation


def apply_generate(cfg, task):
    apply_teleop(cfg, task)
    cfg.terminations = None
    cfg.episode_length_s = (task.spec.raw.get("expert", {}).get("max_steps", 600)+1)*cfg.decimation*cfg.sim.dt
