"""Build the scene objects of a TOML task before Newton initializes its model."""

import os as _os

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.tasks.sampling import park_positions as _park_positions


def apply_teleop(cfg, task):
    _apply(cfg, task, {name: {name: None} for name in task.spec.objects})


def _apply(cfg, task, assets):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from sim.envs.mdp.resets import park_object
    from sim.scene.contact_materials import (GRASP_FRICTION, GRASP_ROLLING_FRICTION,
                                             contact_material, contact_properties)
    links = _load_arm_spec(_os.environ.get("S2R_ARM"))[0]["links"]
    movable = []
    physical_assets = {asset: (task.spec.objects[name], size)
                       for name, variants in assets.items() for asset, size in variants.items()}
    for asset, position in _park_positions(physical_assets).items():
        obj, size = physical_assets[asset]
        path = "{ENV_REGEX_NS}/" + asset
        properties = dict(rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(kinematic_enabled=not obj.rigid),
                          collision_props=contact_properties(),
                          physics_material_path="physicsMaterial",
                          physics_material=contact_material(GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION),
                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obj.params.get("color", (.5, .5, .5)))))
        if obj.params.get("mass") is not None:
            properties["mass_props"] = sim_utils.MassPropertiesCfg(mass=obj.params["mass"])
        if size is not None and obj.usd is not None:
            native_size = obj.size or obj.params.get("size_outer")
            if native_size is None:
                from pxr import Usd, UsdGeom
                stage = Usd.Stage.Open(str(obj.usd))
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                native_size = bounds.ComputeLocalBound(stage.GetDefaultPrim()).ComputeAlignedRange().GetSize()
            scale = tuple(new / old for new, old in zip(size, native_size))
            spawn = sim_utils.UsdFileCfg(usd_path=str(obj.usd), scale=scale, **properties)
        elif obj.size is not None:
            spawn = sim_utils.CuboidCfg(size=size or obj.size, **properties)
        else:
            spawn = sim_utils.UsdFileCfg(usd_path=str(obj.usd), **properties)
        setattr(cfg.scene, asset, RigidObjectCfg(prim_path=path, spawn=spawn,
                                               init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(0., 0., 0., 1.))))
        setattr(cfg.events, f"park_{asset}", EventTermCfg(func=park_object, mode="reset",
                                                        params=dict(asset_cfg=SceneEntityCfg(asset), park_offset=position)))
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
    from sim.envs.mdp.objects import generation_assets

    assets = generation_assets(task.spec)
    _apply(cfg, task, assets)
    if task.spec.raw.get("checks", {}).get("holding", {}).get("contact_signal") == "target_held":
        links = _load_arm_spec(_os.environ.get("S2R_ARM"))[0]["links"]
        target = task.spec.roles["target"]
        for name, variants in assets.items():
            if name != target and task.spec.objects[name].params.get("set") != target:
                continue
            for asset in variants:
                for side, link in (("fixed", links["ee"]), ("moving", links["moving_jaw"])):
                    sensor = cfg.scene.contact_grasp.replace(
                        prim_path="{ENV_REGEX_NS}/Robot/" + link,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + asset],
                    )
                    setattr(cfg.scene, f"contact_{side}_{asset}", sensor)
    cfg.terminations = None
    cfg.episode_length_s = (task.spec.raw.get("expert", {}).get("max_steps", 600)+1)*cfg.decimation*cfg.sim.dt
