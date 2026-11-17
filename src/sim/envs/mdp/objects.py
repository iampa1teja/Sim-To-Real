"""Task-object operations in the selected robot's base-link frame."""

import os as _os

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.tasks.base import Pose as _Pose
from core.tasks.sampling import park_positions as _park_positions


def generation_assets(task_spec):
    """Group the physical scene assets by logical task object and variant size."""
    assets = {}
    for name, obj in task_spec.objects.items():
        variants = obj.params.get("size_variants", ())
        assets[name] = ({f"{name}__v{index}": tuple(size) for index, size in enumerate(variants)}
                        if variants else {name: None})
    return assets


def select_generation_assets(task_spec, sizes):
    active = {}
    for name, variants in generation_assets(task_spec).items():
        size = sizes.get(name)
        if name in variants:
            active[name] = name
            continue
        desired = task_spec.objects[name].size if size is None else tuple(size)
        selected = next((asset for asset, dimensions in variants.items() if dimensions == desired), None)
        if selected is None and size is not None:
            raise ValueError(f"Unknown size variant {size} for {name}")
        active[name] = selected or next(iter(variants))
    return active


def _base(env):
    from sim.envs.mdp.resets import as_torch
    robot = env.scene["robot"]
    links = _load_arm_spec(getattr(env, "_generation_arm", _os.environ.get("S2R_ARM")))[0]["links"]
    ids, _ = robot.find_bodies(links["base"], preserve_order=True)
    if len(ids) != 1:
        raise ValueError("Expected one robot base body")
    return (as_torch(robot.data.body_link_pos_w)[:, ids[0]],
            as_torch(robot.data.body_link_quat_w)[:, ids[0]])


def set_object_pose_base(env, name, pose):
    import torch
    import isaaclab.utils.math as math_utils
    position, rotation = _base(env)
    p = position.new_tensor(pose.position).expand(env.num_envs, 3)
    q = rotation.new_tensor(pose.quat_xyzw).expand(env.num_envs, 4)
    p, q = math_utils.combine_frame_transforms(position, rotation, p, q)
    ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[name]
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((p, q), dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=p.new_zeros((env.num_envs, 6)), env_ids=ids)


def set_object_color(env, name, rgb):
    from pxr import Gf, Usd, UsdShade
    from isaaclab.sim import get_current_stage
    from sim.envs.mdp.resets import _env_prim_path
    stage, asset = get_current_stage(), env.scene[name]
    for index in range(env.num_envs):
        root = stage.GetPrimAtPath(_env_prim_path(env, asset, index))
        shaders = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                shader, _, _ = material.ComputeSurfaceSource()
                if shader:
                    shaders[str(shader.GetPath())] = shader
        if not shaders:
            raise RuntimeError(f"No bound surface shader for {name}")
        for shader in shaders.values():
            diffuse = shader.GetInput("diffuseColor")
            if not diffuse:
                raise RuntimeError(f"Missing diffuseColor input for {name}")
            diffuse.Set(Gf.Vec3f(*map(float, rgb)))


def set_object_physics(env, name, values):
    """Apply episode physics to one asset, restoring omitted properties from its defaults."""
    import torch
    import warp as wp
    from isaaclab.sim import get_current_stage
    from pxr import Usd, UsdPhysics, UsdShade
    from sim.envs.mdp.resets import _env_prim_path, as_torch

    if not hasattr(env, "_generation_physics_defaults"):
        env._generation_physics_defaults = {}
    defaults = env._generation_physics_defaults
    if not values and name not in defaults:
        return
    asset = env.scene[name]
    view = asset.root_view
    manager = env.sim.physics_manager
    newton = hasattr(view, "get_attribute")
    if newton:
        model = manager.get_model()
        friction = wp.to_torch(view.get_attribute("shape_material_mu", model)[:, 0])
    else:
        friction = wp.to_torch(view.get_material_properties())
    if name not in defaults:
        stage = get_current_stage()
        materials = {}
        for index in range(env.num_envs):
            root = stage.GetPrimAtPath(_env_prim_path(env, asset, index))
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    if bound:
                        material = UsdPhysics.MaterialAPI(bound.GetPrim())
                        materials[bound.GetPath()] = (material.GetStaticFrictionAttr(), material.GetDynamicFrictionAttr())
        material_defaults = tuple(tuple((attr, attr.Get()) for attr in attrs) for attrs in materials.values())
        defaults[name] = (as_torch(asset.data.body_mass).clone(), friction.clone(), material_defaults)
    mass, nominal, attributes = defaults[name]
    asset.set_masses_index(masses=torch.full_like(mass, values["mass_kg"]) if "mass_kg" in values else mass)
    keys = ("static_friction", "dynamic_friction")
    for material in attributes:
        for key, (attr, default) in zip(keys, material):
            if attr and (key in values or default is not None):
                attr.Set(values.get(key, default))
    friction.copy_(nominal)
    if newton:
        # Newton represents sliding friction with a single coefficient.
        from newton import ModelFlags
        if "dynamic_friction" in values:
            friction.fill_(values["dynamic_friction"])
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    else:
        for index, key in enumerate(keys):
            if key in values:
                friction[..., index].fill_(values[key])
        ids = torch.arange(env.num_envs, dtype=torch.int32, device="cpu")
        view.set_material_properties(wp.from_torch(friction, dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))


def object_poses_in_base(env, names):
    import isaaclab.utils.math as math_utils
    from sim.envs.mdp.resets import as_torch
    base_position, base_rotation = _base(env)
    result = {}
    for name in names:
        asset = env.scene[name]
        p, q = math_utils.subtract_frame_transforms(base_position, base_rotation,
                                                   as_torch(asset.data.root_pos_w), as_torch(asset.data.root_quat_w))
        result[name] = _Pose(tuple(float(v) for v in p[0].detach().cpu().tolist()),
                             tuple(float(v) for v in q[0].detach().cpu().tolist()))
    return result


def park_all(env, names):
    for name, position in _park_positions(names).items():
        set_object_pose_base(env, name, _Pose(position))
