"""Capture task-independent live geometry for fixed evaluation starts."""
import numpy as np

from core.arms.spec import load_arm_spec
from core.scenes import load_scene_spec
from core.tasks.loader import load_task


def capture_scene(env, task=None, index=0):
    from pxr import Gf, UsdGeom
    from core.benchmark.geometry import hull, rotation_matrix
    from core.scenes.anchors import anchor_quaternion
    from sim.envs.mdp.resets import _wxyz, as_torch
    from sim.scene.geometry import anchor_prim, table_plane

    task = task or load_task(getattr(env.cfg, "task_name", None))
    arm = getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]
    scene = getattr(env.cfg, "scene_spec", None) or load_scene_spec()
    env.sim.forward()
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(arm.links.base)
    if len(ids) != 1:
        raise ValueError(f"Expected one robot base body named {arm.links.base!r}")

    def array(value):
        return as_torch(value).detach().cpu().numpy()

    base_p = array(robot.data.body_link_pos_w)[index, ids[0]]
    base_q = _wxyz(array(robot.data.body_link_quat_w)[index, ids[0]])
    rotation = rotation_matrix(base_q)
    result = dict(source="live scene after reset", arm=arm.name, task=task.name,
                  base_position_world=base_p.tolist(), base_quaternion_wxyz=base_q.tolist(),
                  joint_mapping=getattr(env.cfg, "sim_joint_mapping", None), objects={})
    result["frames"] = {
        name: dict(position_world=list(anchor.position),
                   orientation_world_wxyz=_wxyz(anchor_quaternion(anchor)).tolist())
        for name, anchor in scene.anchors.items()
    }
    result["frames"].update(
        world=dict(position_world=[0., 0., 0.], orientation_world_wxyz=[1., 0., 0., 0.]),
        arm_base=dict(position_world=base_p.tolist(), orientation_world_wxyz=base_q.tolist()),
    )
    anchor = scene.anchors.get("table")
    if anchor is not None:
        stage = env.sim.stage
        root = stage.GetPrimAtPath(env.scene.env_prim_paths[index])
        result["table_plane_world"] = table_plane(stage, root, anchor).tolist()
        table = anchor_prim(stage, root, anchor)
        if table is not None and table.IsA(UsdGeom.Mesh):
            cache = getattr(env, "_task_table_points", None)
            if cache is None:
                cache = env._task_table_points = {}
            key = str(table.GetPath())
            if key not in cache:
                transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
                points = np.array([transform.Transform(Gf.Vec3d(*map(float, point)))
                                   for point in UsdGeom.Mesh(table).GetPointsAttr().Get()])
                cache[key] = points * UsdGeom.GetStageMetersPerUnit(stage)
            result["table_outline_base_xy"] = hull(((cache[key] - base_p) @ rotation)[:, :2]).tolist()
    for name, spec in task.spec.objects.items():
        asset = env.scene[name]
        world_p = array(asset.data.root_pos_w)[index]
        world_q = _wxyz(array(asset.data.root_quat_w)[index])
        # Pure frame math uses XYZW; the benchmark boundary is explicitly WXYZ.
        import isaaclab.utils.math as math_utils
        position, orientation = math_utils.subtract_frame_transforms(
            as_torch(robot.data.body_link_pos_w)[index:index + 1, ids[0]],
            as_torch(robot.data.body_link_quat_w)[index:index + 1, ids[0]],
            as_torch(asset.data.root_pos_w)[index:index + 1],
            as_torch(asset.data.root_quat_w)[index:index + 1],
        )
        size = spec.size or spec.params.get("size_outer") or getattr(asset.cfg.spawn, "size", None)
        row = dict(position_world=world_p.tolist(), orientation_world_wxyz=world_q.tolist(),
                   position=array(position)[0].tolist(), orientation_wxyz=_wxyz(array(orientation)[0]).tolist(),
                   bottom_origin=bool(spec.params.get("bottom_origin", False)))
        if size is not None:
            row["size"] = list(size)
            corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                                for z in ((0., 1.) if row["bottom_origin"] else (-.5, .5))])
            world = (corners * np.asarray(size)) @ rotation_matrix(world_q).T + world_p
            row["footprint_base_xy"] = hull(((world - base_p) @ rotation)[:, :2]).tolist()
        result["objects"][name] = row
    return result


def validate_scene_geometry(live, expected, task, *, atol=1e-5):
    """Keep the frozen base/table/static-object guard without constraining sampled poses."""
    from core.benchmark.eval_set import normalize_scene
    from core.benchmark.geometry import polygons_match

    expected = normalize_scene(expected, task)
    for key in ("base_position_world", "base_quaternion_wxyz", "table_plane_world"):
        if key in expected and (key not in live or np.shape(live[key]) != np.shape(expected[key])
                                or not np.allclose(live[key], expected[key], atol=atol, rtol=0)):
            raise ValueError(f"Live scene differs from eval set: {key}; recapture geometry")
    if "table_outline_base_xy" in expected and not polygons_match(
            live.get("table_outline_base_xy", []), expected["table_outline_base_xy"], atol=atol):
        raise ValueError("Live scene differs from eval set: table outline; recapture geometry")
    for name, saved in expected.get("objects", {}).items():
        if name not in live["objects"]:
            raise ValueError(f"Live scene is missing eval object {name!r}")
        current = live["objects"][name]
        if "size" in saved and (np.shape(current.get("size")) != np.shape(saved["size"])
                                 or not np.allclose(current["size"], saved["size"], atol=atol, rtol=0)):
            raise ValueError(f"Live scene differs from eval set: {name} size")
        if name not in task.spec.tracked_objects and "footprint_base_xy" in saved:
            if not polygons_match(current.get("footprint_base_xy", []), saved["footprint_base_xy"], atol=atol):
                raise ValueError(f"Live scene differs from eval set: {name} footprint; recapture geometry")
