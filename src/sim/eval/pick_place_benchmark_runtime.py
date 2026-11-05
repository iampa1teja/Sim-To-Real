# sim/eval/pick_place_benchmark_runtime.py  (ported from v1 utils/pick_place_benchmark_runtime.py)
"""Read-only benchmark instrumentation; success and geometry remain task-owned."""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Live quaternions are XYZW; they are converted to WXYZ for the on-disk scene JSON and the numpy helpers.
# - Base and end-effector bodies come from arm.toml (BASE_LINK / EE_LINK); there is no ee_frame sensor.
# - Box size comes from task.toml white_box.size_outer; grasp terms come from tasks/pick_place_cube/mdp.py.
import numpy as np

from core.tasks.loader import load_task_module

TASK = "pick_place_cube"


def _numpy(x):
    from sim.envs.mdp.resets import as_torch
    return as_torch(x).detach().cpu().numpy()


def capture_scene(env):
    from pxr import Gf, Usd, UsdGeom
    from isaaclab.sim import get_current_stage

    from core.tasks.loader import load_task
    from sim.envs.base_env_cfg import BASE_LINK
    from sim.envs.mdp.resets import _wxyz
    from sim.scene.gripper_geometry import table_plane

    eval_helpers = load_task_module("eval", TASK)
    benchmark = load_task_module("benchmark", TASK)
    box_size = list(load_task(TASK).spec.objects["white_box"].params["size_outer"])
    stage = get_current_stage()
    env.sim.forward()
    robot, box = env.scene["robot"], env.scene["white_box"]
    ids, _ = robot.find_bodies(BASE_LINK)
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one robot body named {BASE_LINK!r}")
    base_p = _numpy(robot.data.body_link_pos_w)[0, ids[0]]
    base_q = _wxyz(_numpy(robot.data.body_link_quat_w)[0, ids[0]])
    box_p = _numpy(box.data.root_pos_w)[0]
    box_q = _wxyz(_numpy(box.data.root_quat_w)[0])
    root = stage.GetPrimAtPath(env.scene.env_prim_paths[0])
    table = next(p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if p.GetName() == "Tabletop" and p.IsA(UsdGeom.Mesh))
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    # Both projected polygons are expressed in the same live base frame.
    rotation = eval_helpers.rotation_matrix(base_q)
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0., 1.)])
    box_world = (corners * np.asarray(box_size)) @ eval_helpers.rotation_matrix(box_q).T + box_p
    return dict(source="live scene after reset", base_position_world=base_p.tolist(), base_quaternion_wxyz=base_q.tolist(),
                box_position_world=box_p.tolist(), box_quaternion_wxyz=box_q.tolist(), box_size_m=box_size,
                cube_size_m=list(env.scene["blue_cube"].cfg.spawn.size),
                table_plane_world=table_plane(stage, root).tolist(),
                table_outline_base_xy=benchmark.hull(((points - base_p) @ rotation)[:, :2]).tolist(),
                box_footprint_base_xy=benchmark.hull(((box_world - base_p) @ rotation)[:, :2]).tolist(),
                joint_mapping=getattr(env.cfg, "sim_joint_mapping", None))


def install_stage_tracking(env):
    """Wrap the success term to observe each terminal frame BEFORE auto-reset.

    The original term executes once and its exact return value is preserved.
    Observation functions share its per-step cached trackers.
    """
    mdp = load_task_module("mdp", TASK)
    eval_helpers = load_task_module("eval", TASK)
    benchmark = load_task_module("benchmark", TASK)
    from sim.envs.mdp.resets import _wxyz, as_torch

    cfg = env.termination_manager.get_term_cfg("success")
    original = cfg.func
    env._benchmark_trace = benchmark.StageTrace()
    grasp_offset = {}

    def grasp_point(raw, object_name):
        # World position where the cube centre sits when held: the fixed-jaw tip offset measured from the
        # gripper USD (same helper as the jaw spawn, zero clearance), applied to the live ee link pose.
        import isaaclab.utils.math as math_utils
        from isaaclab.sim import get_current_stage
        from sim.envs.mdp.resets import _ee_body
        from sim.scene.gripper_geometry import gripper_tip_local_offset

        robot = raw.scene["robot"]
        ee_id, ee_path = _ee_body(raw, robot)
        pos = as_torch(robot.data.body_link_pos_w)[0, ee_id]
        quat = as_torch(robot.data.body_link_quat_w)[0, ee_id]
        if object_name not in grasp_offset:
            half = float(raw.scene[object_name].cfg.spawn.size[0]) / 2
            grasp_offset[object_name] = pos.new_tensor(gripper_tip_local_offset(get_current_stage(), half, 0.0, ee_path))
        return pos + math_utils.quat_apply(quat.unsqueeze(0), grasp_offset[object_name].unsqueeze(0))[0]

    def observe(raw, **params):
        result = original(raw, **params)
        obs_params = {k: v for k, v in params.items() if k != "confirm_steps"}
        grasp_params = {k: v for k, v in obs_params.items() if not k.startswith("container_")}
        grasped = bool(mdp.object_grasped(raw, **grasp_params)[0].item())
        placed = bool(mdp.object_placed_in_container(raw, **obs_params)[0].item())
        cube = as_torch(raw.scene[params["object_name"]].data.root_pos_w)[0]
        ee = grasp_point(raw, params["object_name"])
        box = raw.scene[params["container_name"]]
        polygon = benchmark.hull(eval_helpers.footprint(_numpy(box.data.root_pos_w)[0],
                                                        eval_helpers.rotation_matrix(_wxyz(_numpy(box.data.root_quat_w)[0])),
                                                        params["container_size"], bottom_origin=True))
        raw._benchmark_trace.observe(distance=float((ee - cube).norm().item()), grasped=grasped,
                                     lift=float((cube[2] - raw._pick_place_rest_z[0]).item()), min_lift=params["min_lift"],
                                     held=bool(raw._pick_place_holding[0].item()),
                                     inside_box=benchmark.inside_polygon(cube[:2].detach().cpu().numpy(), polygon),
                                     placed=placed, success=bool(result[0].item()))
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg("success", cfg)
