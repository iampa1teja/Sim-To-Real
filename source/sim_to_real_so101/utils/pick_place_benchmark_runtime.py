"""Read-only benchmark instrumentation; success and geometry remain task-owned."""
import numpy as np
from .pick_place_eval import rotation_matrix
from .pick_place_benchmark import hull, inside_polygon, StageTrace


def capture_scene(env):
    from pxr import Usd, UsdGeom, Gf
    from isaacsim.core.utils.stage import get_current_stage
    from .gripper_geometry import table_plane
    from sim_to_real_so101.tasks.pick_place_env_cfg import WHITE_BOX_SIZE
    stage = get_current_stage()
    env.sim.forward()
    robot, box = env.scene['robot'], env.scene['white_box']
    ids, _ = robot.find_bodies('base')
    base_p = robot.data.body_pos_w[0, ids[0]].detach().cpu().numpy()
    base_q = robot.data.body_quat_w[0, ids[0]].detach().cpu().numpy()
    box_p = box.data.root_pos_w[0].detach().cpu().numpy()
    box_q = box.data.root_quat_w[0].detach().cpu().numpy()
    root = stage.GetPrimAtPath(env.scene.env_prim_paths[0])
    table = next(p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == 'Tabletop' and p.IsA(UsdGeom.Mesh))
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    # Both projected polygons are expressed in the same live base frame.
    rotation = rotation_matrix(base_q)
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0., 1.)])
    box_world = (corners*np.asarray(WHITE_BOX_SIZE)) @ rotation_matrix(box_q).T + box_p
    return dict(source='live scene after reset', base_position_world=base_p.tolist(), base_quaternion_wxyz=base_q.tolist(),
                box_position_world=box_p.tolist(), box_quaternion_wxyz=box_q.tolist(), box_size_m=list(WHITE_BOX_SIZE),
                cube_size_m=list(env.scene['blue_cube'].cfg.spawn.size),
                table_plane_world=table_plane(stage, root).tolist(),
                table_outline_base_xy=hull(((points-base_p)@rotation)[:, :2]).tolist(),
                box_footprint_base_xy=hull(((box_world-base_p)@rotation)[:, :2]).tolist(),
                joint_mapping=getattr(env.cfg, 'sim_joint_mapping', None))


def install_stage_tracking(env):
    """Wrap the success term to observe each terminal frame BEFORE auto-reset.

    The original term executes once and its exact return value is preserved.
    Observation functions share its per-step cached trackers.
    """
    from sim_to_real_so101.mdp.terms import object_grasped, object_placed_in_container
    from .pick_place_eval import footprint
    cfg = env.termination_manager.get_term_cfg('success')
    original = cfg.func
    env._benchmark_trace = StageTrace()

    def observe(raw, **params):
        result = original(raw, **params)
        obs_params = {k: v for k, v in params.items() if k != 'confirm_steps'}
        grasp_params = {k: v for k, v in obs_params.items() if not k.startswith('container_')}
        grasped = bool(object_grasped(raw, **grasp_params)[0].item())
        placed = bool(object_placed_in_container(raw, **obs_params)[0].item())
        cube = raw.scene[params['object_name']].data.root_pos_w[0]
        ee = raw.scene['ee_frame'].data.target_pos_w[0, 0]
        box = raw.scene[params['container_name']]
        polygon = hull(footprint(box.data.root_pos_w[0].detach().cpu().numpy(),
                                 rotation_matrix(box.data.root_quat_w[0].detach().cpu().numpy()),
                                 params['container_size'], bottom_origin=True))
        raw._benchmark_trace.observe(distance=float((ee-cube).norm().item()), grasped=grasped,
                                     lift=float((cube[2]-raw._pick_place_rest_z[0]).item()), min_lift=params['min_lift'],
                                     held=bool(raw._pick_place_holding[0].item()),
                                     inside_box=inside_polygon(cube[:2].detach().cpu().numpy(), polygon),
                                     placed=placed, success=bool(result[0].item()))
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg('success', cfg)
