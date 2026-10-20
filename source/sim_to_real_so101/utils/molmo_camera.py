"""An additional behind-arm camera, enabled only by Molmo evaluation options."""
from copy import deepcopy


# In the robot base frame, the pick-place workspace lies along negative Y.
# Keep the added camera behind positive Y and high enough to look over the arm.
EYE_IN_BASE = (-0.25, 0.35, 0.85)
TARGET_IN_BASE = (-0.025, -0.14, 0.04)
SENSOR_NAME = "camera_molmo_room"
# Endpoints of the packaged tabletop's front edge in the robot base frame,
# ordered toward camera-right. They set camera roll without moving its aim.
TABLE_EDGE_IN_BASE = ((1.824393, 0.048087, 0.030076),
                      (-0.614616, 0.048028, 0.030076))


def _camera_pose(base_position, base_quaternion):
    from pxr import Gf

    position = Gf.Vec3d(*(float(value) for value in base_position))
    rotation = Gf.Rotation(Gf.Quatd(float(base_quaternion[0]),
                                  Gf.Vec3d(*(float(value) for value in base_quaternion[1:]))))
    eye = position + rotation.TransformDir(Gf.Vec3d(*EYE_IN_BASE))
    target = position + rotation.TransformDir(Gf.Vec3d(*TARGET_IN_BASE))
    forward = (target - eye).GetNormalized()
    # Project the actual edge rays onto the optical plane. Projecting only its
    # 3D direction leaves a slope when perspective changes along the edge.
    rays = [position + rotation.TransformDir(Gf.Vec3d(*point)) - eye
            for point in TABLE_EDGE_IN_BASE]
    right = (rays[1] / Gf.Dot(rays[1], forward)
             - rays[0] / Gf.Dot(rays[0], forward)).GetNormalized()
    up = Gf.Cross(right, forward).GetNormalized()
    transform = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
    quaternion = transform.ExtractRotation().GetQuat()
    return tuple(eye), (quaternion.GetReal(), *quaternion.GetImaginary()), tuple(target)


def _spawn_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the extra sensor without changing the existing named-camera callback."""
    from isaaclab.sim.spawners.sensors import spawn_camera
    from sim_to_real_so101.assets.real_setup import apply_opencv_intrinsics

    prim = spawn_camera(prim_path, cfg.replace(intrinsics=None, response=None),
                        translation=translation, orientation=orientation, **kwargs)
    if cfg.intrinsics is not None:
        apply_opencv_intrinsics(prim, cfg.intrinsics)
    prim.SetDisplayName("Molmo behind-arm room")
    return prim


def add_molmo_overhead_camera(env_cfg):
    """Add an independent RGB sensor and observation to this environment instance.

    Neither the original room/wrist sensors nor their observation terms are
    edited. The caller selects the new view through its policy camera mapping.
    """
    scene, visual = env_cfg.scene, env_cfg.observations.visual
    if (getattr(scene, SENSOR_NAME, None) is not None
            or getattr(visual, "rgb_molmo_room", None) is not None):
        raise ValueError("Molmo overhead camera already exists in this environment")

    camera = deepcopy(scene.camera_realsense_rgb)
    position, quaternion, target = _camera_pose(scene.robot.init_state.pos,
                                               scene.robot.init_state.rot)
    camera.prim_path = "{ENV_REGEX_NS}/molmo_room"
    camera.offset.pos = position
    camera.offset.rot = quaternion
    camera.offset.convention = "opengl"
    camera.data_types = ["rgb"]
    camera.spawn.func = _spawn_camera

    observation = deepcopy(visual.rgb_realsense_rgb)
    observation.params["sensor_cfg"].name = SENSOR_NAME
    setattr(scene, SENSOR_NAME, camera)
    visual.rgb_molmo_room = observation
    return {
        "sensor": "molmo_room",
        "position": list(position),
        "target": list(target),
        "orientation_wxyz": list(quaternion),
        "convention": "opengl",
        "alignment": "table_front_edge",
        "width": camera.width,
        "height": camera.height,
    }
