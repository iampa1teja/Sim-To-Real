"""Pick up the cube and place it in the box: stages, success, expert and the task-specific rules."""
from sim.actions import approach, grasped, hold, lifted, pick, place, placed, reached

lift = lifted("blue_cube", min_lift=0.01)
grasp = grasped("blue_cube", force=2.0, lift=lift, warmup_steps=30)
placed_in_box = placed("blue_cube", "white_box", "inside", grasp=grasp, warmup_steps=30)

stages = {
    "reached": reached("gripper", "blue_cube", 0.03),
    "grasped": grasp,
    "lifted": lift,
    "over_box": reached("blue_cube", "white_box", relation="over", held=grasp),
    "placed": placed_in_box,
    "success": hold(placed_in_box, 25),
}

expert = [
    approach("blue_cube", height=.06),
    pick("blue_cube", lift=.09),
    place("blue_cube", "white_box", "inside", transport="arc", lift=.09, retreat=.06),
]


def failure_mode(flags, state):
    if flags.get("placed"):
        return "placed_not_confirmed"
    if not flags.get("reached"):
        return "never_reached"
    if not flags.get("grasped"):
        return "missed_grasp"
    if state is not None and state.signals.get("holding", False):
        return "timeout_holding"
    if not flags.get("over_box"):
        return "dropped_before_box"
    return "dropped_outside"


def progress_label(label, flags, state):
    if flags.get("grasped") and not flags.get("placed"):
        holding = bool(state is not None and state.signals.get("holding", False))
        label += ", held=" + ("yes" if holding else "no")
    return label


def start_is_valid(task, start, scene):
    """The resting cube footprint must not overlap the box footprint."""
    from utils.geometry import footprint, footprints_overlap, resting_pose, rotation_matrix_wxyz, upright_orientation_wxyz

    pose = start["blue_cube"]
    x, y, z, w = pose.quat_xyzw
    q = upright_orientation_wxyz(scene["base_quaternion_wxyz"], scene["table_plane_world"], (w, x, y, z))
    size = scene.get("objects", {}).get("blue_cube", {}).get("size", scene.get("cube_size_m"))
    world, rotation = resting_pose(pose.position, q, scene["base_position_world"],
                                   scene["base_quaternion_wxyz"], scene["table_plane_world"], size)
    box = scene.get("objects", {}).get("white_box", {})
    position = box.get("position_world", box.get("position_world_m", scene.get("box_position_world")))
    quaternion = box.get("orientation_world_wxyz", scene.get("box_quaternion_wxyz"))
    box_size = box.get("size", scene.get("box_size_m", task.spec.objects["white_box"].params["size_outer"]))
    polygon = footprint(position, rotation_matrix_wxyz(quaternion), box_size, bottom_origin=True)
    return not footprints_overlap(footprint(world, rotation, size), polygon)


def episode_metadata(metadata):
    return {**metadata, "cube_color": metadata["color"]}
