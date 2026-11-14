import numpy as np 

from core.tasks.base import Pose, StepState, Task, load_task_spec

REACH_DISTANCE_M = 0.03 

class PickPlaceTask(Task):

    failure_modes = ("never_reached", "missed_grasp", "dropped_before_box", "dropped_outside",
                     "placed_not_confirmed", "timeout_holding")

    def signals(self, env):
        from .mdp import stage_signals
        return stage_signals(env)

    def episode_metadata(self, env, index=0):
        colors = getattr(env, "_pick_place_cube_color", None)
        color = colors[index] if colors is not None else "blue"
        return {"color": color, "cube_color": color}

    def observe(self, state: StepState):
        cube = np.asarray(state.objects["blue_cube"].position)
        box = state.objects["white_box"]
        signals = state.signals
        holding = bool(signals.get("holding", False))
        return {
            "reached": float(signals.get("reach_distance", np.linalg.norm(np.asarray(state.ee.position) - cube))) <= REACH_DISTANCE_M,
            "grasped": bool(signals.get("grasped", False)),
            "lifted": bool(signals.get("lifted", False)),
            "over_box": holding and self._over_box(cube, box),
            "placed": bool(signals.get("placed", False)),
            "success": bool(signals.get("success", False)),
        }

    def failure_mode(self, flags, state):
        if self.is_success(flags):
            return None
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

    def _over_box(self, cube, box):
        from core.eval.geometry import footprint, hull, inside_polygon, rotation_matrix

        size = tuple(self.spec.objects["white_box"].params["size_outer"])
        qx, qy, qz, qw = box.quat_xyzw
        polygon = footprint(box.position, rotation_matrix((qw, qx, qy, qz)), size, bottom_origin=True)
        return inside_polygon(cube[:2], hull(polygon))

    def progress_label(self, flags, state):
        label = super().progress_label(flags, state)
        if flags.get("grasped") and not flags.get("placed"):
            holding = bool(state is not None and state.signals.get("holding", False))
            label += ", held=" + ("yes" if holding else "no")
        return label

    def start_is_valid(self, start, scene):
        from core.eval.geometry import footprint, footprints_overlap, resting_pose, rotation_matrix, upright_orientation

        if not super().start_is_valid(start, scene):
            return False
        pose = start["blue_cube"]
        x, y, z, w = pose.quat_xyzw
        q = upright_orientation(scene["base_quaternion_wxyz"], scene["table_plane_world"], (w, x, y, z))
        size = scene.get("objects", {}).get("blue_cube", {}).get("size", scene.get("cube_size_m"))
        world, rotation = resting_pose(pose.position, q, scene["base_position_world"],
                                       scene["base_quaternion_wxyz"], scene["table_plane_world"], size)
        box = scene.get("objects", {}).get("white_box", {})
        position = box.get("position_world", box.get("position_world_m", scene.get("box_position_world")))
        quaternion = box.get("orientation_world_wxyz", scene.get("box_quaternion_wxyz"))
        box_size = box.get("size", scene.get("box_size_m", self.spec.objects["white_box"].params["size_outer"]))
        polygon = footprint(position, rotation_matrix(quaternion), box_size, bottom_origin=True)
        return not footprints_overlap(footprint(world, rotation, size), polygon)

TASK = PickPlaceTask(load_task_spec("pick_place_cube"))
