"""Explicit SO101 replay geometry candidates; import alone changes no configuration."""

from core.arms.spec import load_arm_spec, load_arm_utils

_utilities = load_arm_utils(load_arm_spec("so101")[0])
_inscribed_pad_face = _utilities._inscribed_pad_face
_mesh_triangles = _utilities._mesh_triangles
apply_primitive_pads = _utilities.apply_primitive_pads


def apply_moving_jaw_split(root):
    """Use the arm-owned CAD splitter; a disabled whole-mesh hull is a no-op."""
    return _utilities._split_moving_finger(root)
