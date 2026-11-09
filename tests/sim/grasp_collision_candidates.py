"""Explicit replay geometry candidates; import alone changes no configuration."""

from sim.scene.gripper_collisions import (
    _inscribed_pad_face, _mesh_triangles, apply_primitive_pads,
)


def apply_moving_jaw_split(root):
    """Use the production CAD splitter; a disabled whole-mesh hull is a no-op."""
    from sim.scene.contact_materials import _split_moving_finger

    return _split_moving_finger(root)
