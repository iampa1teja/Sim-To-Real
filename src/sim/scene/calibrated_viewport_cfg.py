"""Kit viewport configuration for an existing, calibrated USD camera."""

from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg


@configclass
class CalibratedKitVisualizerCfg(KitVisualizerCfg):
    """Select a scene camera without replacing its measured pose or lens settings."""

    class_type: str = "sim.scene.calibrated_viewport:CalibratedKitVisualizer"
    camera_prim_path: str = ""

