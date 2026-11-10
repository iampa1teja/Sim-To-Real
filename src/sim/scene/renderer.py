"""RTX configuration and a lazily loaded runtime for RGB/depth camera scenes."""

from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg


def build_renderer_cfg(settings=None):
    """Construct RTX configuration from an explicit scene renderer declaration."""
    return IsaacRtxRendererCfg(
        class_type="sim.scene.renderer:CalibratedRtxRenderer",
        global_settings=IsaacRtxRendererGlobalSettingsCfg(**dict(settings or {})),
    )


def __getattr__(name):
    if name != "CalibratedRtxRenderer":
        raise AttributeError(name)
    from isaaclab.app.settings_manager import get_settings_manager
    from isaaclab_physx.renderers import IsaacRtxRenderer

    class CalibratedRtxRenderer(IsaacRtxRenderer):
        """Keep RGB active when a later render product requests only depth."""

        def create_render_data(self, spec):
            data = super().create_render_data(spec)
            # Isaac RTX's per-camera setup writes a process-wide setting. A
            # depth-only product must not disable color needed by another camera.
            if any(kind in spec.cfg.data_types for kind in ("rgb", "rgba", "rgb_hdr")):
                self._has_color_render_product = True
            if getattr(self, "_has_color_render_product", False):
                get_settings_manager().set("/rtx/sdg/force/disableColorRender", False)
            return data


    CalibratedRtxRenderer.__qualname__ = "CalibratedRtxRenderer"
    globals()[name] = CalibratedRtxRenderer
    return CalibratedRtxRenderer
