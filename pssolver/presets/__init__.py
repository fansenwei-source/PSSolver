"""Named physical parameter presets for reproducible model construction."""

from .shendruk import ShendrukPlanePreset, resolve_shendruk_plane_preset

__all__ = [
    "ShendrukPlanePreset",
    "resolve_shendruk_plane_preset",
]
