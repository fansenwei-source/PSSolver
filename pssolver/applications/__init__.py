"""Supported high-level PSSolver applications."""

from .plane_beris_edwards import main as plane_beris_edwards_main
from .plane_beris_edwards import run_plane_beris_edwards

__all__ = [
    "plane_beris_edwards_main",
    "run_plane_beris_edwards",
]
