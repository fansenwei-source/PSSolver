"""Generate authentic retirement fixtures with the installed v0.4.0 wheel only.

Run in the D-0045 independent baseline environment, outside any Provider source
tree. This is a historical fixture producer, not a 0.5 runtime compatibility path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pssolver
import torch

from pssolver.applications.plane_beris_edwards import run_plane_beris_edwards
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.plane_beris_edwards import create_plane_beris_edwards_run_spec
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import strong_homeotropic_q


def generate(output: Path) -> None:
    """Write one-step canary and static-lifting checkpoints, without editing them."""
    assert pssolver.__version__ == "0.4.0"
    assert Path(pssolver.__file__).is_relative_to(Path(sys.prefix))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    for family in ("separated_canary", "private_lifting"):
        spec = create_plane_beris_edwards_run_spec(
            activity_number=18.0, output_dir=output / family,
            device="cpu", dtype="float64", pointwise_execution="eager",
            runtime_path="separated_canary" if family == "separated_canary" else "legacy_production",
            nx=8, ny=8, nz=6, lx=100.0, ly=100.0, height=20.0,
            steps=1, save_start_step=0, save_interval=1, diagnostic_interval=1,
            defect_min_separation=10.0, defect_core_radius=1.5, twist_modes=(1, 2, 3),
            checkpoint_interval=1,
        )
        simulation = None
        if family == "private_lifting":
            base = compose_plane_beris_edwards_simulation(
                decompose_plane_beris_edwards_run_spec(spec)
            )
            q_policy = strong_homeotropic_q(
                scalar_order=0.6,
                face_normals={(2, "lower"): (0.0, 0.0, -1.0), (2, "upper"): (0.0, 0.0, 1.0)},
            )
            simulation = replace(base, boundaries=assign_boundaries(
                model=base.equation_system, geometry=base.geometry,
                policies={"Q": q_policy, "velocity": free_slip_velocity(),
                          "pressure": neumann_pressure_compatibility()},
                name="s24_private_lifting_fixture",
            ))
        run_plane_beris_edwards(spec, application_simulation=simulation)
    manifest = {
        "source_commit": "4c12228aba01e8483e80d28a1471ab72384d5fda",
        "source_wheel_sha256": "c5cf3e26af8bdde27287f6129fe5939d6f202fb627dcdc9a66c19d62871dd873",
        "pssolver_file": pssolver.__file__, "python": sys.executable,
        "device": "cpu", "files": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            manifest["files"][path.relative_to(output).as_posix()] = {
                "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            }
    (output / "generation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
