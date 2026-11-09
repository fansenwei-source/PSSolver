"""P7.7.6 production-entry connection to package-owned construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pssolver.applications import plane_beris_edwards as plane_application
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
)
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = ROOT / ("notes/architecture_v0_2/phase_7_p776_production_connection.json")


PLANE_APPLICATION = ROOT / "pssolver/applications/plane_beris_edwards.py"


CHANNEL_APPLICATION = ROOT / ("pssolver/applications/channel_active_nematics.py")


RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"


RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"


RC6_R6 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rc456_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (RC456, RC5_R57, RC6_R6):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if registration is not None and registration["historical_sha256"] == resolved:
            resolved = registration["current_sha256"]
    return resolve_s03_successor(relative, resolved)


def _plane_spec(output: Path, runtime_path: str):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=output,
        runtime_path=runtime_path,
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=6,
        ny=6,
        nz=6,
        lx=8.0,
        ly=9.0,
        height=20.0,
        steps=2,
        save_start_step=0,
        save_interval=2,
        diagnostic_interval=1,
        disable_spectral_refresh=True,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
        save_hydrodynamics=True,
    )


@pytest.mark.parametrize(
    ("runtime_path", "kind"),
    (
        ("legacy_production", "plane_legacy_production"),
        ("compiled_v2", "plane_compiled_v2"),
    ),
)
def test_plane_production_entry_passes_typed_package_construction(
    tmp_path,
    monkeypatch,
    runtime_path,
    kind,
):
    captured = []
    original = plane_application.build_package_simulation_runtime

    def spy(construction):
        captured.append(construction)
        return original(construction)

    monkeypatch.setattr(
        plane_application,
        "build_package_simulation_runtime",
        spy,
    )
    output = tmp_path / runtime_path
    result = plane_application.run_plane_beris_edwards(_plane_spec(output, runtime_path))

    assert result.final_step == 2
    assert len(captured) == 1
    assert isinstance(captured[0], PackageRuntimeConstructionInput)
    assert captured[0].plan.kind.value == kind
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["runtime_construction"] == {
        "plan": captured[0].plan.to_metadata(),
        "input": captured[0].to_metadata(),
    }
    assert metadata["runtime_selection"]["effective"] == runtime_path
    assert metadata["runtime_selection"]["fallback_used"] is False
