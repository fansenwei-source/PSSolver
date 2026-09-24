"""P7.7.6 production-entry connection to package-owned construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pssolver.applications import channel_active_nematics as channel_application
from pssolver.applications import plane_beris_edwards as plane_application
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p776_production_connection.json"
)
PLANE_APPLICATION = ROOT / "pssolver/applications/plane_beris_edwards.py"
CHANNEL_APPLICATION = ROOT / (
    "pssolver/applications/channel_active_nematics.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _channel_spec(output: Path, runtime_path: str):
    return ChannelActiveNematicRunSpec(
        shape=(8, 6, 5),
        lengths=(8.0, 6.0, 5.0),
        dt=1.0e-3,
        steps=2,
        save_interval=2,
        diagnostic_interval=1,
        generated_output_directory=output,
        device="cpu",
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        pressure_fixed_iterations=8,
        runtime_path=runtime_path,
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
    result = plane_application.run_plane_beris_edwards(
        _plane_spec(output, runtime_path)
    )

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


@pytest.mark.parametrize(
    ("runtime_path", "kind"),
    (
        ("legacy_channel", "channel_legacy"),
        ("compiled_channel_v2", "channel_compiled_v2"),
    ),
)
def test_channel_production_entry_passes_typed_package_construction(
    tmp_path,
    monkeypatch,
    runtime_path,
    kind,
):
    captured = []
    original = channel_application.build_package_simulation_runtime

    def spy(construction):
        captured.append(construction)
        return original(construction)

    monkeypatch.setattr(
        channel_application,
        "build_package_simulation_runtime",
        spy,
    )
    output = tmp_path / runtime_path
    result = channel_application.run_channel_active_nematics(
        _channel_spec(output, runtime_path)
    )

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


def test_application_local_builder_closures_are_removed_and_defaults_frozen():
    plane = PLANE_APPLICATION.read_text(encoding="utf-8")
    channel = CHANNEL_APPLICATION.read_text(encoding="utf-8")
    for source in (plane, channel):
        assert "PackageRuntimeConstructionInput" in source
        assert "build_package_simulation_runtime" in source
        assert "runtime_construction" in source
        assert "def legacy_builder" not in source
        assert "def compiled_builder" not in source
    assert plane.count("build_plane_beris_edwards_runtime(runtime_request)") == 1
    assert "compatibility_exception\": \"separated_canary" in plane
    assert "build_plane_compiled_v2_runtime" not in plane
    assert "build_active_nematic_channel" not in channel
    assert "build_channel_active_nematic_runtime" not in channel
    assert "build_package_compiled_channel_runtime" not in channel
    assert create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p776_plane_default",
    ).runtime_path.value == "legacy_production"
    assert ChannelActiveNematicRunSpec().runtime_path.value == "legacy_channel"


def test_p776_machine_record_matches_candidate_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == (
        "PASS_P7_7_6_LOCAL_PRODUCTION_CONNECTION_H100_PENDING"
    )
    assert record["production_applications_connected"] is True
    assert record["application_local_builder_closures_removed"] is True
    assert record["production_default_changed"] is False
    assert record["runtime_selection_changed"] is False
    assert record["qualification_complete"] is False
    assert record["phase_8_authorized"] is False
    assert record["source_sha256"] == {
        "pssolver/applications/channel_active_nematics.py": _sha256(
            CHANNEL_APPLICATION
        ),
        "pssolver/applications/plane_beris_edwards.py": _sha256(
            PLANE_APPLICATION
        ),
    }
