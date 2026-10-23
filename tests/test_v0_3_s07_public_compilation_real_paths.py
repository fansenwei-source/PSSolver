"""Real public dry-runs for S-07 compilation provenance."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from pssolver import compile_simulation, run_simulation
from pssolver.configuration.simulation import InvocationSpec
from tests.test_phase7_p7710_public_runner_result import (
    _channel_simulation as legacy_channel_simulation,
)
from tests.test_phase8_p82_periodic_application import (
    _simulation as periodic_simulation,
)
from tests.test_phase8_p83_channel_complete_stress_application import (
    _simulation as complete_stress_channel_simulation,
)


PUBLIC_COMPILATION_KEYS = {
    "schema_version",
    "compilation_identity_sha256",
    "source_simulation_sha256",
    "application",
    "application_simulation_sha256",
    "application_request_sha256",
    "normalization",
    "fallback_allowed",
}


def _assert_real_dry_run(source, capsys) -> None:
    compiled = compile_simulation(source)
    result = run_simulation(compiled, emit_metadata=True)
    assert result.status.value == "dry_run"
    assert result.output_directory is None
    metadata = json.loads(capsys.readouterr().out)
    public = metadata["public_compilation"]
    assert set(public) == PUBLIC_COMPILATION_KEYS
    assert public["compilation_identity_sha256"] == (
        compiled.compilation_identity_sha256
    )
    assert public["source_simulation_sha256"] == source.canonical_sha256()
    assert public["application"] == compiled.application
    assert public["application_simulation_sha256"] == (
        compiled.application_specification.canonical_sha256()
    )
    assert public["application_request_sha256"] == (
        compiled.application_request.canonical_sha256()
    )
    assert public["normalization"] == dict(compiled.normalization)
    assert public["fallback_allowed"] is False


@pytest.mark.parametrize("runtime_path", ("legacy_channel", "compiled_channel_v2"))
def test_legacy_channel_real_dry_run_persists_public_compilation(
    tmp_path,
    capsys,
    runtime_path,
):
    source = legacy_channel_simulation(tmp_path, runtime_path=runtime_path)
    source = replace(source, invocation=InvocationSpec({"dry_run": True}))
    _assert_real_dry_run(source, capsys)
    assert not (tmp_path / "generated").exists()


def test_periodic_real_dry_run_persists_public_compilation(tmp_path, capsys):
    source = periodic_simulation(
        tmp_path,
        invocation_options={"dry_run": True},
    )
    _assert_real_dry_run(source, capsys)
    assert not (tmp_path / "run").exists()


def test_complete_stress_channel_real_dry_run_persists_public_compilation(
    tmp_path,
    capsys,
):
    source = complete_stress_channel_simulation(
        tmp_path,
        invocation_options={"dry_run": True},
    )
    _assert_real_dry_run(source, capsys)
    assert not (tmp_path / "run").exists()
