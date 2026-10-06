"""N2 contract for the explicit rc2 public-declaration migration."""

from __future__ import annotations

import inspect
from pathlib import Path

from pssolver import Output, SpectralNumerics


ROOT = Path(__file__).resolve().parents[1]


def test_ambiguous_numerical_and_output_policies_remain_required():
    numerics = inspect.signature(SpectralNumerics).parameters
    output = inspect.signature(Output).parameters

    assert numerics["spectral_storage"].default is inspect.Parameter.empty
    for name in ("save_start_step", "diagnostics", "save_hydrodynamics"):
        assert output[name].default is inspect.Parameter.empty


def test_changelog_marks_the_break_and_gives_the_complete_migration():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "**breaking source-level API change**" in changelog
    assert "spectral_storage=\"full_complex\"" in changelog
    assert "save_start_step=0" in changelog
    assert "diagnostics=True" in changelog
    assert "save_hydrodynamics=True" in changelog
    assert "PSSolver-Control" in changelog


def test_architecture_example_and_consumer_handoff_are_not_stale():
    architecture = (
        ROOT
        / "notes/architecture_v0_2/phase_7_p778_typed_public_declarations.md"
    ).read_text(encoding="utf-8")
    handoff = (
        ROOT / "notes/PSSolver_v0_2_0rc3_n2_control_migration_zh.md"
    ).read_text(encoding="utf-8")

    for token in (
        'spectral_storage="full_complex"',
        "save_start_step=0",
        "diagnostics=True",
        "save_hydrodynamics=True",
    ):
        assert token in architecture
    for path in (
        "tests/periodic_runtime.py",
        "tests/channel_runtime.py",
        "experiments/warmup_3d/periodic_box.py",
        "experiments/warmup_3d/hidden_mode_reproducer.py",
        "experiments/warmup_3d/r1_bridge.py",
    ):
        assert path in handoff
