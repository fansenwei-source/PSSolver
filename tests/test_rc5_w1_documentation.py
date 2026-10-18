"""Contracts for the rc5 W1 public-documentation alignment."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

from pssolver import SimulationRunStatus, capability_catalog
from pssolver.operators.robin import CellCenteredRobinEigenbasisOperator


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc5_w1_documentation.json"
R5_1_RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
R5_2_RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
R5_3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
R5_5_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
R5_6_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_R6_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
)
RC6_FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readme_example() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- rc5-complete-simulation-example:start -->\n"
        r"```python\n(?P<source>.*?)\n```\n"
        r"<!-- rc5-complete-simulation-example:end -->",
        readme,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("source")


def test_w1_record_closes_only_the_documentation_batch():
    record = _read(RECORD)
    assert record["classification"] == "PASS_RC5_W1_DOCUMENTATION_ALIGNMENT"
    assert record["closed_items"] == ["B30", "NEW-4", "NEW-5"]
    assert record["next_batch"] == "rc5_must"
    assert record["scope"] == {
        "runtime_behavior_changed": False,
        "numerical_implementation_changed": False,
        "public_capability_widened": False,
        "lifting_reopened": False,
        "scientific_threshold_changed": False,
        "skip_or_xfail_added": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
        "h100_submitted": False,
        "tag_or_release_modified": False,
        "pushed": False,
    }


def test_live_catalog_keeps_static_q_lifting_declaration_only():
    policies = {
        policy.key: policy for policy in capability_catalog().boundary_policies
    }
    for key in ("prescribed_q", "strong_homeotropic_q", "strong_planar_q"):
        assert key in policies
        assert policies[key].qualified_applications == ()


def test_readme_documents_exact_execution_option_ownership():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    exact_keys = (
        "`tf32`, `molecular_field_linear_space`, "
        "`stress_divergence_sum_space`, `pointwise_execution`, "
        "`disable_q_gradient_reuse`"
    )
    assert readme.count(exact_keys) == 3
    assert "`legacy_channel`, `compiled_channel_v2` | `batch_size`, `dtype`" in readme
    assert "`pssolver.configuration.simulation`" in readme
    assert "static prescribed-Q lifting" in readme
    assert "unsupported_lifted_wall_molecular_field" in readme


def test_complete_readme_simulation_example_executes_verbatim(tmp_path, monkeypatch):
    output = tmp_path / "readme-plane"
    monkeypatch.setenv("PSSOLVER_EXAMPLE_OUTPUT", str(output))
    namespace: dict[str, object] = {}
    exec(compile(_readme_example(), "README.md", "exec"), namespace)
    result = namespace["result"]
    assert result.status is SimulationRunStatus.COMPLETE
    assert result.final_step == 2
    assert result.output_directory == output
    assert (output / "COMPLETE").is_file()


def test_historical_plane_example_sets_a_compatible_save_start_step():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    historical = readme.split("## Historical Plane application", 1)[1]
    assert "save_start_step=0" in historical


def test_robin_and_fixed_work_explanations_match_the_live_contract():
    robin = inspect.getdoc(CellCenteredRobinEigenbasisOperator.boundary_residual)
    scope = (ROOT / "notes" / "pssolver_v0_2_scope.md").read_text(
        encoding="utf-8"
    )
    assert robin is not None
    normalized_robin = " ".join(robin.split())
    normalized_scope = " ".join(scope.split())
    assert "construction" in normalized_robin
    assert "not an independent correctness gate" in normalized_robin
    assert "exact finite iteration algorithm" in normalized_scope
    assert "relative-tolerance mode" in normalized_scope


def test_w1_successor_hashes_match_every_changed_historical_source():
    successors = _read(RECORD)["historical_successor_sources"]
    r5_1_successors = _read(R5_1_RECORD)["historical_successor_sources"]
    r5_2_successors = _read(R5_2_RECORD)["historical_successor_sources"]
    r5_3_successors = _read(R5_3_RECORD)["historical_successor_sources"]
    r5_5_successors = _read(R5_5_RECORD)["historical_successor_sources"]
    r5_6_successors = _read(R5_6_RECORD)["historical_successor_sources"]
    r5_7_successors = _read(R5_7_RECORD)["historical_successor_sources"]
    r5_8_successors = _read(R5_8_RECORD)["historical_successor_sources"]
    r5_9_successors = _read(R5_9_RECORD)["historical_successor_sources"]
    r5_10_successors = _read(R5_10_RECORD)["historical_successor_sources"]
    r5_11_successors = _read(R5_11_RECORD)["historical_successor_sources"]
    r5_12_successors = _read(R5_12_RECORD)["historical_successor_sources"]
    r5_13_successors = _read(R5_13_RECORD)["historical_successor_sources"]
    g1_record = _read(G1_RECORD)
    g1_successors = {
        **g1_record["historical_successor_sources"],
        **g1_record["versioned_sources"],
    }
    g1_glibc_successors = _read(G1_GLIBC_RECORD)[
        "historical_successor_sources"
    ]
    rc6_successors = _read(RC6_R1_RECORD)["historical_successor_sources"]
    rc6_r2_successors = _read(RC6_R2_RECORD)["historical_successor_sources"]
    rc6_r3_successors = _read(RC6_R3_RECORD)["historical_successor_sources"]
    rc6_r4_successors = _read(RC6_R4_RECORD)["historical_successor_sources"]
    rc6_r5_successors = _read(RC6_R5_RECORD)["historical_successor_sources"]
    rc6_r6_successors = _read(RC6_R6_RECORD)["historical_successor_sources"]
    rc6_freeze_record = _read(RC6_FREEZE_RECORD)
    rc6_freeze_successors = {
        **rc6_freeze_record["versioned_sources"],
        **rc6_freeze_record["historical_successor_sources"],
    }
    assert successors
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        resolved = registration["current_sha256"]
        successor = r5_1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_5_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_6_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_7_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_8_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_9_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_10_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_11_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_12_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_13_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = g1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = g1_glibc_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_successors.get(relative)
        accepted = (
            []
            if successor is None
            else successor.get(
                "accepted_predecessor_sha256",
                [successor["historical_sha256"]],
            )
        )
        if successor is not None and resolved in accepted:
            resolved = successor["current_sha256"]
        successor = rc6_r2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r4_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r5_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r6_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_freeze_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolved
        assert registration["reason"]
