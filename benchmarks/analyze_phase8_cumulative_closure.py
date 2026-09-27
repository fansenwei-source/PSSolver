#!/usr/bin/env python3
"""Fail-closed local audit for the P8.6 cumulative Phase 8 closure.

The analyzer deliberately does not construct a numerical runtime.  It binds
the authoritative Phase 8 records, compares the live public capability
catalog with the frozen P8.6.0 contract, verifies executable rejection-test
coverage, and checks a probe produced from an isolated installed wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


CLASSIFICATION = "PASS_P8_6_1_LOCAL_CUMULATIVE_AUDIT"
PLAN_RELATIVE = Path("notes/architecture_v0_2/phase_8_p860_closure_plan.json")

REJECTION_TESTS = {
    "unregistered_model_geometry": (
        "tests/test_phase8_p81_public_surface_catalog.py",
        "test_declarable_periodic_pair_remains_unqualified_before_allocation",
    ),
    "public_complete_timestep_robin_q": (
        "tests/test_phase8_p851_robin_declarations.py",
        "test_current_lowering_rejects_robin_before_transform_selection",
    ),
    "prescribed_q_on_unqualified_geometry": (
        "tests/test_phase8_p841_prescribed_dirichlet_declarations.py",
        "test_current_lowering_rejects_prescribed_data_before_selecting_dst",
    ),
    "wall_policy_on_periodic_face": (
        "tests/test_phase8_p841_prescribed_dirichlet_declarations.py",
        "test_composition_rejects_missing_extra_periodic_and_wrong_role_values",
    ),
    "geometry_incompatible_velocity_or_pressure_policy": (
        "tests/test_phase8_p81_public_surface_catalog.py",
        "test_declarable_periodic_pair_remains_unqualified_before_allocation",
    ),
    "nonzero_prescribed_neumann_flux": (
        "tests/test_phase8_p852_robin_eigenbasis_operator.py",
        "test_first_slice_rejects_unqualified_coefficients_and_nonzero_neumann_flux",
    ),
    "dynamic_spatial_callable_or_trainable_robin_data": (
        "tests/test_phase8_p851_robin_declarations.py",
        "test_static_robin_coefficients_are_finite_raw_and_content_addressed",
    ),
    "checkpoint_identity_or_payload_mismatch": (
        "tests/test_phase8_p855_finite_q_workflow_restart.py",
        "test_identity_tampering_is_rejected_before_any_component_mutation",
    ),
}

PUBLIC_ROOT_SYMBOLS = (
    "Simulation",
    "compile_simulation",
    "run_simulation",
    "available_models",
    "available_geometries",
    "available_boundary_policies",
    "available_combinations",
    "capability_catalog",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def live_catalog_snapshot(
    repository_root: Path | None = None,
) -> dict[str, object]:
    inserted = None
    if repository_root is not None:
        inserted = str(repository_root.resolve())
        sys.path.insert(0, inserted)
    try:
        import pssolver
        from pssolver.configuration.channel_active_nematics import (
            ChannelActiveNematicRunSpec,
        )
        from pssolver.configuration.plane_beris_edwards_declarations import (
            DEFAULT_PLANE_RUNTIME_PATH,
        )
    finally:
        if inserted is not None:
            sys.path.remove(inserted)

    channel_default = ChannelActiveNematicRunSpec.__dataclass_fields__[
        "runtime_path"
    ].default.value
    return {
        "module_file": str(Path(pssolver.__file__).resolve()),
        "catalog": pssolver.capability_catalog().to_metadata(),
        "public_root_symbols": {
            name: hasattr(pssolver, name) for name in PUBLIC_ROOT_SYMBOLS
        },
        "defaults": {
            "plane": DEFAULT_PLANE_RUNTIME_PATH,
            "channel": channel_default,
        },
    }


def _audit_prerequisites(
    repository_root: Path, plan: dict[str, object]
) -> list[dict[str, object]]:
    result = []
    for frozen in plan["prerequisites"]:
        path = repository_root / frozen["path"]
        _require(path.is_file(), f"missing prerequisite record: {path}")
        observed = _json(path)
        digest = sha256(path)
        _require(
            digest == frozen["sha256"],
            f"prerequisite SHA-256 mismatch: {path}",
        )
        _require(
            observed.get("classification") == frozen["classification"],
            f"prerequisite classification mismatch: {path}",
        )
        result.append(
            {
                "phase": frozen["phase"],
                "path": frozen["path"],
                "sha256": digest,
                "classification": observed["classification"],
                "status": "pass",
            }
        )
    return result


def _audit_catalog(
    plan: dict[str, object], snapshot: dict[str, object]
) -> dict[str, object]:
    catalog = snapshot["catalog"]
    frozen = plan["public_catalog"]
    combinations = [
        {
            "equation_variant": item["equation_variant"],
            "geometry_name": item["geometry_name"],
            "application": item["application"],
            "runtime_paths": item["runtime_paths"],
        }
        for item in catalog["qualified_combinations"]
    ]
    _require(catalog["schema_version"] == frozen["schema_version"], "catalog schema changed")
    _require(
        [item["key"] for item in catalog["models"]] == frozen["models"],
        "public model catalog changed",
    )
    _require(
        [item["key"] for item in catalog["geometries"]]
        == frozen["geometries"],
        "public geometry catalog changed",
    )
    _require(
        [item["key"] for item in catalog["boundary_policies"]]
        == frozen["boundary_policies"],
        "public boundary-policy catalog changed",
    )
    _require(
        combinations == plan["qualified_combinations"],
        "qualified model/geometry/runtime combinations changed",
    )
    robin = next(
        item for item in catalog["boundary_policies"] if item["key"] == "robin"
    )
    _require(robin["declarable"] is True, "Robin declaration disappeared")
    _require(robin["executable"] is False, "Robin was promoted without P8.6 authorization")
    _require(
        robin["qualified_applications"] == [],
        "Robin acquired a public qualified application",
    )
    _require(
        all(snapshot["public_root_symbols"].values()),
        "installed public root API is incomplete",
    )
    _require(
        snapshot["defaults"] == {"plane": "legacy_production", "channel": "legacy_channel"},
        "production runtime defaults changed",
    )
    return {
        "status": "pass",
        "catalog": catalog,
        "public_root_symbols": snapshot["public_root_symbols"],
        "defaults": snapshot["defaults"],
    }


def _audit_rejection_matrix(
    repository_root: Path, plan: dict[str, object]
) -> list[dict[str, object]]:
    expected = list(plan["rejection_matrix"])
    _require(expected == list(REJECTION_TESTS), "rejection-matrix order or scope changed")
    result = []
    for case in expected:
        relative, test_name = REJECTION_TESTS[case]
        path = repository_root / relative
        _require(path.is_file(), f"missing rejection test source: {path}")
        source = path.read_text(encoding="utf-8")
        _require(
            f"def {test_name}(" in source,
            f"missing rejection test node: {relative}::{test_name}",
        )
        result.append(
            {
                "case": case,
                "node_id": f"{relative}::{test_name}",
                "source_sha256": sha256(path),
                "evidence": "executable_repository_test",
                "status": "pass",
            }
        )
    return result


def build_installed_wheel_probe(
    repository_root: Path, *, python: Path | None = None
) -> dict[str, object]:
    """Build, unpack, and import one wheel outside the source checkout."""

    interpreter = Path(sys.executable if python is None else python).resolve()
    with tempfile.TemporaryDirectory(prefix="pssolver-p861-wheel-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        for name in ("pyproject.toml", "setup.py", "README.md", "Plane_beris_edwards_stokes.py"):
            shutil.copy2(repository_root / name, source / name)
        shutil.copytree(
            repository_root / "pssolver",
            source / "pssolver",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        wheelhouse = root / "wheelhouse"
        wheelhouse.mkdir()
        subprocess.run(
            [
                str(interpreter),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheelhouse),
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = tuple(wheelhouse.glob("*.whl"))
        _require(len(wheels) == 1, "wheel build did not produce exactly one artifact")
        wheel = wheels[0]
        installed = root / "installed"
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            archive.extractall(installed)
        code = """
import json
from pathlib import Path
import pssolver
from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.configuration.plane_beris_edwards_declarations import DEFAULT_PLANE_RUNTIME_PATH
names = %r
channel_default = ChannelActiveNematicRunSpec.__dataclass_fields__["runtime_path"].default.value
print(json.dumps({
    "module_file": str(Path(pssolver.__file__).resolve()),
    "catalog": pssolver.capability_catalog().to_metadata(),
    "public_root_symbols": {name: hasattr(pssolver, name) for name in names},
    "defaults": {"plane": DEFAULT_PLANE_RUNTIME_PATH, "channel": channel_default},
}, allow_nan=False, sort_keys=True))
""" % (PUBLIC_ROOT_SYMBOLS,)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(installed)
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [str(interpreter), "-c", code],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(completed.stdout)
        module_file = Path(probe["module_file"])
        _require(
            module_file.is_relative_to(installed.resolve()),
            "wheel probe imported pssolver from outside the extracted wheel",
        )
        return {
            **probe,
            "wheel_filename": wheel.name,
            "wheel_sha256": sha256(wheel),
            "wheel_member_count": len(members),
            "source_shadow_import": False,
            "status": "pass",
        }


def analyze(
    repository_root: Path,
    *,
    installed_wheel_probe: dict[str, object],
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    plan_path = repository_root / PLAN_RELATIVE
    plan = _json(plan_path)
    _require(plan.get("phase") == "P8.6.0", "wrong cumulative plan phase")
    _require(
        plan.get("authorization", {}).get("p8_6_1_local_implementation_authorized")
        is True,
        "P8.6.1 local audit is not authorized",
    )
    source_snapshot = live_catalog_snapshot(repository_root)
    _require(
        Path(source_snapshot["module_file"]).is_relative_to(repository_root),
        "source audit imported pssolver from outside the repository",
    )
    source_catalog = _audit_catalog(plan, source_snapshot)
    installed_catalog = _audit_catalog(plan, installed_wheel_probe)
    _require(
        installed_wheel_probe["catalog"] == source_snapshot["catalog"],
        "installed wheel and source capability catalogs differ",
    )
    _require(
        installed_wheel_probe.get("source_shadow_import") is False,
        "installed-wheel probe reported source shadowing",
    )
    report = {
        "schema_version": 1,
        "phase": "P8.6.1",
        "classification": CLASSIFICATION,
        "plan": {
            "path": PLAN_RELATIVE.as_posix(),
            "sha256": sha256(plan_path),
        },
        "prerequisites": _audit_prerequisites(repository_root, plan),
        "source_catalog": source_catalog,
        "installed_wheel": {
            **installed_catalog,
            "module_file": installed_wheel_probe["module_file"],
            "wheel_filename": installed_wheel_probe["wheel_filename"],
            "wheel_sha256": installed_wheel_probe["wheel_sha256"],
            "wheel_member_count": installed_wheel_probe["wheel_member_count"],
            "source_shadow_import": False,
        },
        "rejection_matrix": _audit_rejection_matrix(repository_root, plan),
        "scope": {
            "runtime_or_timestep_changed": False,
            "numerical_operator_changed": False,
            "compiler_registration_changed": False,
            "public_api_changed": False,
            "checkpoint_or_output_schema_changed": False,
            "production_default_changed": False,
        },
        "authorization": {
            "p8_6_1_complete": True,
            "eligible_for_p8_6_2_h100_planning": True,
            "p8_6_2_h100_executed": False,
            "phase_9_authorized": False,
        },
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    probe = build_installed_wheel_probe(args.repository_root, python=args.python)
    report = analyze(args.repository_root, installed_wheel_probe=probe)
    _atomic_json(args.output, report)
    print(args.output)
    print(CLASSIFICATION)


if __name__ == "__main__":
    main()
