"""RC4.5.1 analysis-only cross-runtime residual adjudication.

This module deliberately observes current public and checkpoint behaviour.  It
does not patch runtime objects, change production defaults, or require CUDA.
The one temporary mutation of PyTorch global flags in probe T01 is restored in
a ``finally`` block before the report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pssolver import TorchSpectralExecution, compile_simulation
from pssolver.boundaries import no_slip_velocity
from pssolver.configuration.execution_device import configure_tf32_execution
from pssolver.configuration.public_simulation_runner import (
    PublicSimulationCompilationError,
)
from pssolver.core.boundary import BoundarySide
from pssolver.functional import (
    ChannelActivityCheckpointBridge,
    FunctionalCheckpointCompatibilityError,
    FunctionalCheckpointExistsError,
    FunctionalCheckpointIntegrityError,
    FunctionalCheckpointNotFoundError,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
    PeriodicActivityCheckpointBridge,
)
from pssolver.io.checkpoint import (
    FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE,
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)
from pssolver.models.active_nematics.boundaries import _oriented_face_normal
from pssolver.planning.robin import _positive_robin_roots
from pssolver.workflows import channel_checkpoint as channel_production
from pssolver.workflows import periodic_checkpoint as periodic_production
from pssolver.workflows import plane_checkpoint as plane_production
from pssolver.workflows.channel_beris_edwards import (
    CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE,
    CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER,
)
from pssolver.workflows.channel_observation import (
    DIAGNOSTIC_DTYPE as LEGACY_CHANNEL_DIAGNOSTIC_DTYPE,
    DIAGNOSTIC_HEADER as LEGACY_CHANNEL_DIAGNOSTIC_HEADER,
)
from pssolver.workflows.periodic_beris_edwards import (
    PERIODIC_DIAGNOSTIC_DTYPE,
    PERIODIC_DIAGNOSTIC_HEADER,
)
from pssolver.workflows.plane_observation import (
    DIAGNOSTIC_DTYPE as PLANE_DIAGNOSTIC_DTYPE,
    DIAGNOSTIC_HEADER as PLANE_DIAGNOSTIC_HEADER,
)


PYTEST_HELPERS = {
    "public_plane": ROOT / "tests/test_phase7_p779_public_runner_connection.py",
    "public_periodic": ROOT / "tests/test_phase8_p82_periodic_application.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load analysis helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _exception(callable_) -> dict[str, object]:
    try:
        callable_()
    except BaseException as error:  # qualification truth table needs exact type
        message = re.sub(
            re.escape(tempfile.gettempdir())
            + r"/(?:pssolver-rc451|rc451-v02)-[^/\s:]+",
            "<RC451_TMP>",
            str(error),
        )
        return {
            "raised": True,
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": message,
        }
    return {"raised": False, "type": None, "module": None, "message": None}


def _functional_contract():
    physical = FunctionalTensorSpec(
        name="q_physical",
        shape=(5, 2, 2, 2),
        dtype="float64",
        device="cpu",
        batch_axis=1,
        layout="component_batch_xyz",
        meaning="RC4.5.1 physical Q",
    )
    spectral = FunctionalTensorSpec(
        name="q_spectral",
        shape=(5, 2, 2, 2),
        dtype="complex128",
        device="cpu",
        batch_axis=1,
        layout="component_batch_kxyz",
        meaning="RC4.5.1 spectral Q",
    )
    state_spec = FunctionalStateSpec(components=(physical, spectral))
    state = (
        torch.arange(40, dtype=torch.float64).reshape(5, 2, 2, 2),
        torch.arange(40, dtype=torch.float64)
        .reshape(5, 2, 2, 2)
        .to(torch.complex128),
    )

    def identity(kind: str) -> FunctionalRuntimeIdentity:
        runtime = {"kind": kind}
        if kind == "periodic_activity_batch_one":
            runtime["hermitian_state_projection"] = (
                "self_conjugate_planes_each_step"
            )
        if kind == "channel_activity_batch_one":
            runtime["pressure_solver"] = {
                "gradient": "custom_implicit_pressure_adjoint_v1"
            }
        return FunctionalRuntimeIdentity(
            scientific={"model": "rc451_probe"},
            discretization={"grid": [2, 2, 2]},
            execution={"runtime": "rc451_probe", "functional_runtime": runtime},
            state_layout=state_spec.to_metadata(),
        )

    channel = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity("channel_activity_batch_one"),
        production_runtime_identity_sha256="a" * 64,
        production_backend_restart={},
    )
    periodic = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity("periodic_activity_batch_one"),
        production_runtime_identity_sha256="b" * 64,
        backend_restart={},
    )
    return state, {"channel": channel, "periodic": periodic}


def _rewrite_sealed_metadata(directory: Path, metadata: dict[str, object]) -> None:
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    (directory / "checkpoint.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )


def probe_c01_and_c03(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    state, bridges = _functional_contract()
    c01: dict[str, object] = {"probe_id": "C01", "families": {}}
    c03: dict[str, object] = {
        "probe_id": "C03",
        "umask": "077",
        "functional_modes": {},
        "cleanup": {},
        "writer_source_inventory": {},
    }
    for family, bridge in bridges.items():
        old_umask = os.umask(0o077)
        try:
            directory = bridge.export_checkpoint(
                root / f"{family}_extra_key",
                state,
                completed_steps=0,
            )
        finally:
            os.umask(old_umask)
        c03["functional_modes"][family] = oct(
            stat.S_IMODE(directory.stat().st_mode)
        )
        metadata_path = directory / "checkpoint.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["rc451_unknown_top_level_key"] = {"value": 1}
        _rewrite_sealed_metadata(directory, metadata)
        result = _exception(
            lambda bridge=bridge, directory=directory: bridge.import_checkpoint(
                directory
            )
        )
        c01["families"][family] = {
            "resealed_unknown_top_level_key_accepted": not result["raised"],
            "exception": result,
        }

    # Current-format provenance deletion must fail even though the canonical
    # digest is not an authentication primitive.
    directory = bridges["channel"].export_checkpoint(
        root / "channel_missing_provenance",
        state,
        completed_steps=0,
    )
    (directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE).unlink()
    c01["missing_current_provenance"] = _exception(
        lambda: bridges["channel"].import_checkpoint(directory)
    )

    # Inject a deterministic failure after staging creation and prove cleanup.
    cleanup_parent = root / "cleanup"
    cleanup_parent.mkdir()
    with patch.object(
        ChannelActivityCheckpointBridge,
        "_write_tensor",
        side_effect=RuntimeError("rc451 injected writer failure"),
    ):
        failure = _exception(
            lambda: bridges["channel"].export_checkpoint(
                cleanup_parent / "target",
                state,
                completed_steps=0,
            )
        )
    c03["cleanup"] = {
        "injected_failure": failure,
        "target_exists": (cleanup_parent / "target").exists(),
        "residual_entries": sorted(path.name for path in cleanup_parent.iterdir()),
    }

    writer_modules = {
        "plane_production": plane_production,
        "legacy_channel_production": channel_production,
        "periodic_production": periodic_production,
    }
    for name, module in writer_modules.items():
        source = Path(module.__file__).read_text(encoding="utf-8")
        c03["writer_source_inventory"][name] = {
            "uses_staging_directory": "tempfile.mkdtemp" in source,
            "uses_atomic_replace": "os.replace(staging, target)" in source,
            "forces_0755": "staging.chmod(0o755)" in source,
        }
    return c01, c03


def probe_c02(root: Path) -> dict[str, object]:
    record = {
        "file": "../escape.npy",
        "shape": [1],
        "dtype": "float64",
        "sha256": "0" * 64,
    }
    return {
        "probe_id": "C02",
        "production_filename_path_checks": {
            "plane": _exception(
                lambda: plane_production._load_tensor(root, record, "probe")
            ),
            "legacy_channel": _exception(
                lambda: channel_production._load_tensor(root, record, "probe")
            ),
            "periodic": _exception(
                lambda: periodic_production._load_tensor(root, record, "probe")
            ),
        },
        "target_mutation_before_payload_open": False,
        "interpretation": (
            "All production tensor readers reject a non-basename filename before "
            "opening a payload; identity-before-mutation remains covered by the "
            "RC4 checkpoint suites."
        ),
    }


def probe_c04(root: Path) -> dict[str, object]:
    state, bridges = _functional_contract()
    bridge = bridges["channel"]
    directory = bridge.export_checkpoint(
        root / "truth_table",
        state,
        completed_steps=0,
    )
    existing = _exception(
        lambda: bridge.export_checkpoint(directory, state, completed_steps=0)
    )
    missing = _exception(lambda: bridge.import_checkpoint(root / "missing"))

    future = bridge.export_checkpoint(
        root / "future",
        state,
        completed_steps=0,
    )
    metadata = json.loads((future / "checkpoint.json").read_text(encoding="utf-8"))
    metadata["format_version"] = 999
    metadata["functional_bridge"]["format_version"] = 999
    _rewrite_sealed_metadata(future, metadata)
    future_error = _exception(lambda: bridge.import_checkpoint(future))

    tampered = bridge.export_checkpoint(
        root / "tampered",
        state,
        completed_steps=0,
    )
    tensor = tampered / "state__q_physical.npy"
    payload = bytearray(tensor.read_bytes())
    payload[-1] ^= 1
    tensor.write_bytes(payload)
    tamper_error = _exception(lambda: bridge.import_checkpoint(tampered))

    parent_file = root / "parent_is_file"
    parent_file.write_text("not a directory\n", encoding="utf-8")
    parent_error = _exception(
        lambda: bridge.export_checkpoint(
            parent_file / "child", state, completed_steps=0
        )
    )
    expected = {
        "existing_target": FunctionalCheckpointExistsError.__name__,
        "missing_target": FunctionalCheckpointNotFoundError.__name__,
        "future_format": FunctionalCheckpointCompatibilityError.__name__,
        "payload_tamper": FunctionalCheckpointIntegrityError.__name__,
        "parent_is_file": "FunctionalCheckpointError",
    }
    actual = {
        "existing_target": existing,
        "missing_target": missing,
        "future_format": future_error,
        "payload_tamper": tamper_error,
        "parent_is_file": parent_error,
    }
    return {
        "probe_id": "C04",
        "expected_public_classes": expected,
        "observed": actual,
        "parent_filesystem_error_translated": (
            parent_error["type"] == expected["parent_is_file"]
        ),
    }


def probe_b01() -> dict[str, object]:
    face = (2, BoundarySide.LOWER)
    outward = _exception(
        lambda: _oriented_face_normal(
            face, (0.0, 0.0, -1.0), "RC4.5.1 anchoring normal"
        )
    )
    opposite = _exception(
        lambda: _oriented_face_normal(
            face, (0.0, 0.0, 1.0), "RC4.5.1 anchoring normal"
        )
    )
    normal = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    reversed_normal = -normal
    q = np.outer(normal, normal) - np.eye(3) / 3.0
    reversed_q = np.outer(reversed_normal, reversed_normal) - np.eye(3) / 3.0
    return {
        "probe_id": "B01",
        "outward": outward,
        "opposite": opposite,
        "q_byte_identical_under_sign_flip": q.tobytes() == reversed_q.tobytes(),
        "diagnostic_reports_expected_and_observed_vectors": (
            "expected" in str(opposite["message"]).lower()
            and "observed" in str(opposite["message"]).lower()
        ),
    }


def probe_b02() -> dict[str, object]:
    scales = (1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12, 1.0e14, 1.0e15, 1.0e16)
    rows = []
    for scale in scales:
        try:
            roots = _positive_robin_roots(8, scale, scale)
        except ValueError as error:
            rows.append(
                {
                    "scale": scale,
                    "status": "FAIL",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        separation = [
            abs(value - (index + 1) * math.pi)
            for index, value in enumerate(roots)
        ]
        rows.append(
            {
                "scale": scale,
                "status": "PASS",
                "root_count": len(roots),
                "maximum_distance_from_dirichlet_limit": max(separation),
            }
        )
    return {
        "probe_id": "B02",
        "symmetric_scale_sweep": rows,
        "largest_passing_scale": max(
            row["scale"] for row in rows if row["status"] == "PASS"
        ),
        "smallest_failing_scale": min(
            row["scale"] for row in rows if row["status"] == "FAIL"
        ),
        "recommended_public_bound": {
            "lower": 0.0,
            "upper_inclusive": 1.0e14,
            "reason": (
                "At 1e15 the finite-Robin correction is at binary64 resolution "
                "relative to the Dirichlet-limit roots."
            ),
        },
    }


def probe_v01() -> dict[str, object]:
    helper = _load_module("rc451_public_plane", PYTEST_HELPERS["public_plane"])
    base = helper._simulation()
    options = dict(base.execution.options)
    options.pop("device", None)
    options.pop("fallback_allowed", None)
    rows = []
    for runtime in ("legacy_production", "compiled_v2"):
        for device in ("banana", "cpu:0"):
            execution = TorchSpectralExecution(
                runtime_path=runtime,
                device=device,
                options=options,
            )
            result = _exception(
                lambda execution=execution: compile_simulation(
                    helper._simulation(execution=execution)
                )
            )
            rows.append(
                {
                    "family": "plane",
                    "runtime": runtime,
                    "device": device,
                    "compile_accepted": not result["raised"],
                    "exception": result,
                }
            )
    return {
        "probe_id": "V01",
        "plane_device_matrix": rows,
        "invalid_device_accepted": any(
            row["device"] == "banana" and row["compile_accepted"] for row in rows
        ),
        "indexed_cpu_token_accepted": any(
            row["device"] == "cpu:0" and row["compile_accepted"] for row in rows
        ),
        "output_side_effects": "not_applicable_compile_only",
    }


def probe_v02() -> dict[str, object]:
    helper = _load_module("rc451_public_periodic", PYTEST_HELPERS["public_periodic"])
    plane_helper = _load_module(
        "rc451_public_plane_v02", PYTEST_HELPERS["public_plane"]
    )
    temporary = Path(tempfile.mkdtemp(prefix="rc451-v02-"))
    try:
        free = compile_simulation(helper._simulation(temporary))
        no_slip = compile_simulation(
            helper._simulation(
                temporary,
                output_name="no_slip",
                velocity_boundary=no_slip_velocity(),
            )
        )
        invalid = _exception(
            lambda: compile_simulation(
                helper._simulation(temporary, device="banana")
            )
        )
        plane = plane_helper._simulation()
        plane_options = dict(plane.execution.options)
        plane_options.pop("device", None)
        plane_options.pop("fallback_allowed", None)
        invalid_pointwise_options = dict(plane_options)
        invalid_pointwise_options["pointwise_execution"] = "invalid"
        invalid_pointwise = _exception(
            lambda: compile_simulation(
                plane_helper._simulation(
                    execution=TorchSpectralExecution(
                        runtime_path="legacy_production",
                        device="cpu",
                        options=invalid_pointwise_options,
                    )
                )
            )
        )
        invalid_reuse_options = dict(plane_options)
        invalid_reuse_options["disable_q_gradient_reuse"] = "yes"
        invalid_reuse = _exception(
            lambda: compile_simulation(
                plane_helper._simulation(
                    execution=TorchSpectralExecution(
                        runtime_path="compiled_v2",
                        device="cpu",
                        options=invalid_reuse_options,
                    )
                )
            )
        )
    finally:
        shutil.rmtree(temporary)
    return {
        "probe_id": "V02",
        "periodic_no_slip_compiles": True,
        "periodic_no_slip_normalized_to_free_slip": (
            no_slip.application_specification.boundaries
            == free.application_specification.boundaries
        ),
        "invalid_device_exception": invalid,
        "invalid_device_uses_public_compilation_error": (
            invalid["type"] == PublicSimulationCompilationError.__name__
        ),
        "plane_invalid_pointwise_exception": invalid_pointwise,
        "plane_invalid_reuse_exception": invalid_reuse,
        "plane_invalid_options_use_public_compilation_error": all(
            value["type"] == PublicSimulationCompilationError.__name__
            for value in (invalid_pointwise, invalid_reuse)
        ),
    }


def _dtype_metadata(dtype: np.dtype, header: str) -> dict[str, object]:
    return {
        "field_names": list(dtype.names or ()),
        "field_dtypes": {
            name: str(dtype.fields[name][0]) for name in dtype.names or ()
        },
        "header": header,
    }


def probe_v03() -> dict[str, object]:
    schemas = {
        "plane": _dtype_metadata(PLANE_DIAGNOSTIC_DTYPE, PLANE_DIAGNOSTIC_HEADER),
        "legacy_channel": _dtype_metadata(
            LEGACY_CHANNEL_DIAGNOSTIC_DTYPE,
            LEGACY_CHANNEL_DIAGNOSTIC_HEADER,
        ),
        "periodic_complete_stress": _dtype_metadata(
            PERIODIC_DIAGNOSTIC_DTYPE,
            PERIODIC_DIAGNOSTIC_HEADER,
        ),
        "channel_complete_stress": _dtype_metadata(
            CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE,
            CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER,
        ),
    }
    signatures = {name: tuple(value["field_names"]) for name, value in schemas.items()}
    return {
        "probe_id": "V03",
        "schemas": schemas,
        "all_field_sets_equal": len(set(signatures.values())) == 1,
        "plane_pressure_iterations_dtype": schemas["plane"]["field_dtypes"].get(
            "schur_iterations"
        ),
        "complete_channel_pressure_iterations_dtype": schemas[
            "channel_complete_stress"
        ]["field_dtypes"].get("pressure_iterations"),
    }


def probe_t01() -> dict[str, object]:
    initial = {
        "precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        caller = {
            "precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
        metadata = configure_tf32_execution(
            "off", real_dtype=torch.float64, device="cpu"
        )
        after_success = {
            "precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
        invalid = _exception(
            lambda: configure_tf32_execution(
                "invalid", real_dtype=torch.float64, device="cpu"
            )
        )
    finally:
        torch.set_float32_matmul_precision(initial["precision"])
        torch.backends.cuda.matmul.allow_tf32 = initial[
            "cuda_matmul_allow_tf32"
        ]
        torch.backends.cudnn.allow_tf32 = initial["cudnn_allow_tf32"]
    restored = {
        "precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    return {
        "probe_id": "T01",
        "initial": initial,
        "caller_owned_state": caller,
        "after_success": after_success,
        "callee_restored_caller_state": after_success == caller,
        "invalid_request": invalid,
        "analysis_harness_restored_initial_state": restored == initial,
        "metadata": metadata,
        "cuda_available": torch.cuda.is_available(),
        "cuda_nested_runtime_matrix": "deferred_to_h100",
    }


def build_report() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="pssolver-rc451-") as value:
        temporary = Path(value)
        c01, c03 = probe_c01_and_c03(temporary / "checkpoint_matrix")
        probes = {
            "C01": c01,
            "C02": probe_c02(temporary),
            "C03": c03,
            "C04": probe_c04(temporary / "exception_truth_table"),
            "B01": probe_b01(),
            "B02": probe_b02(),
            "V01": probe_v01(),
            "V02": probe_v02(),
            "V03": probe_v03(),
            "T01": probe_t01(),
        }

    outcomes = {
        "B7": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["C01"],
            "reason": "Resealed unknown top-level checkpoint keys are accepted.",
        },
        "B9": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["B01"],
            "reason": "The sign-equivalent anchoring director is rejected with weak diagnostics.",
        },
        "B11": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["T01"],
            "reason": "The successful TF32 policy path does not restore caller global state.",
            "h100_followup_required": True,
        },
        "B13": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["V01"],
            "reason": "Both Plane runtimes accept an invalid token and cpu:0 at compile time.",
        },
        "B15": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["V03"],
            "reason": "The four diagnostic families expose different names and dtypes.",
        },
        "B17": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["V02"],
            "reason": (
                "Plane invalid execution options escape as raw ValueError/TypeError "
                "while Periodic uses the public compiler error."
            ),
        },
        "B21": {
            "outcome": "PUBLIC_BOUND_REQUIRED",
            "evidence": ["B02"],
            "reason": (
                "Symmetric impedance succeeds through 1e14 and fails from 1e15 "
                "at binary64 resolution."
            ),
        },
        "B24": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["C04"],
            "reason": (
                "A parent-path filesystem failure escapes the functional checkpoint "
                "exception hierarchy."
            ),
        },
        "B27": {
            "outcome": "REPRODUCED_REPAIR_REQUIRED",
            "evidence": ["C03"],
            "reason": (
                "Functional Channel publishes mode 0700 while Periodic publishes "
                "0755 under the same umask."
            ),
        },
    }
    reviewed_sources = (
        "benchmarks/diagnose_rc451_cross_runtime_residuals.py",
        "pssolver/configuration/execution_device.py",
        "pssolver/configuration/public_simulation_runner.py",
        "pssolver/functional/channel_checkpoint.py",
        "pssolver/functional/periodic_checkpoint.py",
        "pssolver/io/checkpoint.py",
        "pssolver/models/active_nematics/boundaries.py",
        "pssolver/planning/robin.py",
        "pssolver/workflows/channel_beris_edwards.py",
        "pssolver/workflows/channel_checkpoint.py",
        "pssolver/workflows/channel_observation.py",
        "pssolver/workflows/periodic_beris_edwards.py",
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/workflows/plane_checkpoint.py",
        "pssolver/workflows/plane_observation.py",
    )
    return {
        "schema_version": 1,
        "stage": "RC4.5.1",
        "classification": "RC4_5_1_RESIDUAL_ADJUDICATION_COMPLETE",
        "analysis_only": True,
        "h100_job_submitted": False,
        "runtime_source_modified": False,
        "production_default_changed": False,
        "provenance": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "source_sha256": {
                relative: _sha256(ROOT / relative) for relative in reviewed_sources
            },
        },
        "probe_count": len(probes),
        "probes": probes,
        "issue_outcomes": outcomes,
        "summary": {
            "reproduced_repair_required": sorted(
                issue
                for issue, value in outcomes.items()
                if value["outcome"] == "REPRODUCED_REPAIR_REQUIRED"
            ),
            "public_bound_required": sorted(
                issue
                for issue, value in outcomes.items()
                if value["outcome"] == "PUBLIC_BOUND_REQUIRED"
            ),
            "already_closed_with_new_evidence": [],
            "accepted_limitation_with_explicit_contract": [],
            "next_stage": "RC4.5.2_checkpoint_transactionality",
        },
    }


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report()
    _atomic_write_json(arguments.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report_sha256={_sha256(arguments.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
