#!/usr/bin/env python3
"""Profile one Periodic Hermitian-repair qualification case.

The frozen P9.5 profiler remains the authority for timing, memory, and
forward/VJP correctness.  This versioned wrapper adds a separate, untimed
runtime-capability audit.  The audit counts actual transform dispatches on a
fresh runtime and records fallback identity from the runtime that owns it.

The qualified Periodic paths explicitly request eager pointwise execution.
TorchDynamo graph breaks and compile fallback therefore are not observable
events for these paths.  They are reported as inapplicable, never as invented
zero or false values.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np
import torch

from benchmarks.profile_periodic_functional import (
    PROFILE_ROLES,
    PeriodicFunctionalProfileConfig,
    _activity,
    _allocated_device,
    _atomic_json,
    _one_vjp,
    _production,
    _production_state,
    _simulation,
    _tensor_sha256,
    _tuple,
    run_profile,
)
from pssolver.functional import (
    build_functional_runtime,
    periodic_activity_functional_request,
)


QUALIFICATION_PROFILE_SCHEMA_VERSION = 1
QUALIFICATION_PROFILE_KIND = "periodic_hermitian_qualification_profile"
QUALIFICATION_VARIANTS = ("baseline", "candidate")
EAGER_INAPPLICABLE_REASON = "pointwise_execution_eager"


class _TransformDispatchCounter:
    """Count transform calls without adding timers or synchronization."""

    def __init__(self, backend: object) -> None:
        original_forward = getattr(backend, "forward", None)
        original_inverse = getattr(backend, "inverse", None)
        if not callable(original_forward) or not callable(original_inverse):
            raise TypeError("transform backend must provide forward and inverse")
        self.backend = backend
        self.original_forward = original_forward
        self.original_inverse = original_inverse
        self.forward_calls = 0
        self.inverse_calls = 0

    def install(self) -> None:
        def counted_forward(*args, **kwargs):
            self.forward_calls += 1
            return self.original_forward(*args, **kwargs)

        def counted_inverse(*args, **kwargs):
            self.inverse_calls += 1
            return self.original_inverse(*args, **kwargs)

        self.backend.forward = counted_forward
        self.backend.inverse = counted_inverse

    def restore(self) -> None:
        self.backend.forward = self.original_forward
        self.backend.inverse = self.original_inverse


@contextmanager
def _count_transform_dispatches(backend: object) -> Iterator[_TransformDispatchCounter]:
    counter = _TransformDispatchCounter(backend)
    counter.install()
    try:
        yield counter
    finally:
        counter.restore()


def _compilation_capability(pointwise_execution: str) -> dict[str, object]:
    if pointwise_execution != "eager":
        raise ValueError(
            "Hermitian-repair qualification is frozen to eager pointwise execution"
        )
    inapplicable = {
        "applicable": False,
        "value": None,
        "reason": EAGER_INAPPLICABLE_REASON,
    }
    return {
        "pointwise_execution": {
            "requested": "eager",
            "effective": "eager",
        },
        "graph_breaks": dict(inapplicable),
        "compile_fallback": dict(inapplicable),
    }


def _production_activity(simulation, device: torch.device, base: float) -> torch.Tensor:
    request = periodic_activity_functional_request(simulation.specification)
    spec = request.control_fields[0].tensor
    indices = torch.arange(
        math.prod(spec.shape),
        device=device,
        dtype=torch.float64,
    ).reshape(spec.shape)
    return base + 0.002 * torch.sin(indices * 0.071)


def _finite(values) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_physical_q(path: Path, value: torch.Tensor) -> dict[str, object]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if path.suffix != ".npy":
        raise ValueError("physical Q artifact must use a .npy filename")
    if value.ndim != 5 or value.shape[0] != 5 or value.shape[1] != 1:
        raise ValueError("physical Q tensor has an unexpected layout")
    array = (
        value[:, 0]
        .movedim(0, -1)
        .detach()
        .to(device="cpu")
        .contiguous()
        .numpy()
    )
    if array.dtype != np.float64 or not bool(np.isfinite(array).all()):
        raise ValueError("physical Q artifact must be finite float64")
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npy")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": True,
    }


def _runtime_capability_audit(
    config: PeriodicFunctionalProfileConfig,
    *,
    physical_artifact_path: Path | None = None,
) -> dict[str, object]:
    """Run one fresh, untimed operation and report actual capabilities."""

    device = _allocated_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    with tempfile.TemporaryDirectory(prefix="pssolver-hermitian-audit-") as root:
        simulation = _simulation(
            config,
            device=device,
            output_directory=Path(root) / "unused-output",
        )
        pointwise_execution = simulation.execution.options["pointwise_execution"]
        compilation = _compilation_capability(pointwise_execution)

        if config.role == "production_forward":
            runtime = _production(simulation, device)
            runtime.solver.model.parameters["alpha"] = _production_activity(
                simulation,
                device,
                config.base_activity,
            )
            with _count_transform_dispatches(
                runtime.solver.transform_backend
            ) as counter:
                runtime.advance(1)
            values = tuple(value.detach() for value in _production_state(runtime))
            metadata = runtime.to_metadata()
            if metadata.get("requested") != metadata.get("effective"):
                raise RuntimeError("production runtime requested/effective mismatch")
            if metadata.get("fallback_used") is not False:
                raise RuntimeError("production runtime fallback was used")
            runtime_selection = {
                "source": "PeriodicRuntimeAdapter.to_metadata",
                "requested": metadata["requested"],
                "effective": metadata["effective"],
                "fallback_used": metadata["fallback_used"],
                "metadata": dict(metadata),
            }
            result_kind = "state"
        else:
            request = periodic_activity_functional_request(simulation.specification)
            runtime = build_functional_runtime(request)
            controls = _activity(runtime, config.base_activity)
            state = runtime.initial_state()
            # This is repository-owned diagnostic code.  The private solver is
            # used only to install an ephemeral counter on a fresh runtime; it
            # is not a Consumer dependency or part of the public API contract.
            with _count_transform_dispatches(
                runtime._solver.transform_backend
            ) as counter:
                if config.role == "functional_forward":
                    values = tuple(
                        value.detach()
                        for value in runtime.step(state, controls, 0)
                    )
                    result_kind = "state"
                else:
                    values = _one_vjp(runtime, state, controls)
                    result_kind = "gradient"
            identity = runtime.identity().to_metadata()
            functional = identity["execution"]["functional_runtime"]
            if functional.get("fallback_used") is not False:
                raise RuntimeError("functional runtime fallback was used")
            runtime_selection = {
                "source": "FunctionalRuntimeIdentity.execution.functional_runtime",
                "kind": functional["kind"],
                "fallback_allowed": functional["fallback_allowed"],
                "fallback_used": functional["fallback_used"],
                "metadata": dict(functional),
            }

    if counter.forward_calls <= 0 or counter.inverse_calls <= 0:
        raise RuntimeError("transform dispatch audit observed no transform calls")
    if not _finite(values):
        raise RuntimeError("capability audit produced a non-finite result")
    physical_artifact = (
        None
        if physical_artifact_path is None
        else _atomic_physical_q(physical_artifact_path, values[0])
    )
    return {
        "measurement_scope": "one_fresh_untimed_operation",
        "timing_contaminated": False,
        "role": config.role,
        "runtime_selection": runtime_selection,
        "compilation": compilation,
        "transform_dispatch": {
            "semantics": "python_backend_forward_inverse_dispatches",
            "forward_calls": counter.forward_calls,
            "inverse_calls": counter.inverse_calls,
        },
        "result": {
            "kind": result_kind,
            "finite": True,
            "sha256": _tensor_sha256(values),
        },
        "physical_q_artifact": physical_artifact,
    }


def run_qualification_profile(
    config: PeriodicFunctionalProfileConfig,
    *,
    variant: str,
    repository_root: Path | None = None,
    physical_artifact_path: Path | None = None,
) -> dict[str, object]:
    if variant not in QUALIFICATION_VARIANTS:
        raise ValueError(f"variant must be one of {QUALIFICATION_VARIANTS}")
    base = run_profile(config, repository_root=repository_root)
    if physical_artifact_path is not None and (
        config.role != "production_forward" or config.trial != 1
    ):
        raise ValueError(
            "physical Q artifacts are restricted to production_forward trial 1"
        )
    capability = _runtime_capability_audit(
        config,
        physical_artifact_path=physical_artifact_path,
    )
    return {
        "schema_version": QUALIFICATION_PROFILE_SCHEMA_VERSION,
        "kind": QUALIFICATION_PROFILE_KIND,
        "qualification": "periodic_hermitian_state_repair",
        "variant": variant,
        "base_profile": base,
        "capability_audit": capability,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=QUALIFICATION_VARIANTS, required=True)
    parser.add_argument("--role", choices=PROFILE_ROLES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", nargs=3, required=True)
    parser.add_argument("--lengths", nargs=3, required=True)
    parser.add_argument("--initial-q-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--base-activity", type=float, default=0.013)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--physical-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PeriodicFunctionalProfileConfig(
        role=args.role,
        trial=args.trial,
        shape=_tuple(args.shape, int),
        lengths=_tuple(args.lengths, float),
        initial_q_path=args.initial_q_path,
        device=args.device,
        dtype=args.dtype,
        dt=args.dt,
        base_activity=args.base_activity,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
    )
    report = run_qualification_profile(
        config,
        variant=args.variant,
        repository_root=args.repository_root,
        physical_artifact_path=args.physical_artifact,
    )
    _atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
