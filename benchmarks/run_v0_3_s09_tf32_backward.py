#!/usr/bin/env python3
"""Qualify public functional forward/backward TF32 ownership on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from benchmarks.plan_v0_3_s09_h100 import (
    CALLER_STATES,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
)


SCHEMA = "pssolver.v0_3.s09.tf32_backward.v1"
OFF = {
    "torch_float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def _set_state(torch, value: dict[str, object]) -> None:
    torch.set_float32_matmul_precision(str(value["float32_matmul_precision"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(value["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(value["cudnn_allow_tf32"])


def run(support_root: Path, *, device: str = "cuda") -> dict[str, object]:
    import pssolver
    import torch
    import pssolver.functional.api as stable
    from pssolver.configuration.execution_device import snapshot_tf32_execution_state

    provider_import = Path(pssolver.__file__).resolve()
    if "site-packages" not in str(provider_import):
        raise RuntimeError("S-09 TF32 gate requires an installed Provider wheel")
    if device not in {"cpu", "cuda"}:
        raise ValueError("S-09 TF32 gate device must be cpu or cuda")
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S-09 TF32 gate requires one visible CUDA device")
    support_root = support_root.expanduser().resolve()
    sys.path.insert(0, str(support_root))
    try:
        from tests.test_phase9_p92_periodic_functional_runtime import (
            _control as periodic_control,
            _simulation as periodic_simulation,
        )
        from tests.test_phase9_p974_channel_functional_runtime import (
            _control as channel_control,
            _simulation as channel_simulation,
        )
    finally:
        sys.path.remove(str(support_root))

    def build(family: str, root: Path):
        # The public fixture writes ``root/functional_source`` directly.  Each
        # qualification case owns a distinct, initially absent parent.
        root.mkdir(parents=True, exist_ok=False)
        if family == "periodic":
            simulation = periodic_simulation(root, device=device)
            request = stable.periodic_activity_functional_request(simulation.specification)
            runtime = stable.build_functional_runtime(request)
            controls = periodic_control(runtime, requires_grad=True)
        else:
            simulation = channel_simulation(root, device=device)
            request = stable.channel_activity_functional_request(simulation.specification)
            runtime = stable.build_channel_activity_functional_runtime(request)
            controls = channel_control(runtime, requires_grad=True)
        return runtime, controls

    def loss(state):
        value = state[0]
        weights = torch.linspace(
            0.5, 1.5, value.numel(), dtype=value.dtype, device=value.device
        ).reshape(value.shape)
        return (value * weights).sum()

    original = snapshot_tf32_execution_state()
    rows: list[dict[str, object]] = []
    outside_scope_preserved = True
    nested_lifo = True
    exception_restored = True
    try:
        with tempfile.TemporaryDirectory(prefix="pssolver-s09-tf32-") as temporary:
            temporary_root = Path(temporary)
            for family in ("periodic", "channel"):
                for index, caller in enumerate(CALLER_STATES):
                    runtime, controls = build(family, temporary_root / f"{family}_{index}")
                    _set_state(torch, caller)
                    before = snapshot_tf32_execution_state()
                    observed = []
                    controls["activity"].register_hook(
                        lambda gradient: (
                            observed.append(snapshot_tf32_execution_state().to_metadata()),
                            gradient,
                        )[1]
                    )
                    with runtime.execution_scope() as policy:
                        state = runtime.step(runtime.initial_state(), controls, 0)
                        loss(state).backward()
                        inside = snapshot_tf32_execution_state().to_metadata()
                    after = snapshot_tf32_execution_state()
                    if policy.get("tf32_effective") is not False or inside != OFF:
                        raise RuntimeError(f"{family} declared TF32 policy was not effective")
                    if not observed or any(value != OFF for value in observed):
                        raise RuntimeError(f"{family} backward escaped the public scope")
                    if after != before:
                        raise RuntimeError(f"{family} caller state was not restored")
                    rows.append(
                        {
                            "family": family,
                            "caller_state": caller,
                            "inside": inside,
                            "backward_observations": observed,
                            "restored": True,
                        }
                    )

                runtime, controls = build(family, temporary_root / f"{family}_outside")
                caller = CALLER_STATES[-1]
                _set_state(torch, caller)
                before = snapshot_tf32_execution_state()
                observed = []
                controls["activity"].register_hook(
                    lambda gradient: (
                        observed.append(snapshot_tf32_execution_state()), gradient
                    )[1]
                )
                state = runtime.step(runtime.initial_state(), controls, 0)
                loss(state).backward()
                outside_scope_preserved = outside_scope_preserved and bool(observed) and all(
                    value == before for value in observed
                ) and snapshot_tf32_execution_state() == before

                _set_state(torch, CALLER_STATES[-1])
                caller_before = snapshot_tf32_execution_state()
                try:
                    with runtime.execution_scope():
                        outer = snapshot_tf32_execution_state()
                        with runtime.execution_scope():
                            nested_lifo = nested_lifo and snapshot_tf32_execution_state() == outer
                        nested_lifo = nested_lifo and snapshot_tf32_execution_state() == outer
                        raise RuntimeError("qualification-owned exception")
                except RuntimeError as error:
                    if str(error) != "qualification-owned exception":
                        raise
                exception_restored = (
                    exception_restored
                    and snapshot_tf32_execution_state() == caller_before
                )
    finally:
        _set_state(
            torch,
            {
                "float32_matmul_precision": original.torch_float32_matmul_precision,
                "cuda_matmul_allow_tf32": original.cuda_matmul_allow_tf32,
                "cudnn_allow_tf32": original.cudnn_allow_tf32,
            },
        )
    if len(rows) != 12 or not (outside_scope_preserved and nested_lifo and exception_restored):
        raise RuntimeError("S-09 TF32 backward contract failed")
    if stable.FUNCTIONAL_API_VERSION != "1.1":
        raise RuntimeError("functional API version differs")
    return {
        "schema": SCHEMA,
        "name": "tf32_forward_backward_scope",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "execution_device": device,
        "caller_state_count": 6,
        "functional_case_count": 12,
        "forward_observed_policy": True,
        "backward_observed_policy": True,
        "success_restored": True,
        "exception_restored": exception_restored,
        "nested_lifo": nested_lifo,
        "outside_scope_preserved": outside_scope_preserved,
        "rows": rows,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, run(args.support_root, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
