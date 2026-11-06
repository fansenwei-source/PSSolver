"""Fixed-cotangent first-order VJP records, using only the functional runtime API."""

from __future__ import annotations

from pathlib import Path

import torch

from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import RuntimeCase


def fixed_cotangent(value: torch.Tensor) -> torch.Tensor:
    """Use a deterministic CPU cotangent, including imaginary spectral components."""
    index = torch.arange(value.numel(), dtype=value.real.dtype, device="cpu").reshape(value.shape)
    real = torch.sin(index * 0.017) / max(1, value.numel())
    if value.is_complex():
        return torch.complex(real, torch.cos(index * 0.013) / max(1, value.numel()))
    return real


def vjp_record(scene: dict, root: Path) -> dict:
    """Compare derivatives of the entire 32-step trajectory and final observations."""
    case = RuntimeCase(scene, root)
    initial = tuple(value.detach().clone().requires_grad_(True) for value in case.state)
    activity = case.controls["activity"].detach().clone().requires_grad_(True)
    state = initial
    with case.runtime.execution_scope():
        for step in range(scene["steps"]):
            state = case.runtime.step(state, {"activity": activity}, step)
        observed = case.runtime.observe(state, {"activity": activity})
        outputs = (*state, *(observed[key] for key in sorted(observed)))
        cotangents = tuple(fixed_cotangent(value) for value in outputs)
        gradients = torch.autograd.grad(outputs, (*initial, activity), grad_outputs=cotangents)
    result = {
        "steps": scene["steps"],
        "output_names": [*(f"state_{i}" for i in range(len(state))), *sorted(observed)],
        "outputs": [tensor_record(value) for value in outputs],
        "cotangents": [tensor_record(value) for value in cotangents],
        "gradients": [tensor_record(value) for value in gradients],
        "identity": case.runtime.identity().to_metadata(),
    }
    if not all(
        item["finite"] for key in ("outputs", "cotangents", "gradients") for item in result[key]
    ):
        raise RuntimeError("Nonfinite functional trajectory/VJP: no exception authorized")
    return result
