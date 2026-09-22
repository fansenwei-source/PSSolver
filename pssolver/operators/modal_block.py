"""Bound fixed-size modal block solve for the Phase 4 qualification path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec


MODAL_BLOCK_IMPLEMENTATIONS = (
    "torch_linalg_reference",
    "closed_form_2x2",
)


def _storage_identity(value: torch.Tensor) -> tuple[str, int]:
    return str(value.device), int(value.untyped_storage().data_ptr())


def _positive_finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class TwoComponentModalSolveWorkspace:
    """Preallocated output and scratch for a closed-form 2-by-2 solve."""

    output: torch.Tensor
    diagonal_0: torch.Tensor
    diagonal_1: torch.Tensor
    determinant: torch.Tensor
    temporary: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                self.output,
                self.diagonal_0,
                self.diagonal_1,
                self.determinant,
                self.temporary,
            )
        ):
            raise TypeError("modal solve workspace entries must be tensors")
        mode_shape = tuple(self.output.shape[:-1])
        if self.output.ndim < 1 or self.output.shape[-1] != 2:
            raise ValueError("workspace output must have trailing component size 2")
        for value in (
            self.diagonal_0,
            self.diagonal_1,
            self.determinant,
            self.temporary,
        ):
            if tuple(value.shape) != mode_shape:
                raise ValueError("workspace scratch shape does not match output")
            if value.dtype != self.output.dtype or value.device != self.output.device:
                raise ValueError("workspace tensors must share dtype and device")
        identities = {
            _storage_identity(value)
            for value in (
                self.output,
                self.diagonal_0,
                self.diagonal_1,
                self.determinant,
                self.temporary,
            )
        }
        if len(identities) != 5:
            raise ValueError("workspace tensors must not share storage")

    @property
    def mode_shape(self) -> tuple[int, ...]:
        return tuple(self.output.shape[:-1])

    @property
    def allocated_tensor_count(self) -> int:
        return 5

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode_shape": list(self.mode_shape),
            "output_shape": list(self.output.shape),
            "dtype": str(self.output.dtype),
            "device": str(self.output.device),
            "allocated_tensor_count": self.allocated_tensor_count,
            "bounded": True,
        }


@dataclass(frozen=True, slots=True)
class BoundTwoComponentModalOperator:
    """Device-bound mode-local coefficients with one selected solve path."""

    spec: TwoComponentModalOperatorSpec
    coefficients: torch.Tensor
    implementation: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TwoComponentModalOperatorSpec):
            raise TypeError("spec must be TwoComponentModalOperatorSpec")
        if not isinstance(self.coefficients, torch.Tensor):
            raise TypeError("coefficients must be a tensor")
        if self.coefficients.ndim < 2 or self.coefficients.shape[-2:] != (2, 2):
            raise ValueError("bound coefficients must have trailing shape (2, 2)")
        if self.coefficients.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("bound coefficients must use a complex dtype")
        if not bool(torch.isfinite(self.coefficients).all()):
            raise ValueError("bound coefficients must be finite")
        if self.implementation not in MODAL_BLOCK_IMPLEMENTATIONS:
            raise ValueError(
                f"implementation must be one of {MODAL_BLOCK_IMPLEMENTATIONS}"
            )
        if (
            self.implementation == "torch_linalg_reference"
            and self.coefficients.device.type != "cpu"
        ):
            raise ValueError("torch_linalg_reference is CPU-only")

    @property
    def mode_shape(self) -> tuple[int, ...]:
        return tuple(self.coefficients.shape[:-2])

    @property
    def device(self) -> torch.device:
        return self.coefficients.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficients.dtype

    def allocate_workspace(self) -> TwoComponentModalSolveWorkspace:
        mode_shape = self.mode_shape
        options = {"dtype": self.dtype, "device": self.device}
        return TwoComponentModalSolveWorkspace(
            output=torch.empty((*mode_shape, 2), **options),
            diagonal_0=torch.empty(mode_shape, **options),
            diagonal_1=torch.empty(mode_shape, **options),
            determinant=torch.empty(mode_shape, **options),
            temporary=torch.empty(mode_shape, **options),
        )

    def _validate_rhs(self, rhs: torch.Tensor) -> None:
        if not isinstance(rhs, torch.Tensor):
            raise TypeError("modal block RHS must be a tensor")
        if tuple(rhs.shape) != (*self.mode_shape, 2):
            raise ValueError("modal block RHS shape or component packing is invalid")
        if rhs.dtype != self.dtype:
            raise ValueError("modal block RHS dtype does not match coefficients")
        if rhs.device != self.device:
            raise ValueError("modal block RHS device does not match coefficients")
        if not bool(torch.isfinite(rhs).all()):
            raise ValueError("modal block RHS must be finite")

    def _validate_workspace(
        self,
        rhs: torch.Tensor,
        workspace: TwoComponentModalSolveWorkspace,
    ) -> None:
        if not isinstance(workspace, TwoComponentModalSolveWorkspace):
            raise TypeError("workspace must be TwoComponentModalSolveWorkspace")
        if (
            workspace.mode_shape != self.mode_shape
            or workspace.output.dtype != self.dtype
            or workspace.output.device != self.device
        ):
            raise ValueError("workspace does not match bound modal operator")
        if _storage_identity(rhs) in {
            _storage_identity(workspace.output),
            _storage_identity(workspace.diagonal_0),
            _storage_identity(workspace.diagonal_1),
            _storage_identity(workspace.determinant),
            _storage_identity(workspace.temporary),
        }:
            raise ValueError("RHS must not share storage with solve workspace")

    def _prepare_determinant(
        self,
        alpha: float,
        workspace: TwoComponentModalSolveWorkspace,
    ) -> None:
        coefficient = self.coefficients
        workspace.diagonal_0.copy_(coefficient[..., 0, 0]).neg_().add_(alpha)
        workspace.diagonal_1.copy_(coefficient[..., 1, 1]).neg_().add_(alpha)
        torch.mul(
            workspace.diagonal_0,
            workspace.diagonal_1,
            out=workspace.determinant,
        )
        torch.mul(
            coefficient[..., 0, 1],
            coefficient[..., 1, 0],
            out=workspace.temporary,
        )
        workspace.determinant.sub_(workspace.temporary)
        scale = torch.maximum(
            torch.maximum(
                workspace.diagonal_0.abs(),
                workspace.diagonal_1.abs(),
            ),
            torch.maximum(
                coefficient[..., 0, 1].abs(),
                coefficient[..., 1, 0].abs(),
            ),
        )
        tolerance = 64.0 * torch.finfo(self.dtype).eps * torch.maximum(
            torch.ones((), dtype=scale.dtype, device=scale.device),
            scale.square(),
        )
        if bool(torch.any(workspace.determinant.abs() <= tolerance)):
            raise ValueError("modal block matrix is singular or numerically rejected")

    def solve_into(
        self,
        rhs: torch.Tensor,
        *,
        alpha: float,
        workspace: TwoComponentModalSolveWorkspace,
    ) -> torch.Tensor:
        """Solve ``(alpha I - L(k)) x(k) = rhs(k)`` into fixed output."""

        self._validate_rhs(rhs)
        self._validate_workspace(rhs, workspace)
        normalized_alpha = _positive_finite(alpha, "alpha")
        self._prepare_determinant(normalized_alpha, workspace)

        if self.implementation == "torch_linalg_reference":
            identity = torch.eye(2, dtype=self.dtype, device=self.device)
            matrices = normalized_alpha * identity - self.coefficients
            solution = torch.linalg.solve(matrices, rhs.unsqueeze(-1)).squeeze(-1)
            workspace.output.copy_(solution)
            return workspace.output

        coefficient = self.coefficients
        output_0 = workspace.output[..., 0]
        output_1 = workspace.output[..., 1]
        torch.mul(workspace.diagonal_1, rhs[..., 0], out=output_0)
        torch.mul(
            coefficient[..., 0, 1],
            rhs[..., 1],
            out=workspace.temporary,
        )
        output_0.add_(workspace.temporary).div_(workspace.determinant)
        torch.mul(coefficient[..., 1, 0], rhs[..., 0], out=output_1)
        torch.mul(
            workspace.diagonal_0,
            rhs[..., 1],
            out=workspace.temporary,
        )
        output_1.add_(workspace.temporary).div_(workspace.determinant)
        return workspace.output

    def solve(self, rhs: torch.Tensor, *, alpha: float) -> torch.Tensor:
        """Allocate a bounded workspace once and return an owned solution."""

        workspace = self.allocate_workspace()
        return self.solve_into(rhs, alpha=alpha, workspace=workspace).clone()

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "bound_two_component_modal_operator",
            "spec": self.spec.to_metadata(),
            "coefficient_shape": list(self.coefficients.shape),
            "coefficient_trailing_shape": [2, 2],
            "dtype": str(self.dtype),
            "device": str(self.device),
            "implementation": self.implementation,
            "component_packing": "trailing_component_axis_size_2",
            "mode_independent_execution": True,
            "bounded_workspace": True,
            "gpu_capable": self.implementation == "closed_form_2x2",
        }


def bind_two_component_modal_operator(
    spec: TwoComponentModalOperatorSpec,
    laplacian_eigenvalues: torch.Tensor,
    *,
    component_order: tuple[str, str],
    geometry_identity: str,
    basis_signature: tuple[str, ...],
    spectral_dtype: torch.dtype,
    implementation: str,
) -> BoundTwoComponentModalOperator:
    """Bind tensor-free coefficients to one exact mode layout and device."""

    if not isinstance(spec, TwoComponentModalOperatorSpec):
        raise TypeError("spec must be TwoComponentModalOperatorSpec")
    if component_order != spec.component_order:
        raise ValueError("component order does not match modal block declaration")
    if geometry_identity != spec.geometry_identity:
        raise ValueError("geometry capability does not match modal block declaration")
    if basis_signature != spec.basis_signature:
        raise ValueError("basis capability does not match modal block declaration")
    if not isinstance(laplacian_eigenvalues, torch.Tensor):
        raise TypeError("laplacian_eigenvalues must be a tensor")
    if laplacian_eigenvalues.ndim < 1 or laplacian_eigenvalues.numel() == 0:
        raise ValueError("laplacian_eigenvalues must be nonempty")
    if spectral_dtype not in (torch.complex64, torch.complex128):
        raise ValueError("spectral_dtype must be complex64 or complex128")
    expected_real_dtype = (
        torch.float32 if spectral_dtype is torch.complex64 else torch.float64
    )
    if laplacian_eigenvalues.dtype is not expected_real_dtype:
        raise ValueError("Laplacian dtype is incompatible with spectral dtype")
    if not bool(torch.isfinite(laplacian_eigenvalues).all()):
        raise ValueError("laplacian_eigenvalues must be finite")
    if implementation not in MODAL_BLOCK_IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {MODAL_BLOCK_IMPLEMENTATIONS}"
        )

    coefficients = torch.empty(
        (*laplacian_eigenvalues.shape, 2, 2),
        dtype=spectral_dtype,
        device=laplacian_eigenvalues.device,
    )
    coefficients[..., 0, 0] = (
        spec.coupling[0][0]
        + spec.diffusion[0] * laplacian_eigenvalues
    )
    coefficients[..., 0, 1] = spec.coupling[0][1]
    coefficients[..., 1, 0] = spec.coupling[1][0]
    coefficients[..., 1, 1] = (
        spec.coupling[1][1]
        + spec.diffusion[1] * laplacian_eigenvalues
    )
    return BoundTwoComponentModalOperator(
        spec=spec,
        coefficients=coefficients,
        implementation=implementation,
    )


__all__ = [
    "MODAL_BLOCK_IMPLEMENTATIONS",
    "BoundTwoComponentModalOperator",
    "TwoComponentModalSolveWorkspace",
    "bind_two_component_modal_operator",
]
