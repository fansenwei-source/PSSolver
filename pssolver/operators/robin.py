"""CPU-constructed cell-centered Robin eigenbasis for CPU or CUDA use."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from pssolver.planning.robin import CellCenteredRobinEigenbasisPlan


def _tensor_sha256(value: torch.Tensor) -> str:
    materialized = value.detach().cpu().contiguous()
    header = json.dumps(
        {
            "dtype": str(materialized.dtype),
            "shape": list(materialized.shape),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(materialized.numpy().tobytes(order="C"))
    return digest.hexdigest()


class CellCenteredRobinEigenbasisOperator:
    """Materialized float64 operator for one bounded Plane axis."""

    __slots__ = (
        "plan",
        "dtype",
        "device",
        "coordinates",
        "basis_matrix",
        "inverse_basis_matrix",
        "basis_scales",
        "wavenumbers",
        "eigenvalues",
        "condition_number",
        "lift_intercept",
        "lift_slope",
        "lift_values",
        "materialized_sha256",
        "_lower_basis_values",
        "_lower_basis_derivatives",
        "_upper_basis_values",
        "_upper_basis_derivatives",
        "_operation_counts",
    )

    def __init__(
        self,
        plan: CellCenteredRobinEigenbasisPlan,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(plan, CellCenteredRobinEigenbasisPlan):
            raise TypeError("plan must be a CellCenteredRobinEigenbasisPlan")
        if dtype is not torch.float64:
            raise ValueError("Robin eigenbasis supports torch.float64 only")
        normalized_device = torch.device(device)
        if normalized_device.type not in {"cpu", "cuda"}:
            raise ValueError("Robin eigenbasis supports CPU or CUDA only")
        if normalized_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            if normalized_device.index is None:
                normalized_device = torch.device(
                    "cuda",
                    torch.cuda.current_device(),
                )
        self.plan = plan
        self.dtype = dtype
        self.device = normalized_device
        coordinates = (
            torch.arange(plan.size, dtype=torch.float64) + 0.5
        ) * (plan.length / plan.size)
        wavenumbers = torch.tensor(
            plan.wavenumbers,
            dtype=torch.float64,
        )
        lower_impedance = (
            plan.lower.alpha.value / plan.lower.beta.value
        )
        unscaled = torch.empty(
            (plan.size, plan.size),
            dtype=torch.float64,
        )
        lower_values = torch.empty(plan.size, dtype=torch.float64)
        lower_derivatives = torch.empty(plan.size, dtype=torch.float64)
        upper_values = torch.empty(plan.size, dtype=torch.float64)
        upper_derivatives = torch.empty(plan.size, dtype=torch.float64)
        for index, wavenumber in enumerate(plan.wavenumbers):
            if wavenumber == 0.0:
                column = torch.ones_like(coordinates)
                lower_values[index] = 1.0
                lower_derivatives[index] = 0.0
                upper_values[index] = 1.0
                upper_derivatives[index] = 0.0
            else:
                column = torch.cos(wavenumber * coordinates) + (
                    lower_impedance / wavenumber
                ) * torch.sin(wavenumber * coordinates)
                lower_values[index] = 1.0
                lower_derivatives[index] = lower_impedance
                upper_values[index] = math.cos(wavenumber * plan.length) + (
                    lower_impedance / wavenumber
                ) * math.sin(wavenumber * plan.length)
                upper_derivatives[index] = (
                    -wavenumber * math.sin(wavenumber * plan.length)
                    + lower_impedance * math.cos(wavenumber * plan.length)
                )
            unscaled[:, index] = column
        scales = torch.linalg.vector_norm(unscaled, dim=0)
        if not bool(torch.isfinite(scales).all().item()) or bool(
            (scales <= 0.0).any().item()
        ):
            raise ValueError("Robin sampled basis has an invalid column norm")
        basis_matrix = (unscaled / scales).contiguous()
        inverse_basis_matrix = torch.linalg.inv(basis_matrix).contiguous()
        self.condition_number = float(
            torch.linalg.cond(basis_matrix).item()
        )
        if not math.isfinite(self.condition_number):
            raise ValueError("Robin sampled basis is singular")
        if self.condition_number > 1.0e10:
            raise ValueError(
                "Robin sampled basis exceeds the P8.5.2 conditioning limit"
            )
        self.coordinates = coordinates.to(self.device)
        self.wavenumbers = wavenumbers.to(self.device)
        self.basis_scales = scales.to(self.device)
        self.basis_matrix = basis_matrix.to(self.device)
        self.inverse_basis_matrix = inverse_basis_matrix.to(self.device)
        self.eigenvalues = self.wavenumbers.square()
        self._lower_basis_values = (lower_values / scales).to(
            self.device
        ).contiguous()
        self._lower_basis_derivatives = (
            lower_derivatives / scales
        ).to(self.device).contiguous()
        self._upper_basis_values = (upper_values / scales).to(
            self.device
        ).contiguous()
        self._upper_basis_derivatives = (
            upper_derivatives / scales
        ).to(self.device).contiguous()
        self.lift_intercept, self.lift_slope = self._solve_affine_lift()
        self.lift_values = (
            self.lift_intercept + self.lift_slope * self.coordinates
        ).contiguous()
        identity_payload = torch.cat(
            (
                self.basis_matrix.reshape(-1),
                self.inverse_basis_matrix.reshape(-1),
                self.lift_values,
            )
        )
        self.materialized_sha256 = _tensor_sha256(identity_payload)
        self._operation_counts = {
            "forward_transform": 0,
            "inverse_transform": 0,
            "helmholtz_apply": 0,
            "helmholtz_solve": 0,
        }

    def _solve_affine_lift(self) -> tuple[float, float]:
        lower = self.plan.lower
        upper = self.plan.upper
        if lower.alpha.value == 0.0 and upper.alpha.value == 0.0:
            if lower.gamma.value != 0.0 or upper.gamma.value != 0.0:
                raise ValueError(
                    "P8.5.2 does not support prescribed nonzero Neumann flux"
                )
            return 0.0, 0.0
        matrix = torch.tensor(
            (
                (lower.alpha.value, -lower.beta.value),
                (
                    upper.alpha.value,
                    upper.alpha.value * self.plan.length
                    + upper.beta.value,
                ),
            ),
            dtype=self.dtype,
        )
        right_hand_side = torch.tensor(
            (lower.gamma.value, upper.gamma.value),
            dtype=self.dtype,
        )
        solution = torch.linalg.solve(matrix, right_hand_side)
        if not bool(torch.isfinite(solution).all().item()):
            raise ValueError("Robin affine lift is not finite")
        return float(solution[0].item()), float(solution[1].item())

    def _validate_values(
        self,
        values: torch.Tensor,
        description: str,
    ) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{description} must be a tensor")
        if values.device != self.device:
            raise ValueError(
                f"{description} must be on device {self.device}"
            )
        if values.dtype is not self.dtype:
            raise ValueError(f"{description} must use torch.float64")
        if values.ndim < 1 or values.shape[-1] != self.plan.size:
            raise ValueError(
                f"{description} final dimension must equal the plan size"
            )
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{description} must be finite")

    def to_modal(self, homogeneous_values: torch.Tensor) -> torch.Tensor:
        """Transform sampled homogeneous-remainder values to coefficients."""

        self._validate_values(homogeneous_values, "homogeneous values")
        self._operation_counts["forward_transform"] += 1
        return torch.matmul(
            homogeneous_values,
            self.inverse_basis_matrix.transpose(0, 1),
        )

    def from_modal(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Transform Robin modal coefficients to cell-centered values."""

        self._validate_values(coefficients, "Robin modal coefficients")
        self._operation_counts["inverse_transform"] += 1
        return torch.matmul(
            coefficients,
            self.basis_matrix.transpose(0, 1),
        )

    def homogeneous_remainder(self, physical_values: torch.Tensor) -> torch.Tensor:
        self._validate_values(physical_values, "physical values")
        return physical_values - self.lift_values

    def reconstruct_physical(
        self,
        homogeneous_values: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_values(homogeneous_values, "homogeneous values")
        return homogeneous_values + self.lift_values

    def apply_helmholtz(
        self,
        physical_values: torch.Tensor,
        *,
        mass: float,
    ) -> torch.Tensor:
        """Apply ``mass - d2/dz2`` through the Robin modal oracle."""

        normalized_mass = self._validate_mass(mass)
        self._operation_counts["helmholtz_apply"] += 1
        coefficients = self.to_modal(
            self.homogeneous_remainder(physical_values)
        )
        homogeneous = self.from_modal(
            coefficients * (normalized_mass + self.eigenvalues)
        )
        return homogeneous + normalized_mass * self.lift_values

    def solve_helmholtz(
        self,
        forcing: torch.Tensor,
        *,
        mass: float,
    ) -> torch.Tensor:
        """Solve ``(mass - d2/dz2) phi = forcing`` with the planned laws."""

        self._validate_values(forcing, "Helmholtz forcing")
        normalized_mass = self._validate_mass(mass)
        self._operation_counts["helmholtz_solve"] += 1
        denominator = normalized_mass + self.eigenvalues
        if bool((denominator == 0.0).any().item()):
            raise ValueError(
                "Robin Helmholtz operator has an unresolved Neumann null mode"
            )
        remainder_forcing = forcing - normalized_mass * self.lift_values
        coefficients = self.to_modal(remainder_forcing) / denominator
        return self.reconstruct_physical(self.from_modal(coefficients))

    @staticmethod
    def _validate_mass(mass: float) -> float:
        if (
            not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not math.isfinite(float(mass))
            or float(mass) < 0.0
        ):
            raise ValueError("Helmholtz mass must be finite and non-negative")
        return float(mass)

    def boundary_residual(
        self,
        physical_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower and upper outward-normal Robin residuals."""

        coefficients = self.to_modal(
            self.homogeneous_remainder(physical_values)
        )
        lower_value = self.lift_intercept + torch.sum(
            coefficients * self._lower_basis_values,
            dim=-1,
        )
        lower_derivative = self.lift_slope + torch.sum(
            coefficients * self._lower_basis_derivatives,
            dim=-1,
        )
        upper_value = (
            self.lift_intercept
            + self.lift_slope * self.plan.length
            + torch.sum(
                coefficients * self._upper_basis_values,
                dim=-1,
            )
        )
        upper_derivative = self.lift_slope + torch.sum(
            coefficients * self._upper_basis_derivatives,
            dim=-1,
        )
        lower = self.plan.lower
        upper = self.plan.upper
        lower_residual = (
            lower.alpha.value * lower_value
            - lower.beta.value * lower_derivative
            - lower.gamma.value
        )
        upper_residual = (
            upper.alpha.value * upper_value
            + upper.beta.value * upper_derivative
            - upper.gamma.value
        )
        return lower_residual, upper_residual

    def reset_operation_counts(self) -> None:
        for name in self._operation_counts:
            self._operation_counts[name] = 0

    def operation_counts(self) -> dict[str, int]:
        return dict(self._operation_counts)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operator_kind": self.plan.operator_kind,
            "plan_sha256": self.plan.canonical_sha256(),
            "materialized_sha256": self.materialized_sha256,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "condition_number": self.condition_number,
            "lift": {
                "kind": "affine_wall_normal",
                "intercept": self.lift_intercept,
                "slope": self.lift_slope,
            },
            "bounded_transform_complexity": "dense_O_N_squared",
            "construction_only_work": [
                "root_solve",
                "basis_materialization",
                "basis_inverse",
                "affine_lift_solve",
            ],
            "timestep_root_solve": False,
            "timestep_matrix_factorization": False,
            "operation_counts": self.operation_counts(),
            "runtime_connected": False,
        }


def materialize_cell_centered_robin_eigenbasis(
    plan: CellCenteredRobinEigenbasisPlan,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> CellCenteredRobinEigenbasisOperator:
    return CellCenteredRobinEigenbasisOperator(
        plan,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "CellCenteredRobinEigenbasisOperator",
    "materialize_cell_centered_robin_eigenbasis",
]
