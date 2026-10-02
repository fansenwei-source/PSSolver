"""Model-neutral materialization of static Plane lifting plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

import torch

from pssolver.planning.lifting import StaticLiftingPlan


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


def _require_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("static lifting dtype must be float32 or float64")
    return dtype


def _require_digest(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MaterializedLinearLiftCorrection:
    """One explicitly materialized, construction-time ``L(phi_lift)``."""

    component: str
    operator_name: str
    values: torch.Tensor = field(repr=False, compare=False)
    source_lift_sha256: str
    values_sha256: str
    storage_layout: str = "full_domain"

    def __post_init__(self) -> None:
        for value, description in (
            (self.component, "linear-correction component"),
            (self.operator_name, "linear operator name"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("linear correction values must be a tensor")
        if not bool(torch.isfinite(self.values).all().item()):
            raise ValueError("linear correction values must be finite")
        _require_digest(self.source_lift_sha256, "source lift identity")
        _require_digest(self.values_sha256, "linear correction identity")
        if self.storage_layout not in ("full_domain", "broadcast_zero"):
            raise ValueError("unsupported linear correction storage layout")
        if _tensor_sha256(self.values) != self.values_sha256:
            raise ValueError("linear correction identity does not match its values")

    def to_metadata(self) -> dict[str, object]:
        return {
            "component": self.component,
            "operator_name": self.operator_name,
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "device": str(self.values.device),
            "source_lift_sha256": self.source_lift_sha256,
            "values_sha256": self.values_sha256,
        }


class PlaneStaticLiftingOperator:
    """One-time materialization and field conversion for an affine Plane lift."""

    __slots__ = (
        "plan",
        "dtype",
        "device",
        "_component_indices",
        "_lift_profiles",
        "_affine_laplacian_profiles",
        "_materialized_sha256",
    )

    def __init__(
        self,
        plan: StaticLiftingPlan,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if not isinstance(plan, StaticLiftingPlan):
            raise TypeError("plan must be a StaticLiftingPlan")
        self.plan = plan
        self.dtype = _require_real_dtype(dtype)
        # ``torch.device("cuda")`` is an unindexed allocation request, not
        # the concrete identity reported by allocated tensors (for example
        # ``cuda:0``).  Use it for construction, then bind validation and
        # provenance to the device that actually owns the materialized lift.
        self.device = torch.device(device)
        axis = plan.wall_normal_axis
        count = plan.domain_shape[axis]
        fraction = (
            torch.arange(count, dtype=dtype, device=self.device) + 0.5
        ) / count
        broadcast_shape = [1] * len(plan.domain_shape)
        broadcast_shape[axis] = count
        fraction = fraction.reshape(broadcast_shape)

        profiles = []
        for component in plan.components:
            lower = component.lower_value.value
            upper = component.upper_value.value
            affine = lower + (upper - lower) * fraction
            profiles.append(affine)
        # Constant face data produce an affine field that varies only along
        # the bounded axis.  Retain that one-dimensional profile and expose
        # zero-stride full-domain views instead of owning one dense 3-D tensor
        # per component.  This preserves the logical field exactly while
        # avoiding duplicate construction-owned domain storage.
        self._lift_profiles = torch.stack(profiles, dim=0).contiguous()
        self.device = self._lift_profiles.device
        self._affine_laplacian_profiles = torch.zeros_like(
            self._lift_profiles
        )
        self._component_indices = MappingProxyType(
            {
                component: index
                for index, component in enumerate(plan.component_order)
            }
        )
        self._materialized_sha256 = _tensor_sha256(self.stacked_lift)

    @property
    def component_order(self) -> tuple[str, ...]:
        return self.plan.component_order

    @property
    def materialized_sha256(self) -> str:
        return self._materialized_sha256

    def _index(self, component: str) -> int:
        try:
            return self._component_indices[component]
        except KeyError as exc:
            raise KeyError(f"component '{component}' has no static lift") from exc

    def _validate_field_tensor(
        self,
        value: torch.Tensor,
        description: str,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} must be a tensor")
        if tuple(value.shape) != self.plan.domain_shape:
            raise ValueError(f"{description} shape does not match the lifting plan")
        if value.dtype is not self.dtype:
            raise ValueError(f"{description} dtype does not match the lifting plan")
        if value.device != self.device:
            raise ValueError(f"{description} device does not match the lifting plan")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} must be finite")

    def _validate_output_workspace(
        self,
        value: torch.Tensor,
        description: str,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} must be a tensor")
        if tuple(value.shape) != self.plan.domain_shape:
            raise ValueError(f"{description} shape does not match the lifting plan")
        if value.dtype is not self.dtype:
            raise ValueError(f"{description} dtype does not match the lifting plan")
        if value.device != self.device:
            raise ValueError(f"{description} device does not match the lifting plan")

    def lift(self, component: str) -> torch.Tensor:
        """Return the construction-owned lift view for one component."""

        return self._lift_profiles[self._index(component)].expand(
            self.plan.domain_shape
        )

    @property
    def stacked_lift(self) -> torch.Tensor:
        """Return the construction-owned component-first lift tensor."""

        return self._lift_profiles.expand(
            len(self.component_order),
            *self.plan.domain_shape,
        )

    def affine_laplacian(self, component: str) -> torch.Tensor:
        """Return the explicit zero Laplacian of the affine extension."""

        return self._affine_laplacian_profiles[
            self._index(component)
        ].expand(self.plan.domain_shape)

    def component_lift_sha256(self, component: str) -> str:
        """Return the dtype-aware content identity of one component lift."""

        return _tensor_sha256(self.lift(component))

    def reconstruct_physical(
        self,
        component: str,
        homogeneous_remainder: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``phi_homogeneous + phi_lift``."""

        self._validate_field_tensor(homogeneous_remainder, "homogeneous remainder")
        if out is None:
            return homogeneous_remainder + self.lift(component)
        self._validate_output_workspace(out, "physical output workspace")
        if out.data_ptr() == homogeneous_remainder.data_ptr():
            raise ValueError("physical output must not alias the evolved remainder")
        return torch.add(homogeneous_remainder, self.lift(component), out=out)

    def extract_homogeneous_remainder(
        self,
        component: str,
        physical_field: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``phi_physical - phi_lift``."""

        self._validate_field_tensor(physical_field, "physical field")
        if out is None:
            return physical_field - self.lift(component)
        self._validate_output_workspace(out, "remainder output workspace")
        if out.data_ptr() == physical_field.data_ptr():
            raise ValueError("remainder output must not alias the physical field")
        return torch.sub(physical_field, self.lift(component), out=out)

    def extract_boundary_compatible_initial_remainder(
        self,
        component: str,
        physical_field: torch.Tensor,
    ) -> torch.Tensor:
        """Return a smooth homogeneous remainder for fresh initialization.

        A generic generated field need not satisfy the prescribed wall values.
        Subtracting the lift alone would leave an order-one jump at the walls;
        truncating that jump in the DST basis produces a Gibbs layer.  Taper the
        construction-time remainder to zero at both physical faces before its
        first transform.  The evolved representation and its spectral method
        remain unchanged.
        """

        remainder = self.extract_homogeneous_remainder(
            component,
            physical_field,
        )
        axis = self.plan.wall_normal_axis
        count = self.plan.domain_shape[axis]
        fraction = (
            torch.arange(count, dtype=self.dtype, device=self.device) + 0.5
        ) / count
        envelope = torch.sin(torch.pi * fraction)
        shape = [1] * len(self.plan.domain_shape)
        shape[axis] = count
        return remainder * envelope.reshape(shape)

    def materialize_linear_correction(
        self,
        component: str,
        *,
        operator_name: str,
        linear_operator: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaterializedLinearLiftCorrection:
        """Apply a model-supplied linear operator once to an isolated lift copy."""

        if not isinstance(operator_name, str) or not operator_name.isidentifier():
            raise ValueError("linear operator name must be a Python identifier")
        if not callable(linear_operator):
            raise TypeError("linear_operator must be callable")
        source = self.lift(component)
        source_sha256 = _tensor_sha256(source)
        values = linear_operator(source.clone())
        self._validate_field_tensor(values, "linear lift correction")
        if _tensor_sha256(source) != source_sha256:
            raise RuntimeError("construction-owned lift changed during correction")
        values = values.detach().clone().contiguous()
        return MaterializedLinearLiftCorrection(
            component=component,
            operator_name=operator_name,
            values=values,
            source_lift_sha256=source_sha256,
            values_sha256=_tensor_sha256(values),
            storage_layout="full_domain",
        )

    def materialize_zero_linear_correction(
        self,
        component: str,
        *,
        operator_name: str,
        linear_operator: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaterializedLinearLiftCorrection:
        """Represent a model-declared zero correction without dense storage."""

        if not isinstance(operator_name, str) or not operator_name.isidentifier():
            raise ValueError("linear operator name must be a Python identifier")
        if not callable(linear_operator):
            raise TypeError("linear_operator must be callable")
        index = self._index(component)
        source = self.lift(component)
        source_sha256 = _tensor_sha256(source)
        compact = linear_operator(self._lift_profiles[index].clone())
        if not isinstance(compact, torch.Tensor):
            raise TypeError("zero linear correction must be a tensor")
        if compact.shape != self._lift_profiles[index].shape:
            raise ValueError("zero linear correction profile shape mismatch")
        if compact.dtype is not self.dtype or compact.device != self.device:
            raise ValueError("zero linear correction profile identity mismatch")
        if not bool(torch.isfinite(compact).all().item()):
            raise ValueError("zero linear correction must be finite")
        if bool(torch.count_nonzero(compact).item()):
            raise ValueError("declared zero linear correction is nonzero")
        if _tensor_sha256(source) != source_sha256:
            raise RuntimeError("construction-owned lift changed during correction")
        values = compact.detach().clone().contiguous().expand(
            self.plan.domain_shape
        )
        return MaterializedLinearLiftCorrection(
            component=component,
            operator_name=operator_name,
            values=values,
            source_lift_sha256=source_sha256,
            values_sha256=_tensor_sha256(values),
            storage_layout="broadcast_zero",
        )

    def verify_materialized_identity(self) -> None:
        """Fail if construction-owned lift storage has been mutated."""

        if _tensor_sha256(self.stacked_lift) != self._materialized_sha256:
            raise RuntimeError("materialized static lift identity mismatch")

    def storage_metadata(self) -> dict[str, object]:
        """Describe physical storage separately from logical field extent."""

        logical_numel = len(self.component_order)
        for size in self.plan.domain_shape:
            logical_numel *= size
        return {
            "layout": "wall_normal_profile_broadcast",
            "logical_lift_numel": logical_numel,
            "lift_profile_storage_bytes": (
                self._lift_profiles.untyped_storage().nbytes()
            ),
            "affine_laplacian_storage_bytes": (
                self._affine_laplacian_profiles.untyped_storage().nbytes()
            ),
        }

    def to_metadata(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan.canonical_sha256(),
            "component_order": list(self.component_order),
            "shape": list(self.plan.domain_shape),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "materialized_lift_sha256": self.materialized_sha256,
            "affine_laplacian_explicit": True,
            "allocation_lifetime": "construction_owned_static",
            "fresh_initial_remainder_conditioning": {
                "kind": "sine_wall_taper",
                "wall_values": [0.0, 0.0],
                "applied_before_first_transform": True,
            },
        }


def materialize_plane_static_lifting(
    plan: StaticLiftingPlan,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> PlaneStaticLiftingOperator:
    """Materialize a validated lifting plan exactly once."""

    return PlaneStaticLiftingOperator(plan, dtype=dtype, device=device)


__all__ = [
    "MaterializedLinearLiftCorrection",
    "PlaneStaticLiftingOperator",
    "materialize_plane_static_lifting",
]
