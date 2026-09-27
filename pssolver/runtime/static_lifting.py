"""Runtime state for field-neutral, static Plane lifting.

The evolved :class:`~pssolver.Field.Fields` object remains the homogeneous
remainder authority.  This module owns immutable lift data and reconstructs
physical fields on demand; it does not know about Q tensors or any model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from types import MappingProxyType

import torch

from pssolver.operators.lifting import (
    PlaneStaticLiftingOperator,
    materialize_plane_static_lifting,
)
from pssolver.planning.lifting import StaticLiftingPlan


def _frozen_json(value: Mapping[str, object] | None) -> Mapping[str, object]:
    payload = json.dumps(
        {} if value is None else dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return MappingProxyType(json.loads(payload))


class PlaneStaticLiftingRuntime:
    """Materialized static lift with allocation-free hot-path metadata."""

    def __init__(
        self,
        plan: StaticLiftingPlan,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        batch_size: int,
        convention: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.operator = materialize_plane_static_lifting(
            plan,
            dtype=dtype,
            device=device,
        )
        self.component_order = self.operator.component_order
        self._indices = MappingProxyType(
            {name: index for index, name in enumerate(self.component_order)}
        )
        wall_axis = plan.wall_normal_axis
        length = plan.domain_lengths[wall_axis]
        slopes = []
        for name in self.component_order:
            component = plan.for_component(name)
            slopes.append(
                (component.upper_value.value - component.lower_value.value)
                / length
            )
        self._wall_normal_slopes = torch.tensor(
            slopes,
            dtype=dtype,
            device=device,
        )
        self._batch_size = batch_size
        self._linear_correction_hat: torch.Tensor | None = None
        self._linear_correction_metadata: tuple[dict[str, object], ...] = ()
        self._linear_correction_storage_metadata: tuple[
            dict[str, object], ...
        ] = ()
        self._convention = _frozen_json(convention)

    @property
    def plan(self) -> StaticLiftingPlan:
        return self.operator.plan

    def _index(self, component: str) -> int:
        try:
            return self._indices[component]
        except KeyError as exc:
            raise KeyError(f"component {component!r} has no static lift") from exc

    def extract_initial_remainders(
        self,
        physical_values: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Convert physical construction values into evolved remainders."""

        if not isinstance(physical_values, Mapping):
            raise TypeError("physical_values must be a mapping")
        if set(physical_values) != set(self.component_order):
            raise ValueError("physical initial values must exactly cover the lift")
        remainders: dict[str, torch.Tensor] = {}
        for name in self.component_order:
            value = physical_values[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("physical initial values must be tensors")
            value = value.to(
                dtype=self.operator.dtype,
                device=self.operator.device,
            )
            remainders[name] = self.operator.extract_homogeneous_remainder(
                name,
                value,
            )
        return remainders

    def physical_component(self, fields: object, component: str) -> torch.Tensor:
        """Reconstruct one physical component for observation."""

        homogeneous = fields[component]
        if homogeneous.shape != (self._batch_size, *self.plan.domain_shape):
            raise ValueError("evolved remainder shape does not match lift plan")
        return homogeneous + self.operator.lift(component)

    def physical_components(
        self,
        fields: object,
        components: Sequence[str],
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct physical components outside the production hot path."""

        return tuple(
            self.physical_component(fields, name) for name in components
        )

    def lift_components(
        self,
        components: Sequence[str],
    ) -> tuple[torch.Tensor, ...]:
        """Return immutable broadcast lift views for fused pointwise kernels."""

        return tuple(self.operator.lift(name) for name in components)

    def gradient(
        self,
        fields: object,
        component: str,
        *,
        axis: int,
        projector: object,
    ) -> torch.Tensor:
        """Return the remainder derivative plus the affine-lift derivative."""

        value = fields.gradient(component, axis=axis, projector=projector)
        if axis == self.plan.wall_normal_axis:
            value.add_(self._wall_normal_slopes[self._index(component)])
        return value

    def laplacian(
        self,
        fields: object,
        component: str,
        *,
        projector: object,
    ) -> torch.Tensor:
        """Return the physical Laplacian (affine lift contribution is zero)."""

        return fields.laplacian(component, projector=projector)

    def materialize_linear_correction(
        self,
        *,
        projector: object,
        boundary_conditions: Sequence[str],
        operator_name: str,
        linear_operator: Callable[[torch.Tensor], torch.Tensor],
        identically_zero: bool = False,
    ) -> torch.Tensor:
        """Materialize and transform all ``L(phi_lift)`` terms once."""

        if not isinstance(identically_zero, bool):
            raise TypeError("identically_zero must be a bool")
        if not callable(linear_operator):
            raise TypeError("linear_operator must be callable")
        if identically_zero:
            corrections = tuple(
                self.operator.materialize_zero_linear_correction(
                    name,
                    operator_name=operator_name,
                    linear_operator=linear_operator,
                )
                for name in self.component_order
            )
            spectral_dtype = projector.transform_backend.spectral_dtype
            zero = torch.zeros(
                (),
                dtype=spectral_dtype,
                device=self.operator.device,
            )
            self._linear_correction_hat = zero.expand(
                len(self.component_order),
                1,
                *projector.shape,
            )
        else:
            corrections = tuple(
                self.operator.materialize_linear_correction(
                    name,
                    operator_name=operator_name,
                    linear_operator=linear_operator,
                )
                for name in self.component_order
            )
            values = torch.stack(tuple(item.values for item in corrections))
            transformed = projector.forward_transform(
                values,
                boundary_conditions,
            )
            self._linear_correction_hat = transformed[:, None].contiguous()
        self._linear_correction_metadata = tuple(
            item.to_metadata() for item in corrections
        )
        self._linear_correction_storage_metadata = tuple(
            {
                "component": item.component,
                "storage_layout": item.storage_layout,
                "storage_bytes": item.values.untyped_storage().nbytes(),
            }
            for item in corrections
        )
        return self._linear_correction_hat

    @property
    def linear_correction_hat(self) -> torch.Tensor:
        if self._linear_correction_hat is None:
            raise RuntimeError("linear lift correction has not been materialized")
        return self._linear_correction_hat

    def verify_identity(self) -> None:
        self.operator.verify_materialized_identity()

    def storage_metadata(self) -> dict[str, object]:
        """Report owned persistent storage used by the lifting runtime."""

        correction_bytes = (
            0
            if self._linear_correction_hat is None
            else self._linear_correction_hat.untyped_storage().nbytes()
        )
        return {
            "operator": self.operator.storage_metadata(),
            "physical_workspace_layout": "on_demand_fused_pointwise",
            "physical_workspace_bytes": 0,
            "wall_normal_slope_bytes": (
                self._wall_normal_slopes.untyped_storage().nbytes()
            ),
            "linear_correction_storage_bytes": correction_bytes,
            "linear_correction_layouts": [
                item["storage_layout"]
                for item in self._linear_correction_storage_metadata
            ],
            "linear_corrections": list(
                self._linear_correction_storage_metadata
            ),
        }

    def restart_metadata(self) -> dict[str, object]:
        self.verify_identity()
        return {
            "schema_version": 1,
            "representation": "homogeneous_remainder",
            "physical_reconstruction": "homogeneous_remainder_plus_lift",
            "lifting": self.operator.to_metadata(),
            "linear_corrections": list(self._linear_correction_metadata),
            "convention": dict(self._convention),
        }


__all__ = ["PlaneStaticLiftingRuntime"]
