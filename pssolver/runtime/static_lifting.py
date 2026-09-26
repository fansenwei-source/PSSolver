"""Runtime state for field-neutral, static Plane lifting.

The evolved :class:`~pssolver.Field.Fields` object remains the homogeneous
remainder authority.  This module owns immutable lift data and reusable
physical-field workspaces; it does not know about Q tensors or any model.
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
    """Materialized static lift and preallocated physical-field workspace."""

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
        # This is a view of the operator-owned, identity-checked storage.  It
        # avoids retaining a second full-domain copy of every static lift.
        self._stacked_lift = self.operator.stacked_lift[:, None]
        self._physical_workspace = torch.empty(
            len(self.component_order),
            batch_size,
            *plan.domain_shape,
            dtype=dtype,
            device=device,
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
        self._cached_spatial_version: int | None = None
        self._linear_correction_hat: torch.Tensor | None = None
        self._linear_correction_metadata: tuple[dict[str, object], ...] = ()
        self._convention = _frozen_json(convention)

    @property
    def plan(self) -> StaticLiftingPlan:
        return self.operator.plan

    @staticmethod
    def _tensor_version(value: torch.Tensor) -> int | None:
        try:
            return int(value._version)
        except (AttributeError, RuntimeError):
            return None

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

    def _refresh_physical_workspace(self, fields: object) -> None:
        spatial = fields.spatial
        version = self._tensor_version(spatial)
        if version is not None and version == self._cached_spatial_version:
            return
        homogeneous = fields.select_spatial_group(self.component_order)
        if homogeneous.shape != self._physical_workspace.shape:
            raise ValueError("evolved remainder shape does not match lift workspace")
        torch.add(
            homogeneous,
            self._stacked_lift,
            out=self._physical_workspace,
        )
        self._cached_spatial_version = version

    def physical_component(self, fields: object, component: str) -> torch.Tensor:
        """Return one non-owning physical-field workspace view."""

        self._refresh_physical_workspace(fields)
        return self._physical_workspace[self._index(component)]

    def physical_components(
        self,
        fields: object,
        components: Sequence[str],
    ) -> tuple[torch.Tensor, ...]:
        self._refresh_physical_workspace(fields)
        return tuple(
            self._physical_workspace[self._index(name)] for name in components
        )

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
    ) -> torch.Tensor:
        """Materialize and transform all ``L(phi_lift)`` terms once."""

        corrections = tuple(
            self.operator.materialize_linear_correction(
                name,
                operator_name=operator_name,
                linear_operator=linear_operator,
            )
            for name in self.component_order
        )
        values = torch.stack(tuple(item.values for item in corrections))
        transformed = projector.forward_transform(values, boundary_conditions)
        self._linear_correction_hat = transformed[:, None].contiguous()
        self._linear_correction_metadata = tuple(
            item.to_metadata() for item in corrections
        )
        return self._linear_correction_hat

    @property
    def linear_correction_hat(self) -> torch.Tensor:
        if self._linear_correction_hat is None:
            raise RuntimeError("linear lift correction has not been materialized")
        return self._linear_correction_hat

    def verify_identity(self) -> None:
        self.operator.verify_materialized_identity()

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
