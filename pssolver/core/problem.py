"""Top-level immutable problem specification and consistency checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json

from .boundary import BoundaryKind
from .fields import FieldSpec
from .geometry import AxisTopology, GeometrySpec
from .model import ModelProtocol
from .numerics import NumericsConfig, SpectralStorage


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Combine independent model, geometry, and numerical specifications."""

    model: ModelProtocol
    geometry: GeometrySpec
    numerics: NumericsConfig
    _field_specs: tuple[FieldSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _model_name: str = field(init=False, repr=False, compare=False)
    _parameter_metadata_json: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelProtocol):
            raise TypeError("model must implement ModelProtocol")
        if not isinstance(self.geometry, GeometrySpec):
            raise TypeError("geometry must be a GeometrySpec")
        if not isinstance(self.numerics, NumericsConfig):
            raise TypeError("numerics must be a NumericsConfig")
        if not isinstance(self.model.name, str) or not self.model.name:
            raise ValueError("model name must be a non-empty string")
        model_name = self.model.name

        parameter_metadata = self.model.parameter_metadata()
        if not isinstance(parameter_metadata, Mapping):
            raise TypeError("model.parameter_metadata() must return a mapping")
        try:
            parameter_metadata_json = json.dumps(
                dict(parameter_metadata),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model parameter metadata must be finite and JSON-compatible"
            ) from exc

        try:
            field_specs = tuple(self.model.field_specs())
        except TypeError as exc:
            raise TypeError("model.field_specs() must return an iterable") from exc
        if not field_specs:
            raise ValueError("a model must declare at least one field")
        if not all(isinstance(spec, FieldSpec) for spec in field_specs):
            raise TypeError("model fields must all be FieldSpec instances")

        field_names = tuple(spec.name for spec in field_specs)
        if len(set(field_names)) != len(field_names):
            raise ValueError("logical field names must be unique")
        component_names = tuple(
            component.name
            for spec in field_specs
            for component in spec.components
        )
        if len(set(component_names)) != len(component_names):
            raise ValueError("component names must be globally unique")

        for spec in field_specs:
            if spec.ndim != self.geometry.domain.ndim:
                raise ValueError(
                    f"field '{spec.name}' dimension does not match geometry"
                )
            self._validate_boundaries(spec)

        hermitian_axis = self.numerics.hermitian_axis
        if (
            self.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
            and hermitian_axis not in self.geometry.periodic_axes
        ):
            raise ValueError("Hermitian storage axis must be periodic")

        object.__setattr__(self, "_field_specs", field_specs)
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(
            self,
            "_parameter_metadata_json",
            parameter_metadata_json,
        )

    def _validate_boundaries(self, spec: FieldSpec) -> None:
        for component in spec.components:
            for axis, condition in enumerate(component.boundaries.axes):
                topology = self.geometry.axis_topologies[axis]
                is_periodic_bc = condition.kind is BoundaryKind.PERIODIC
                if topology is AxisTopology.PERIODIC and not is_periodic_bc:
                    raise ValueError(
                        f"component '{component.name}' requires a periodic "
                        f"condition on geometry axis {axis}"
                    )
                if topology is AxisTopology.BOUNDED and is_periodic_bc:
                    raise ValueError(
                        f"component '{component.name}' cannot be periodic on "
                        f"bounded geometry axis {axis}"
                    )

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        """Validated, deterministic field declarations."""

        return self._field_specs

    def to_metadata(self) -> dict[str, object]:
        """Return the auditable, JSON-compatible problem description."""

        return {
            "model": {
                "name": self._model_name,
                "parameters": json.loads(self._parameter_metadata_json),
                "fields": [spec.to_metadata() for spec in self.field_specs],
            },
            "geometry": self.geometry.to_metadata(),
            "numerics": self.numerics.to_metadata(),
        }
