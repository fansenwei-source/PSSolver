"""Tensor-free declarations for geometry-neutral equation systems.

The declarations in this module intentionally contain no boundary assignment,
geometry, transform, tensor, backend, runtime, or workflow object.  They state
which logical fields exist, which equation terms produce them, and which
algebraic capabilities are required.  A later composition/lowering stage must
attach physical boundaries and a geometry before selecting numerical
implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from pssolver.core.fields import FieldRole

from .algebraic import AlgebraicSystemSpec


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


def _identifiers(
    values: object,
    description: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{description} must be an iterable, not a string")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{description} must be iterable") from exc
    if not normalized and not allow_empty:
        raise ValueError(f"{description} must not be empty")
    for value in normalized:
        _identifier(value, description)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{description} must contain unique identifiers")
    return normalized


def _json_mapping(
    value: object,
    description: str,
) -> tuple[Mapping[str, object], str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if any(
        not isinstance(name, str) or not name.isidentifier()
        for name in value
    ):
        raise ValueError(f"{description} keys must be Python identifiers")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must be finite and JSON-compatible"
        ) from exc
    normalized = json.loads(encoded)
    return MappingProxyType(normalized), encoded


@dataclass(frozen=True, slots=True)
class EquationFieldSpec:
    """Boundary-free logical field owned by an equation declaration."""

    name: str
    role: FieldRole
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "field name"))
        if not isinstance(self.role, FieldRole):
            raise TypeError("role must be a FieldRole")
        object.__setattr__(
            self,
            "components",
            _identifiers(self.components, "field components"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "components": list(self.components),
        }


@dataclass(frozen=True, slots=True)
class EquationTermSpec:
    """One geometry-neutral evolution or constitutive dependency."""

    name: str
    capability: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)
    _parameters_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "term name"))
        object.__setattr__(
            self,
            "capability",
            _identifier(self.capability, "term capability"),
        )
        object.__setattr__(
            self,
            "output_components",
            _identifiers(self.output_components, "term outputs"),
        )
        object.__setattr__(
            self,
            "dependencies",
            _identifiers(self.dependencies, "term dependencies"),
        )
        parameters, encoded = _json_mapping(
            self.parameters,
            "term parameters",
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_parameters_json", encoded)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capability": self.capability,
            "output_components": list(self.output_components),
            "dependencies": list(self.dependencies),
            "parameters": json.loads(self._parameters_json),
        }


@dataclass(frozen=True, slots=True)
class EquationSystemSpec:
    """Complete boundary-free declaration of one coupled PDE system.

    This is not an executable model and deliberately does not implement
    ``ModelProtocol``.  Boundary assignment plus geometry-aware lowering must
    first convert it into field spaces, derivative maps, and concrete
    algebraic implementations.
    """

    name: str
    variant: str
    fields: tuple[EquationFieldSpec, ...]
    evolution_laws: tuple[EquationTermSpec, ...]
    constitutive_laws: tuple[EquationTermSpec, ...]
    algebraic_systems: tuple[AlgebraicSystemSpec, ...]
    parameters: Mapping[str, object]
    diagnostics: tuple[str, ...] = ()
    initial_condition_families: tuple[str, ...] = ()
    _parameters_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "system name"))
        object.__setattr__(
            self,
            "variant",
            _identifier(self.variant, "system variant"),
        )
        declarations = (
            ("fields", self.fields, EquationFieldSpec),
            ("evolution_laws", self.evolution_laws, EquationTermSpec),
            ("constitutive_laws", self.constitutive_laws, EquationTermSpec),
            ("algebraic_systems", self.algebraic_systems, AlgebraicSystemSpec),
        )
        normalized: dict[str, tuple[object, ...]] = {}
        for name, values, value_type in declarations:
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be an iterable, not a string")
            try:
                items = tuple(values)
            except TypeError as exc:
                raise TypeError(f"{name} must be iterable") from exc
            if not items:
                raise ValueError(f"{name} must not be empty")
            if not all(isinstance(item, value_type) for item in items):
                raise TypeError(
                    f"{name} must contain only {value_type.__name__} objects"
                )
            normalized[name] = items
            object.__setattr__(self, name, items)

        parameters, encoded = _json_mapping(
            self.parameters,
            "equation-system parameters",
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_parameters_json", encoded)
        object.__setattr__(
            self,
            "diagnostics",
            _identifiers(
                self.diagnostics,
                "diagnostics",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "initial_condition_families",
            _identifiers(
                self.initial_condition_families,
                "initial-condition families",
                allow_empty=True,
            ),
        )
        self._validate_dependency_graph()

    def _validate_dependency_graph(self) -> None:
        fields = self.fields
        field_names = tuple(item.name for item in fields)
        if len(set(field_names)) != len(field_names):
            raise ValueError("logical field names must be unique")
        component_roles: dict[str, FieldRole] = {}
        for item in fields:
            for component in item.components:
                if component in component_roles:
                    raise ValueError(
                        "field component names must be globally unique"
                    )
                component_roles[component] = item.role

        term_names = [
            term.name
            for term in (*self.evolution_laws, *self.constitutive_laws)
        ]
        term_names.extend(system.name for system in self.algebraic_systems)
        if len(set(term_names)) != len(term_names):
            raise ValueError(
                "evolution, constitutive, and algebraic names must be unique"
            )

        known = set(component_roles)
        produced: dict[str, str] = {}

        def validate_term(
            term: EquationTermSpec,
            expected_role: FieldRole,
        ) -> None:
            unknown = (set(term.output_components) | set(term.dependencies)) - known
            if unknown:
                raise ValueError(
                    f"term '{term.name}' references unknown components "
                    f"{tuple(sorted(unknown))!r}"
                )
            for component in term.output_components:
                if component_roles[component] is not expected_role:
                    raise ValueError(
                        f"term '{term.name}' output '{component}' has the "
                        f"wrong field role"
                    )
                if component in produced:
                    raise ValueError(
                        f"component '{component}' has multiple producers"
                    )
                produced[component] = term.name

        for term in self.evolution_laws:
            validate_term(term, FieldRole.EVOLVED)
        for term in self.constitutive_laws:
            validate_term(term, FieldRole.TRANSIENT)
        for system in self.algebraic_systems:
            unknown = (
                set(system.output_components) | set(system.dependencies)
            ) - known
            if unknown:
                raise ValueError(
                    f"algebraic system '{system.name}' references unknown "
                    f"components {tuple(sorted(unknown))!r}"
                )
            for component in system.output_components:
                if component_roles[component] is not FieldRole.ALGEBRAIC:
                    raise ValueError(
                        f"algebraic output '{component}' has the wrong "
                        "field role"
                    )
                if component in produced:
                    raise ValueError(
                        f"component '{component}' has multiple producers"
                    )
                produced[component] = system.name

        required = {
            component
            for component, role in component_roles.items()
            if role is not FieldRole.DIAGNOSTIC
        }
        missing = required - set(produced)
        if missing:
            raise ValueError(
                "non-diagnostic components must have exactly one producer; "
                f"missing={tuple(sorted(missing))!r}"
            )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(
            component
            for field_spec in self.fields
            for component in field_spec.components
        )

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        capabilities = (
            *(term.capability for term in self.evolution_laws),
            *(term.capability for term in self.constitutive_laws),
            *(system.capability for system in self.algebraic_systems),
        )
        return tuple(dict.fromkeys(capabilities))

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "variant": self.variant,
            "fields": [item.to_metadata() for item in self.fields],
            "evolution_laws": [
                item.to_metadata() for item in self.evolution_laws
            ],
            "constitutive_laws": [
                item.to_metadata() for item in self.constitutive_laws
            ],
            "algebraic_systems": [
                item.to_metadata() for item in self.algebraic_systems
            ],
            "parameters": json.loads(self._parameters_json),
            "diagnostics": list(self.diagnostics),
            "initial_condition_families": list(
                self.initial_condition_families
            ),
            "required_capabilities": list(self.required_capabilities),
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "EquationFieldSpec",
    "EquationSystemSpec",
    "EquationTermSpec",
]
