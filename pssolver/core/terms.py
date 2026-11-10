"""Tensor-free model declarations and symbolic, component-uniform terms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar


class _Arithmetic:
    def __add__(self, other):
        return Expr("add", (self, other))

    def __radd__(self, other):
        return Expr("add", (other, self))

    def __sub__(self, other):
        return Expr("add", (self, -other))

    def __rsub__(self, other):
        return Expr("add", (other, -self))

    def __neg__(self):
        return Expr("multiply", (-1.0, self))

    def __mul__(self, other):
        return Expr("multiply", (self, other))

    def __rmul__(self, other):
        return Expr("multiply", (other, self))


@dataclass(frozen=True, slots=True)
class Expr(_Arithmetic):
    """An unevaluated expression; the planning layer owns validation."""

    operation: str
    arguments: tuple


@dataclass(frozen=True, slots=True)
class FieldRef(_Arithmetic):
    """A whole field, or an explicitly selected component."""

    name: str
    components: tuple[str, ...]

    def __getitem__(self, index):
        return FieldRef(self.name, (self.components[index],))


def lap(value):
    """Declare the sum of the three coordinate second derivatives."""
    return Expr("laplacian", (value,))


def d(value, axis, order=1):
    """Declare a coordinate derivative without evaluating a tensor."""
    return Expr("derivative", (value, axis, order))


def grad(value, axis):
    """Declare one coordinate derivative."""
    return d(value, axis)


def div(components):
    """Declare divergence of exactly three expressions."""
    if len(components) != 3:
        raise ValueError("div requires three components")
    return d(components[0], 0) + d(components[1], 1) + d(components[2], 2)


class EvolvedField:
    """Declare an evolved field, preserving class declaration order."""

    role = "evolved"

    def __init__(self, components=1):
        if isinstance(components, int):
            if isinstance(components, bool) or components < 1:
                raise ValueError("field components must be positive")
        elif (
            not isinstance(components, tuple)
            or not components
            or not all(isinstance(name, str) and name for name in components)
        ):
            raise TypeError("components must be a positive integer or a tuple of names")
        self.components = components
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        return self

    def reference(self, name):
        components = self.components
        if isinstance(components, int):
            components = (
                (name,)
                if components == 1
                else tuple(f"{name}{index}" for index in range(components))
            )
        return FieldRef(name, components)


class ConstrainedField(EvolvedField):
    """Declare a field supplied by a constraint solver."""

    role = "constrained"


class Parameter:
    """Declare a scalar or field parameter and its control boundary."""

    def __init__(self, default=None, *, kind="scalar", controllable=False):
        if kind not in {"scalar", "field"}:
            raise ValueError("parameter kind must be scalar or field")
        self.default = None if default is None else float(default)
        self.kind = kind
        self.controllable = bool(controllable)
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        return self if instance is None else instance.parameters[self.name]


class Model:
    """Ordered model declarations; subclasses supply tensor-free term builders."""

    version: ClassVar[str] = "1"

    def __init__(self, **values):
        declarations = {}
        for base in reversed(type(self).__mro__):
            declarations.update(
                (name, value)
                for name, value in vars(base).items()
                if isinstance(value, (EvolvedField, Parameter))
            )
        fields = {
            name: value for name, value in declarations.items() if isinstance(value, EvolvedField)
        }
        parameters = {
            name: value for name, value in declarations.items() if isinstance(value, Parameter)
        }
        unknown = values.keys() - parameters.keys()
        if unknown:
            raise TypeError(f"unknown model parameters: {sorted(unknown)}")
        for name, value in values.items():
            original = parameters[name]
            parameters[name] = (
                value
                if isinstance(value, Parameter)
                else Parameter(value, kind=original.kind, controllable=original.controllable)
            )
        self.fields = MappingProxyType(fields)
        self.parameters = MappingProxyType(parameters)

    def namespaces(self):
        """Return symbolic field references and fixed scalar parameter values."""
        return (
            SimpleNamespace(**{name: value.reference(name) for name, value in self.fields.items()}),
            SimpleNamespace(
                **{
                    name: Expr("parameter", (name,))
                    if value.kind == "field" or value.controllable or value.default is None
                    else value.default
                    for name, value in self.parameters.items()
                }
            ),
        )

    def terms(self, f, p):
        raise NotImplementedError


class Recipe:
    """Reference a package-owned diagonal recipe."""

    name: str
    arguments: Mapping[str, float]

    def __init__(self, name, **arguments):
        self.name = name
        self.arguments = MappingProxyType(dict(arguments))


@dataclass(frozen=True, slots=True)
class PackageTerm:
    """Reference a package-owned explicit implementation."""

    name: str


@dataclass(frozen=True, slots=True)
class Coordinates:
    """Request the physical coordinate tensor."""


@dataclass(frozen=True, slots=True)
class Time:
    """Request the step's Python-valued physical time."""


@dataclass(frozen=True, slots=True)
class Linear:
    """Declare exactly one expression or a package diagonal recipe."""

    target: FieldRef
    expr: Expr | FieldRef | None = None
    diagonal: Recipe | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if (self.expr is None) == (self.diagonal is None):
            raise ValueError("Linear requires exactly one of expr and diagonal")


@dataclass(frozen=True, slots=True)
class Explicit:
    """Declare an explicit package term or a model-local physical kernel."""

    target: FieldRef
    kernel: Callable | PackageTerm
    inputs: tuple | None = field(default=None, kw_only=True)
    arguments: Mapping[str, object] = field(default_factory=dict, kw_only=True)
    execution_options: tuple[str, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True, slots=True)
class Constraint:
    """Declare fields supplied by one registered constraint solver."""

    kind: str
    inputs: tuple[FieldRef, ...] = field(kw_only=True)
    outputs: tuple[FieldRef, ...] = field(kw_only=True)
    arguments: Mapping[str, object] = field(default_factory=dict, kw_only=True)
    execution_options: tuple[str, ...] = field(default=(), kw_only=True)
