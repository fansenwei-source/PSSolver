"""Read-only comparison between a SpectralPlan and the current runtime.

The comparison is deliberately outside both the core contracts and planning
layer.  It reads public runtime metadata but never installs the plan into a
solver or changes a field value.
"""

from __future__ import annotations

from dataclasses import dataclass

from pssolver.planning.plan import SpectralPlan

from .legacy_boundaries import boundary_set_to_legacy


@dataclass(frozen=True, slots=True)
class ShadowMismatch:
    """One deterministic difference between planned and runtime metadata."""

    path: str
    expected: object
    observed: object

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": self.path,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """Immutable result of a non-invasive runtime metadata comparison."""

    mismatches: tuple[ShadowMismatch, ...]
    checked_components: int
    checked_fields: bool
    checked_projector: bool

    def __post_init__(self) -> None:
        mismatches = tuple(self.mismatches)
        if not all(
            isinstance(mismatch, ShadowMismatch) for mismatch in mismatches
        ):
            raise TypeError("mismatches must contain ShadowMismatch objects")
        if (
            not isinstance(self.checked_components, int)
            or isinstance(self.checked_components, bool)
            or self.checked_components < 0
        ):
            raise ValueError("checked_components must be non-negative")
        if not isinstance(self.checked_fields, bool):
            raise TypeError("checked_fields must be a bool")
        if not isinstance(self.checked_projector, bool):
            raise TypeError("checked_projector must be a bool")
        object.__setattr__(self, "mismatches", mismatches)

    @property
    def matches(self) -> bool:
        return not self.mismatches

    def require_match(self) -> None:
        """Raise a concise error if any planned/runtime value differs."""

        if self.matches:
            return
        details = "; ".join(
            f"{item.path}: expected {item.expected!r}, "
            f"observed {item.observed!r}"
            for item in self.mismatches
        )
        raise RuntimeError(f"spectral-plan shadow mismatch: {details}")

    def to_metadata(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "checked_components": self.checked_components,
            "checked_fields": self.checked_fields,
            "checked_projector": self.checked_projector,
            "mismatches": [
                mismatch.to_metadata() for mismatch in self.mismatches
            ],
        }


def _require_attributes(value: object, description: str, names: tuple[str, ...]) -> None:
    missing = tuple(name for name in names if not hasattr(value, name))
    if missing:
        raise TypeError(
            f"{description} is missing required attributes {missing!r}"
        )


def _dtype_name(dtype: object) -> str:
    name = str(dtype)
    return name.split(".")[-1]


def compare_spectral_plan_to_runtime(
    plan: SpectralPlan,
    backend: object,
    *,
    fields: object | None = None,
    projector: object | None = None,
) -> ShadowComparison:
    """Compare a plan with current backend, field, and projector metadata.

    Calling ``backend.get_metadata`` or projector inspection methods may fill
    their internal metadata caches.  No solver fields, coefficients, model
    objects, numerical defaults, or execution routes are modified.
    """

    if not isinstance(plan, SpectralPlan):
        raise TypeError("plan must be a SpectralPlan")
    _require_attributes(
        backend,
        "backend",
        (
            "shape",
            "lengths",
            "spectral_shape",
            "execution_order",
            "spectral_storage",
            "hermitian_axis",
            "real_dtype",
            "get_metadata",
        ),
    )
    if fields is not None:
        _require_attributes(
            fields,
            "fields",
            (
                "name_to_idx",
                "boundary_conditions",
                "dyn_count",
                "stat_count",
                "spectral_shape",
            ),
        )
    if projector is not None:
        _require_attributes(
            projector,
            "projector",
            (
                "rule",
                "transform_execution",
                "shape",
                "retained_axis_counts",
                "computed_axis_sizes",
            ),
        )

    mismatches = []

    def compare(path: str, expected: object, observed: object) -> None:
        if expected != observed:
            mismatches.append(ShadowMismatch(path, expected, observed))

    compare("backend.shape", plan.physical_shape, tuple(backend.shape))
    compare("backend.lengths", plan.lengths, tuple(backend.lengths))
    compare(
        "backend.spectral_shape",
        plan.spectral_shape,
        tuple(backend.spectral_shape),
    )
    compare(
        "backend.execution_order",
        plan.numerics.transform_execution_order.value,
        backend.execution_order,
    )
    compare(
        "backend.spectral_storage",
        plan.numerics.spectral_storage.value,
        backend.spectral_storage,
    )
    compare(
        "backend.hermitian_axis",
        plan.numerics.hermitian_axis,
        backend.hermitian_axis,
    )
    compare(
        "backend.real_dtype",
        plan.numerics.precision.value,
        _dtype_name(backend.real_dtype),
    )

    if projector is not None:
        compare(
            "projector.rule",
            plan.numerics.dealias_rule.value,
            projector.rule,
        )
        compare(
            "projector.transform_execution",
            plan.numerics.projected_transform_execution.value,
            projector.transform_execution,
        )
        compare(
            "projector.shape",
            plan.spectral_shape,
            tuple(projector.shape),
        )

    for component in plan.stored_components:
        prefix = f"components.{component.component_name}"
        boundary_conditions = boundary_set_to_legacy(component.boundaries)
        try:
            metadata = backend.get_metadata(boundary_conditions)
        except (TypeError, ValueError, RuntimeError) as exc:
            mismatches.append(
                ShadowMismatch(
                    f"{prefix}.runtime_metadata",
                    "available",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        compare(
            f"{prefix}.boundary_conditions",
            boundary_conditions,
            tuple(metadata.boundary_conditions),
        )
        compare(
            f"{prefix}.transform_kinds",
            tuple(kind.value for kind in component.transform_kinds),
            tuple(metadata.transform_kinds),
        )
        compare(
            f"{prefix}.axis_mode_counts",
            plan.spectral_shape,
            tuple(int(modes.numel()) for modes in metadata.axis_modes),
        )
        compare(
            f"{prefix}.q2_shape",
            plan.spectral_shape,
            tuple(metadata.q2.shape),
        )
        compare(
            f"{prefix}.laplacian_shape",
            plan.spectral_shape,
            tuple(metadata.laplacian_eigs.shape),
        )
        if projector is not None:
            try:
                retained_counts = tuple(
                    projector.retained_axis_counts(boundary_conditions)
                )
                computed_sizes = tuple(
                    projector.computed_axis_sizes(boundary_conditions)
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                mismatches.append(
                    ShadowMismatch(
                        f"{prefix}.projector_metadata",
                        "available",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                compare(
                    f"{prefix}.retained_mode_counts",
                    component.retained_mode_counts,
                    retained_counts,
                )
                compare(
                    f"{prefix}.computed_axis_sizes",
                    component.computed_axis_sizes,
                    computed_sizes,
                )

    if fields is not None:
        expected_names = tuple(
            component.component_name for component in plan.stored_components
        )
        try:
            observed_names = tuple(
                name
                for name, _ in sorted(
                    fields.name_to_idx.items(),
                    key=lambda item: item[1],
                )
            )
        except (AttributeError, TypeError) as exc:
            raise TypeError("fields.name_to_idx must be an index mapping") from exc
        compare("fields.component_names", expected_names, observed_names)
        compare(
            "fields.dynamic_count",
            plan.evolved_component_count,
            fields.dyn_count,
        )
        compare(
            "fields.static_count",
            plan.algebraic_component_count,
            fields.stat_count,
        )
        compare(
            "fields.spectral_shape",
            plan.spectral_shape,
            tuple(fields.spectral_shape),
        )
        for component in plan.stored_components:
            index = component.storage_index
            if index >= len(fields.boundary_conditions):
                mismatches.append(
                    ShadowMismatch(
                        f"fields.{component.component_name}.boundary_conditions",
                        boundary_set_to_legacy(component.boundaries),
                        "<missing>",
                    )
                )
                continue
            compare(
                f"fields.{component.component_name}.boundary_conditions",
                boundary_set_to_legacy(component.boundaries),
                tuple(fields.boundary_conditions[index]),
            )

    return ShadowComparison(
        mismatches=tuple(mismatches),
        checked_components=len(plan.stored_components),
        checked_fields=fields is not None,
        checked_projector=projector is not None,
    )
