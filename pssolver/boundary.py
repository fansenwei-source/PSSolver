from enum import Enum


class BoundaryCondition(str, Enum):
    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


def _coerce_bc(value):
    if isinstance(value, BoundaryCondition):
        return value
    if isinstance(value, str):
        try:
            return BoundaryCondition(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown boundary condition: {value!r}") from exc
    raise TypeError(f"Boundary condition must be a string or BoundaryCondition, got {type(value)}")


def normalize_bcs(bcs, dim):
    if dim < 1 or dim > 3:
        raise ValueError(f"Only 1D-3D supported, got dim={dim}")

    if bcs is None:
        return tuple([BoundaryCondition.PERIODIC] * dim)

    if isinstance(bcs, (str, BoundaryCondition)):
        return tuple([_coerce_bc(bcs)] * dim)

    if isinstance(bcs, (list, tuple)):
        if len(bcs) != dim:
            raise ValueError(f"Expected {dim} boundary conditions, got {len(bcs)}")
        return tuple(_coerce_bc(bc) for bc in bcs)

    if isinstance(bcs, dict):
        axis_names = ("x", "y", "z")
        resolved = []
        for i in range(dim):
            name = axis_names[i]
            if name in bcs:
                resolved.append(_coerce_bc(bcs[name]))
            elif i in bcs:
                resolved.append(_coerce_bc(bcs[i]))
            else:
                resolved.append(BoundaryCondition.PERIODIC)
        return tuple(resolved)

    raise TypeError(
        "Boundary conditions must be None, a string, a sequence, or a dict keyed by axis names."
    )


def is_periodic_only(bcs):
    return all(bc == BoundaryCondition.PERIODIC for bc in bcs)
