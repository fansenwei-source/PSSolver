"""Normalize component-uniform linear expressions without numerical imports."""

from pssolver.core.terms import Expr, FieldRef


class LinearAnalysisError(ValueError):
    """A mathematical classification failure, translated by the assembler."""

    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code


def _fail(code="invalid_linear_expression", reason="not a linear scalar expression"):
    raise LinearAnalysisError(code, reason)


def normalize(expression, target):
    """Return sorted, combined ``(coefficient, derivative_orders)`` monomials."""

    def walk(value):
        if isinstance(value, FieldRef):
            if value != target:
                _fail(
                    "linear_requires_modal_block",
                    "cross-field or component coupling needs a modal block",
                )
            return [(1.0, (0, 0, 0))]
        if not isinstance(value, Expr):
            if value == 0:
                return []
            _fail()
        operation, args = value.operation, value.arguments
        if operation == "parameter":
            _fail(
                "linear_requires_general_solver",
                "field or controllable coefficients need a general solver",
            )
        if operation == "add":
            return walk(args[0]) + walk(args[1])
        if operation == "multiply":
            left, right = args
            if isinstance(right, (int, float)):
                left, right = right, left
            if not isinstance(left, (int, float)):
                if any(isinstance(item, Expr) and item.operation == "parameter" for item in args):
                    _fail(
                        "linear_requires_general_solver",
                        "field or controllable coefficients need a general solver",
                    )
                _fail()
            return [(float(left) * coefficient, orders) for coefficient, orders in walk(right)]
        if operation in {"laplacian", "derivative"}:
            rows = walk(args[0])
            axes = range(3) if operation == "laplacian" else (args[1],)
            order = 2 if operation == "laplacian" else args[2]
            if not isinstance(order, int) or isinstance(order, bool) or order < 1:
                _fail()
            if any(not isinstance(axis, int) or axis not in range(3) for axis in axes):
                _fail()
            if operation == "derivative" and len(target.components) > 1:
                _fail(
                    reason="a coordinate derivative is not a scalar term on a multi-component target"
                )
            return [
                (coefficient, tuple(n + (order if i == axis else 0) for i, n in enumerate(orders)))
                for coefficient, orders in rows
                for axis in axes
            ]
        _fail()

    combined = {}
    for coefficient, orders in walk(expression):
        combined[orders] = combined.get(orders, 0.0) + coefficient
    return tuple((combined[orders], orders) for orders in sorted(combined) if combined[orders] != 0)
