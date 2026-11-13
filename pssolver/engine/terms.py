"""Generic periodic expression symbols and physical model-local kernels."""

import torch


def expression_symbol(backend, boundaries, monomials, dtype):
    """Materialize a diagonal symbol using the backend's derivative convention."""
    reference = backend.get_q2(boundaries)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    one = torch.ones(reference.shape, dtype=complex_dtype, device=reference.device)
    result = torch.zeros_like(one)
    for coefficient, orders in monomials:
        symbol = one
        basis = boundaries
        for axis, order in enumerate(orders):
            if order and order % 2 == 0:
                modes = backend.get_metadata(basis).axis_modes[axis]
                shape = [1] * len(boundaries)
                shape[axis] = modes.numel()
                symbol = symbol * (-(modes ** 2)).reshape(shape) ** (order // 2)
            else:
                for _ in range(order):
                    symbol, basis = backend.gradient_hat(symbol, basis, axis)
        result = result + coefficient * symbol
    return result.real if all(n % 2 == 0 for _, orders in monomials for n in orders) else result


def pointwise_explicit_term(engine, plan_term, kernel):
    """Bind physical inputs and project one exact-shape local-kernel output."""
    inputs = plan_term.arguments["inputs"]
    coordinates = None
    if any(item[0] == "coordinates" for item in inputs):
        axes = [
            torch.arange(n, device=engine.device, dtype=engine.dtype) * (length / n)
            for n, length in zip(engine.shape, engine.plan.geometry.lengths, strict=True)
        ]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij")).unsqueeze(1)
    target = engine._observable_slices[plan_term.target]
    expected = (target.stop - target.start, engine.batch, *engine.shape)

    def derivative(view, item):
        spectral = view.spectral[engine._observable_slices[item[1]]]
        basis = engine.boundaries
        for _ in range(item[3]):
            spectral, basis = engine.transform_backend.gradient_hat(spectral, basis, item[2])
        return engine.projector.inverse_transform(spectral, basis)

    def evaluate(view, controls, *, completed_steps, derivatives=None):
        values = []
        for item in inputs:
            if item[0] == "coordinates":
                value = coordinates
            elif item[0] == "time":
                value = completed_steps * engine.dt
            elif item[0] == "field":
                value = view.spatial[engine._observable_slices[item[1]]]
            elif derivatives is None:
                value = derivative(view, item)
            else:
                orders = tuple(item[3] if axis == item[2] else 0 for axis in range(3))
                key = (item[1], orders, engine.boundaries)
                value = derivatives.resolve(key, lambda item=item: derivative(view, item))
            values.append(value)
        result = kernel(*values)
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != expected:
            raise ValueError(f"model kernel must return exactly shape {expected}")
        return engine.projector.forward_transform(result, engine.boundaries)

    return evaluate
