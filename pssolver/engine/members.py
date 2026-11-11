"""Execute each batch member using the exact singleton program and storage layout."""

from dataclasses import replace

import torch


def _progress(state):
    return state.completed_steps, state.refresh_phase, state.constrained_current


def _axis(engine, value):
    axis = value.ndim - len(engine.shape) - 1
    if axis < 0 or value.shape[axis] != engine.batch:
        raise ValueError("state tensor has no matching batch axis")
    return axis


def _map_state(state, function):
    def apply(value, name):
        return None if value is None else function(value, name)

    return replace(
        state,
        spectral=function(state.spectral, "spectral"),
        spatial=function(state.spatial, "spatial"),
        history=tuple(function(value, f"history.{i}") for i, value in enumerate(state.history)),
        solver={key: apply(value, f"solver.{key}") for key, value in state.solver.items()},
        pure_evolved=None
        if state.pure_evolved is None
        else tuple(
            function(value, f"pure_evolved.{i}") for i, value in enumerate(state.pure_evolved)
        ),
        constraint_physical=apply(state.constraint_physical, "constraint_physical"),
    )


def _diagnostics(diagnostics, count):
    if diagnostics is None:
        return [None] * count
    if not isinstance(diagnostics, list):
        raise TypeError("batched diagnostics must be a list of member dictionaries")
    if not diagnostics:
        diagnostics.extend({} for _ in range(count))
    if len(diagnostics) != count or not all(isinstance(row, dict) for row in diagnostics):
        raise ValueError("batched diagnostics must contain one dictionary per member")
    return diagnostics


def _join(engine, states):
    first = states[0]
    assert all(_progress(value) == _progress(first) for value in states)
    assert all(tuple(value.solver) == tuple(first.solver) for value in states)
    flattened = []
    for state in states:
        values = {}
        _map_state(state, lambda value, name, values=values: values.setdefault(name, value))
        flattened.append(values)
    assert all(values.keys() == flattened[0].keys() for values in flattened)

    def concatenate(value, name):
        axis = value.ndim - len(engine.shape) - 1
        assert axis >= 0 and value.shape[axis] == 1
        return torch.cat([values[name] for values in flattened], dim=axis)

    return _map_state(first, concatenate)


def _remember_layouts(engine, state):
    def record(value, name):
        layout = (tuple(value.shape), tuple(value.stride()))
        previous = engine._member_layouts.setdefault(name, layout)
        assert previous[0] == layout[0]
        assert all(
            a == b
            for size, a, b in zip(layout[0], previous[1], layout[1], strict=True)
            if size != 1
        )
        return value

    _map_state(state, record)


def initial_state(engine, evolved, *, diagnostics):
    expected = engine.components[: engine.evolved_count]
    if tuple(evolved) != expected:
        raise ValueError("initial values must contain ordered evolved components")
    for value in evolved.values():
        if (
            tuple(value.shape) not in {engine.shape, (engine.batch, *engine.shape)}
            or value.dtype != engine.dtype
            or value.device != engine.device
        ):
            raise ValueError("initial value shape, dtype or device differs from engine")
    reports = _diagnostics(diagnostics, engine.batch)
    states = []
    for index in range(engine.batch):
        values = {
            name: (value if tuple(value.shape) == engine.shape else value[index]).contiguous()
            for name, value in evolved.items()
        }
        state = engine._member_engine.initial_state(values, diagnostics=reports[index])
        _remember_layouts(engine, state)
        states.append(state)
    return _join(engine, states)


def _member_state(engine, state, index, *, buffer):
    existing = {}
    if buffer is not None:
        _map_state(buffer, lambda value, name: existing.setdefault(name, value))

    def copy(value, name):
        selected = value.narrow(_axis(engine, value), index, 1)
        shape, strides = engine._member_layouts.get(name, (tuple(selected.shape), None))
        if strides is None:
            # Optional history/solver slots have their own singleton-contiguous layout.
            selected = selected.contiguous()
            strides = selected.stride()
            engine._member_layouts[name] = (shape, strides)
        assert tuple(selected.shape) == shape
        target = existing.get(name)
        if target is None:
            target = torch.empty_strided(shape, strides, device=value.device, dtype=value.dtype)
        assert tuple(target.shape) == shape
        assert all(
            a == b for size, a, b in zip(shape, target.stride(), strides, strict=True) if size != 1
        )
        return target.copy_(selected)

    return _map_state(state, copy)


def _controls(engine, controls, index):
    if controls is None:
        return None
    return {
        name: (
            value[index : index + 1].contiguous()
            if isinstance(value, torch.Tensor)
            and tuple(value.shape) == (engine.batch, *engine.shape)
            else value
        )
        for name, value in controls.items()
    }


def execute(engine, operation, state, controls, *, inplace, diagnostics):
    reports = _diagnostics(diagnostics, engine.batch)
    results = []
    final_progress = None
    for index in range(engine.batch):
        member = _member_state(
            engine, state, index, buffer=engine._member_buffer if inplace else None
        )
        result = engine._member_engine._execute(
            operation,
            member,
            _controls(engine, controls, index),
            inplace=inplace,
            diagnostics=reports[index],
        )
        if final_progress is None:
            final_progress = _progress(result)
        assert _progress(result) == final_progress
        if inplace:
            engine._member_buffer = result
            destination = {}
            _map_state(
                state,
                lambda value, name, destination=destination: destination.setdefault(name, value),
            )

            def publish(value, name, destination=destination, index=index):
                destination[name].narrow(_axis(engine, destination[name]), index, 1).copy_(value)
                return value

            _map_state(result, publish)
        else:
            results.append(result)
    if not inplace:
        return _join(engine, results)
    state.completed_steps, state.refresh_phase, state.constrained_current = final_progress
    return state
