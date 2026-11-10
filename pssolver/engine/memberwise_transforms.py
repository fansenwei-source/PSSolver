"""Preserve singleton FFT execution for batched single-component fields."""

import torch


class _MemberwiseTransforms:
    """Delegate transforms, splitting only the explicit scalar-field batch axis."""

    def __init__(self, backend, batch):
        self._backend = backend
        self._batch = batch

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _apply(self, method, value, boundary_conditions, *, retained_axis_counts):
        if value.ndim != 1 + len(self._backend.shape):
            return method(value, boundary_conditions, retained_axis_counts=retained_axis_counts)
        if value.shape[0] != self._batch:
            raise RuntimeError("single-component transform batch differs from engine")
        return torch.cat(
            [
                method(
                    value.narrow(0, index, 1),
                    boundary_conditions,
                    retained_axis_counts=retained_axis_counts,
                )
                for index in range(self._batch)
            ],
            dim=0,
        )

    def forward(self, tensor, boundary_conditions, *, retained_axis_counts=None):
        return self._apply(
            self._backend.forward,
            tensor,
            boundary_conditions,
            retained_axis_counts=retained_axis_counts,
        )

    def inverse(self, spectral, boundary_conditions, *, retained_axis_counts=None):
        return self._apply(
            self._backend.inverse,
            spectral,
            boundary_conditions,
            retained_axis_counts=retained_axis_counts,
        )
