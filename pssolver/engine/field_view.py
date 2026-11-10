"""Read-only compatibility surface for the frozen active-nematics kernels."""

from types import MappingProxyType


class FieldView:
    __slots__ = ("_spatial", "_spectral", "_backend", "_indices", "_boundaries")

    def __init__(self, spatial, spectral, backend, components, boundaries):
        self._bind(spatial, spectral)
        self._backend = backend
        self._indices = MappingProxyType({name: index for index, name in enumerate(components)})
        self._boundaries = tuple(boundaries)

    def _bind(self, spatial, spectral):
        self._spatial = spatial
        self._spectral = spectral

    @property
    def spatial(self):
        return self._spatial

    @property
    def spectral(self):
        return self._spectral

    def _index(self, key):
        if isinstance(key, str):
            if key not in self._indices:
                raise KeyError(f"Field name '{key}' not found.")
            return self._indices[key]
        if isinstance(key, int):
            return key
        raise KeyError("Field key must be a field name (str) or index (int).")

    def _boundary(self, key):
        index = self._index(key)
        if index < 0 or index >= len(self._boundaries):
            raise IndexError(f"Field index {index} is out of range for boundary conditions.")
        return self._boundaries[index]

    def __getitem__(self, key):
        if isinstance(key, str):
            if key.endswith(".hat"):
                return self.spectral[self._index(key[:-4])]
            if key.endswith(".bc"):
                return self._boundary(key[:-3])
            return self.spatial[self._index(key)]
        raise KeyError("Key must be a field name (str), optionally ending with '.hat' or '.bc'")

    def laplacian_hat(self, key, spectral=None):
        key = self._index(key)
        if spectral is None:
            spectral = self.spectral[key]
        return self._backend.laplacian_hat(spectral, self._boundary(key))

    def laplacian(self, key, spectral=None, projector=None):
        lap_hat = self.laplacian_hat(key, spectral=spectral)
        if projector is not None:
            return projector.inverse_transform(lap_hat, self._boundary(key))
        return self._backend.inverse(lap_hat, self._boundary(key))

    def gradient_hat(self, key, axis, spectral=None, tensor=None):
        key = self._index(key)
        boundary_conditions = self._boundary(key)
        if spectral is None:
            if tensor is None:
                spectral = self.spectral[key]
            else:
                spectral = self._backend.forward(tensor, boundary_conditions)
        return self._backend.gradient_hat(spectral, boundary_conditions, axis)

    def gradient(self, key, axis, spectral=None, tensor=None, projector=None):
        grad_hat, grad_bcs = self.gradient_hat(
            key,
            axis,
            spectral=spectral,
            tensor=tensor,
        )
        if projector is not None:
            return projector.inverse_transform(grad_hat, grad_bcs)
        return self._backend.inverse(grad_hat, grad_bcs)
