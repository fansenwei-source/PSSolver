import torch


DEFAULT_TRANSFORM_GROUP_INDEXING = "contiguous_slice"
TRANSFORM_GROUP_INDEXING_MODES = ("advanced", "contiguous_slice")


class Parameters:
    def __init__(self):
        self._params = {}

    def new_param(self, name, value):
        """Set a parameter by name, avoiding duplicates."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Parameter '{name}' must be a torch.Tensor.")
        if name in self._params:
            raise KeyError(f"Parameter '{name}' already exists.")
        self._params[name] = value

    def update_param(self, name, value):
        """Update an existing parameter by name."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Parameter '{name}' must be a torch.Tensor.")
        if name in self._params:
            self._params[name] = value
        else:
            raise KeyError(f"Parameter '{name}' does not exist.")

    def __getitem__(self, key):
        """Return the value for the given key from internal parameters."""
        if key not in self._params:
            raise KeyError(f"Parameter '{key}' does not exist.")
        return self._params[key]
    
    def __setitem__(self, key, value):
        """Set the value for the given key in internal parameters"""
        self.update_param(key, value)
        
    def keys(self):
        return list(self._params.keys())

    def values(self):
        return list(self._params.values())

    def items(self):
        return list(self._params.items())


class Fields:
    def __init__(
        self,
        shape,
        device="cuda",
        dtype=torch.float32,
        batchsize=1,
        transform_group_indexing=DEFAULT_TRANSFORM_GROUP_INDEXING,
    ):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: spatial grid shape of length 1, 2, or 3, e.g. (Nx,), (Nx, Ny), or (Nx, Ny, Nz).
        """
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "dtype must be torch.float32 or torch.float64, "
                f"got {dtype}"
            )
        if transform_group_indexing not in TRANSFORM_GROUP_INDEXING_MODES:
            raise ValueError(
                "transform_group_indexing must be one of "
                f"{TRANSFORM_GROUP_INDEXING_MODES}, got "
                f"{transform_group_indexing!r}"
            )
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.batchsize = batchsize
        self.transform_group_indexing = transform_group_indexing

        self.qx = None
        self.qy = None
        self.qz = None
        self.q2 = None
        self.spatial_grids = None

        self.dim = len(shape)

        self.name_to_idx = {}
        self.dyn_count = 0
        self.stat_count = 0
        self.allowed_boundary_conditions = ("periodic", "dirichlet", "neumann")
        self.default_boundary_conditions = ("periodic",) * self.dim
        self.boundary_conditions = []
        self.transform_plans = []
        self.field_q2 = []
        self.field_q2_safe = []
        self.field_laplacian_eigs = []

        self.transform_backend = None
        self.spectral_shape = tuple(shape)

        self.spatial  = None  # shape: (number_of_fields, batch, *shape)
        self.L_hat    = None  # shape: (dynamic fields, batch, *spectral_shape)
        self.spectral = None  # shape: (all fields, batch, *spectral_shape)

    def set_transform_backend(self, backend):
        if backend.real_dtype != self.dtype:
            raise ValueError(
                "Transform backend real dtype must match Fields dtype; "
                f"got backend={backend.real_dtype}, fields={self.dtype}"
            )
        self.transform_backend = backend
        self.spectral_shape = backend.spectral_shape
        self.spatial_grids = backend.spatial_grids
        self._refresh_metadata()

    def _require_transform_backend(self):
        if self.transform_backend is None:
            raise RuntimeError("Fields does not have a transform backend. Build the solver first.")
        return self.transform_backend

    def _refresh_metadata(self):
        if self.transform_backend is None:
            return

        self.transform_plans = []
        self.field_q2 = []
        self.field_q2_safe = []
        self.field_laplacian_eigs = []

        for bc in self.boundary_conditions:
            metadata = self.transform_backend.get_metadata(bc)
            self.transform_plans.append(metadata.transform_kinds)
            self.field_q2.append(metadata.q2)
            self.field_q2_safe.append(metadata.q2_safe)
            self.field_laplacian_eigs.append(metadata.laplacian_eigs)

    def set_wavenumbers(self, qx=None, qy=None, qz=None, q2=None):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.q2 = q2

    def fftn(self):
        """Return the native spectral representation of all fields."""
        if self.spatial is None:
            raise RuntimeError("Fields.spatial is not initialized.")

        outputs = []
        for idx in range(len(self.boundary_conditions)):
            outputs.append(self.forward_transform(idx))
        return torch.stack(outputs)

    def ifftn(self):
        """Return all fields transformed back to real space."""
        if self.spectral is None:
            raise RuntimeError("Fields.spectral is not initialized.")

        outputs = []
        for idx in range(len(self.boundary_conditions)):
            outputs.append(self.inverse_transform(idx))
        return torch.stack(outputs)

    def _get_field_index(self, name):
        """Return the index for a field name, raising error if not present."""
        if name not in self.name_to_idx:
            raise KeyError(f"Field name '{name}' not found.")
        return self.name_to_idx[name]

    def _normalize_boundary_conditions(self, boundary_conditions):
        """Normalize per-axis boundary-condition labels for a field."""
        if boundary_conditions is None:
            return self.default_boundary_conditions

        if isinstance(boundary_conditions, str):
            boundary_conditions = (boundary_conditions,) * self.dim
        else:
            try:
                boundary_conditions = tuple(boundary_conditions)
            except TypeError as exc:
                raise TypeError(
                    "Boundary conditions must be a string or an iterable of strings."
                ) from exc

        if len(boundary_conditions) == 1 and self.dim > 1:
            boundary_conditions = boundary_conditions * self.dim

        if len(boundary_conditions) != self.dim:
            raise ValueError(
                f"Boundary conditions must provide {self.dim} labels, got {len(boundary_conditions)}."
            )

        normalized = []
        for bc in boundary_conditions:
            if not isinstance(bc, str):
                raise TypeError("Each boundary-condition label must be a string.")
            if bc not in self.allowed_boundary_conditions:
                raise ValueError(
                    "Each boundary-condition label must be one of "
                    f"{self.allowed_boundary_conditions}, got '{bc}'."
                )
            normalized.append(bc)

        return tuple(normalized)

    def _normalize_field_key(self, key):
        if isinstance(key, str):
            return self._get_field_index(key)
        if isinstance(key, int):
            return key
        raise KeyError("Field key must be a field name (str) or index (int).")

    def get_boundary_conditions(self, key):
        """Return boundary-condition labels by field name or index."""
        key = self._normalize_field_key(key)

        if key < 0 or key >= len(self.boundary_conditions):
            raise IndexError(f"Field index {key} is out of range for boundary conditions.")
        return self.boundary_conditions[key]

    def set_boundary_conditions(self, key, boundary_conditions):
        """Set boundary-condition labels by field name or index."""
        normalized = self._normalize_boundary_conditions(boundary_conditions)
        key = self._normalize_field_key(key)

        if key < 0 or key >= len(self.boundary_conditions):
            raise IndexError(f"Field index {key} is out of range for boundary conditions.")

        if self.spatial is not None and key < self.dyn_count and self.L_hat is not None:
            raise RuntimeError(
                "Changing boundary conditions for a built dynamic field requires rebuilding L_hat."
            )

        self.boundary_conditions[key] = normalized
        self._refresh_metadata()

        if self.spatial is not None and self.spectral is not None:
            self.spectral[key] = self.forward_transform(key)

    def get_transform_plan(self, key):
        key = self._normalize_field_key(key)
        if key < 0 or key >= len(self.transform_plans):
            raise IndexError(f"Field index {key} is out of range for transform plans.")
        return self.transform_plans[key]

    def get_q2(self, key, regularize=False):
        key = self._normalize_field_key(key)
        values = self.field_q2_safe if regularize else self.field_q2
        if key < 0 or key >= len(values):
            raise IndexError(f"Field index {key} is out of range for q2 metadata.")
        return values[key]

    def get_laplacian_eigs(self, key):
        key = self._normalize_field_key(key)
        if key < 0 or key >= len(self.field_laplacian_eigs):
            raise IndexError(f"Field index {key} is out of range for laplacian metadata.")
        return self.field_laplacian_eigs[key]

    def get_gradient_boundary_conditions(self, key, axis):
        backend = self._require_transform_backend()
        boundary_conditions = self.get_boundary_conditions(key)
        return backend.get_gradient_boundary_conditions(boundary_conditions, axis)

    def transform_tensor(self, tensor, boundary_conditions):
        backend = self._require_transform_backend()
        boundary_conditions = self._normalize_boundary_conditions(boundary_conditions)
        return backend.forward(tensor, boundary_conditions)

    def inverse_transform_tensor(self, spectral, boundary_conditions):
        backend = self._require_transform_backend()
        boundary_conditions = self._normalize_boundary_conditions(boundary_conditions)
        return backend.inverse(spectral, boundary_conditions)

    def _normalize_field_indices(self, keys):
        try:
            indices = [self._normalize_field_key(key) for key in keys]
        except TypeError as exc:
            raise TypeError("Field indices must be an iterable of names or indices.") from exc
        if len(indices) == 0:
            raise ValueError("At least one field index is required.")
        for index in indices:
            if index < 0 or index >= len(self.boundary_conditions):
                raise IndexError(f"Field index {index} is out of range.")
        return indices

    def _common_boundary_conditions(self, indices):
        boundary_conditions = self.get_boundary_conditions(indices[0])
        for index in indices[1:]:
            if self.get_boundary_conditions(index) != boundary_conditions:
                raise ValueError("All fields in a transform group must have matching boundary conditions.")
        return boundary_conditions

    def group_indices_by_boundary_conditions(self, keys):
        groups = {}
        for index in self._normalize_field_indices(keys):
            groups.setdefault(self.get_boundary_conditions(index), []).append(index)
        return list(groups.values())

    def _transform_group_indexer(self, indices):
        if self.transform_group_indexing == "contiguous_slice":
            start = indices[0]
            if indices == list(range(start, start + len(indices))):
                return slice(start, start + len(indices)), "contiguous_slice"
        return indices, "advanced"

    def transform_group_indexing_metadata(self, keys):
        indices = self._normalize_field_indices(keys)
        _, effective = self._transform_group_indexer(indices)
        fallback_reason = None
        if (
            self.transform_group_indexing == "contiguous_slice"
            and effective == "advanced"
        ):
            fallback_reason = "noncontiguous_field_indices"
        return {
            "requested": self.transform_group_indexing,
            "effective": effective,
            "fallback_reason": fallback_reason,
            "indices": list(indices),
        }

    def select_spatial_group(self, keys):
        indices = self._normalize_field_indices(keys)
        indexer, _ = self._transform_group_indexer(indices)
        return self.spatial[indexer]

    def select_spectral_group(self, keys):
        indices = self._normalize_field_indices(keys)
        indexer, _ = self._transform_group_indexer(indices)
        return self.spectral[indexer]

    def _store_group(self, destination, keys, values):
        indices = self._normalize_field_indices(keys)
        indexer, effective = self._transform_group_indexer(indices)
        if effective == "contiguous_slice":
            destination[indexer].copy_(values)
        else:
            destination[indexer] = values

    def store_spatial_group(self, keys, values):
        self._store_group(self.spatial, keys, values)

    def store_spectral_group(self, keys, values):
        self._store_group(self.spectral, keys, values)

    def forward_transform(self, key, tensor=None):
        backend = self._require_transform_backend()
        key = self._normalize_field_key(key)
        boundary_conditions = self.get_boundary_conditions(key)
        if tensor is None:
            tensor = self.spatial[key]
        return backend.forward(tensor, boundary_conditions)

    def forward_transform_group(self, keys, tensor=None):
        backend = self._require_transform_backend()
        indices = self._normalize_field_indices(keys)
        boundary_conditions = self._common_boundary_conditions(indices)
        if tensor is None:
            tensor = self.select_spatial_group(indices)
        return backend.forward(tensor, boundary_conditions)

    def inverse_transform(self, key, spectral=None, boundary_conditions=None):
        backend = self._require_transform_backend()
        key = self._normalize_field_key(key)
        if boundary_conditions is None:
            boundary_conditions = self.get_boundary_conditions(key)
        else:
            boundary_conditions = self._normalize_boundary_conditions(boundary_conditions)
        if spectral is None:
            spectral = self.spectral[key]
        return backend.inverse(spectral, boundary_conditions)

    def inverse_transform_group(self, keys, spectral=None, boundary_conditions=None):
        backend = self._require_transform_backend()
        indices = self._normalize_field_indices(keys)
        if boundary_conditions is None:
            boundary_conditions = self._common_boundary_conditions(indices)
        else:
            boundary_conditions = self._normalize_boundary_conditions(boundary_conditions)
        if spectral is None:
            spectral = self.select_spectral_group(indices)
        return backend.inverse(spectral, boundary_conditions)

    def laplacian_hat(self, key, spectral=None):
        backend = self._require_transform_backend()
        key = self._normalize_field_key(key)
        if spectral is None:
            spectral = self.spectral[key]
        return backend.laplacian_hat(spectral, self.get_boundary_conditions(key))

    def laplacian(self, key, spectral=None, projector=None):
        lap_hat = self.laplacian_hat(key, spectral=spectral)
        if projector is not None:
            boundary_conditions = self.get_boundary_conditions(key)
            return projector.inverse_transform(
                lap_hat,
                boundary_conditions,
            )
        return self.inverse_transform(key, spectral=lap_hat)

    def gradient_hat(self, key, axis, spectral=None, tensor=None):
        backend = self._require_transform_backend()
        key = self._normalize_field_key(key)
        boundary_conditions = self.get_boundary_conditions(key)

        if spectral is None:
            if tensor is None:
                spectral = self.spectral[key]
            else:
                spectral = self.forward_transform(key, tensor=tensor)

        return backend.gradient_hat(spectral, boundary_conditions, axis)

    def gradient(
        self,
        key,
        axis,
        spectral=None,
        tensor=None,
        projector=None,
    ):
        grad_hat, grad_bcs = self.gradient_hat(
            key,
            axis,
            spectral=spectral,
            tensor=tensor,
        )
        if projector is not None:
            return projector.inverse_transform(grad_hat, grad_bcs)
        return self.inverse_transform(
            key,
            spectral=grad_hat,
            boundary_conditions=grad_bcs,
        )

    def __getitem__(self, key):
        """Access a field by name (optionally with '.hat' or '.bc')."""
        if isinstance(key, str):
            if key.endswith('.hat'):
                field_idx = self._get_field_index(key[:-4])
                return self.spectral[field_idx]
            if key.endswith('.bc'):
                return self.get_boundary_conditions(key[:-3])
            return self.spatial[self._get_field_index(key)]
        raise KeyError("Key must be a field name (str), optionally ending with '.hat' or '.bc'")

    def __setitem__(self, key, value):
        """Set a field by name (optionally with '.hat' or '.bc')."""
        if isinstance(key, str):
            if key.endswith('.hat'):
                field_idx = self._get_field_index(key[:-4])
                self.spectral[field_idx] = value
                if self.spatial is not None:
                    self.spatial[field_idx] = self.inverse_transform(field_idx, spectral=value)
                return
            if key.endswith('.bc'):
                self.set_boundary_conditions(key[:-3], value)
                return

            field_idx = self._get_field_index(key)
            self.spatial[field_idx] = value
            if self.spectral is not None:
                self.spectral[field_idx] = self.forward_transform(field_idx)
            return

        raise KeyError("Key must be a field name (str), optionally ending with '.hat' or '.bc'")

    def keys(self):
        return list(self.name_to_idx.keys())

    def values(self):
        return [self[name] for name in self.name_to_idx.keys()]

    def items(self):
        return [(name, self[name]) for name in self.name_to_idx.keys()]
