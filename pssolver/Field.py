import torch
from .boundary import normalize_bcs, is_periodic_only
from .transforms import _forward_1d_bc, _inverse_1d_bc

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
    def __init__(self, shape, device="cuda", dtype=torch.float32, batchsize = 1, bcs=None):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: tuple of length 1, 2, or 3 (e.g., (B, Nx,), (B, Nx, Ny), (B, Nx, Ny, Nz))
        """
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.batchsize = batchsize

        self.qx = None
        self.qy = None 
        self.qz = None
        self.q2 = None
        self.field_wavenumbers = None
        self.field_spectral_shapes = None
        self.field_spectral_slices = None

        self.dim = len(shape)
        self._field_bcs_config = bcs if self._is_field_bcs_config(bcs) else None
        if self._field_bcs_config is None:
            self.bcs = normalize_bcs(bcs, self.dim)
        else:
            default_bcs = self._field_bcs_config.get("default", None)
            self.bcs = normalize_bcs(default_bcs, self.dim)
        self.field_bcs = None

        self.name_to_idx = {}
        self.dyn_count = 0
        self.stat_count = 0
        self.dyn_names = []
        self.stat_names = []

        self.spatial = {}
        self.L_hat = {}
        self.spectral = {}

    @staticmethod
    def _is_field_bcs_config(bcs):
        if not isinstance(bcs, dict):
            return False
        axis_keys = {"x", "y", "z", 0, 1, 2}
        return any(key not in axis_keys for key in bcs.keys())

    def finalize_field_bcs(self, field_names):
        if self._field_bcs_config is None:
            self.field_bcs = {name: self.bcs for name in field_names}
            return

        resolved = {}
        for name in field_names:
            if name in self._field_bcs_config:
                resolved[name] = normalize_bcs(self._field_bcs_config[name], self.dim)
            elif "default" in self._field_bcs_config:
                resolved[name] = normalize_bcs(self._field_bcs_config["default"], self.dim)
            else:
                resolved[name] = normalize_bcs(None, self.dim)
        self.field_bcs = resolved

    
    def set_wavenumbers(self, qx=None, qy=None, qz=None, q2=None):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.q2 = q2
        self.field_wavenumbers = None

    def set_wavenumbers_for_fields(self, field_wavenumbers):
        self.field_wavenumbers = field_wavenumbers

    def set_spectral_meta_for_fields(self, spectral_shapes, spectral_slices):
        self.field_spectral_shapes = spectral_shapes
        self.field_spectral_slices = spectral_slices

    def get_spectral_shape(self, field_name):
        if self.field_spectral_shapes and field_name in self.field_spectral_shapes:
            return self.field_spectral_shapes[field_name]
        return self.shape

    def get_spectral_slices(self, field_name):
        if self.field_spectral_slices and field_name in self.field_spectral_slices:
            return self.field_spectral_slices[field_name]
        return tuple(slice(None) for _ in range(self.dim))

    def embed_spectral_full(self, tensor, field_name):
        """
        Embed a field's spectral tensor into full-grid shape using its BC slices.
        """
        if self.field_spectral_shapes is None or self.field_spectral_slices is None:
            return tensor
        if field_name not in self.field_spectral_shapes:
            return tensor
        target_shape = (tensor.shape[0], *self.shape)
        if tensor.shape == target_shape:
            return tensor
        out = tensor.new_zeros(target_shape)
        slicers = (slice(None),) + self.get_spectral_slices(field_name)
        out[slicers] = tensor
        return out

    def crop_spectral_full(self, tensor, field_name):
        """
        Crop a full-grid spectral tensor down to the field's effective spectral shape.
        """
        if self.field_spectral_shapes is None or self.field_spectral_slices is None:
            return tensor
        if field_name not in self.field_spectral_shapes:
            return tensor
        slicers = (slice(None),) + self.get_spectral_slices(field_name)
        return tensor[slicers]

    def get_wavenumbers(self, field_name):
        if self.field_wavenumbers and field_name in self.field_wavenumbers:
            return self.field_wavenumbers[field_name]
        return self.qx, self.qy, self.qz, self.q2

    def map_grouped(self, names, func, spectral=False):
        """
        Group fields by BCs, apply func to stacked tensors per group, and return dict outputs.

        func signature: func(tensor, bcs, names) -> tensor with same leading dim
        """
        field_bcs = self.field_bcs or {name: self.bcs for name in names}
        groups = {}
        for name in names:
            groups.setdefault(field_bcs[name], []).append(name)

        source = self.spectral if spectral else self.spatial
        outputs = {}
        for bcs, group_names in groups.items():
            stacked = torch.stack([source[name] for name in group_names], dim=0)
            result = func(stacked, bcs, group_names)
            if result.shape[0] != len(group_names):
                raise ValueError("Grouped function must preserve leading dimension")
            for i, name in enumerate(group_names):
                outputs[name] = result[i]
        return outputs

    def fftn(self):
        """Return batched N-dimensional FFT of all fields"""
        field_order = [name for name, _ in sorted(self.name_to_idx.items(), key=lambda kv: kv[1])]
        field_bcs = self.field_bcs or {name: self.bcs for name in field_order}
        if not isinstance(self.spatial, dict):
            if all(is_periodic_only(field_bcs[name]) for name in field_order):
                return torch.fft.fftn(self.spatial, dim=tuple(range(-self.dim, 0)))

        groups = {}
        for name in field_order:
            groups.setdefault(field_bcs[name], []).append(name)

        outputs = {}
        for bcs, names in groups.items():
            X = torch.stack([self.spatial[name] for name in names], dim=0)
            if is_periodic_only(bcs):
                X = torch.fft.fftn(X, dim=tuple(range(-self.dim, 0)))
            else:
                for axis, bc in enumerate(bcs):
                    X = _forward_1d_bc(X, bc, dim=-(self.dim - axis))
            for idx, name in enumerate(names):
                outputs[name] = X[idx]
        return outputs
        # return torch.fft.fftn(self.spatial, dim=tuple(range(2, 2 + self.dim)))

    def ifftn(self):
        """Return batched N-dimensional IFFT (real part) of all fields"""
        field_order = [name for name, _ in sorted(self.name_to_idx.items(), key=lambda kv: kv[1])]
        field_bcs = self.field_bcs or {name: self.bcs for name in field_order}
        if not isinstance(self.spectral, dict):
            if all(is_periodic_only(field_bcs[name]) for name in field_order):
                return torch.fft.ifftn(self.spectral, dim=tuple(range(-self.dim, 0))).real

        groups = {}
        for name in field_order:
            groups.setdefault(field_bcs[name], []).append(name)

        outputs = {}
        for bcs, names in groups.items():
            x = torch.stack([self.spectral[name] for name in names], dim=0)
            if is_periodic_only(bcs):
                x = torch.fft.ifftn(x, dim=tuple(range(-self.dim, 0))).real
            else:
                for axis, bc in enumerate(bcs):
                    N_full = self.shape[axis]
                    x = _inverse_1d_bc(x, bc, dim=-(self.dim - axis), N_full=N_full)
            for idx, name in enumerate(names):
                outputs[name] = x[idx]
        return outputs

    def _get_field_index(self, name):
        """Return the index for a field name, raising error if not present."""
        if name not in self.name_to_idx:
            raise KeyError(f"Field name '{name}' not found.")
        return self.name_to_idx[name]
    
    def __getitem__(self, key):
        """Access a field by name (optionally with '.hat') """
        if isinstance(key, str):
            if key.endswith('.hat'):
                name = key[:-4]
                if isinstance(self.spectral, dict):
                    return self.spectral[name]
                field_idx = self._get_field_index(name)
                return self.spectral[field_idx]
            if isinstance(self.spatial, dict):
                return self.spatial[key]
            field_idx = self._get_field_index(key)
            return self.spatial[field_idx]
        else:
            raise KeyError("Key must be a field name (str), optionally ending with '.hat'")

    def __setitem__(self, key, value):
        """Set a field by name (optionally with '.hat')"""
        if isinstance(key, str):
            if key.endswith('.hat'):
                name = key[:-4]
                if isinstance(self.spectral, dict):
                    self.spectral[name] = value
                    return
                field_idx = self._get_field_index(name)
                self.spectral[field_idx] = value
                return
            if isinstance(self.spatial, dict):
                self.spatial[key] = value
                return
            field_idx = self._get_field_index(key)
            self.spatial[field_idx] = value
        else:
            raise KeyError("Key must be a field name (str), optionally ending with '.hat'")


    def keys(self):
        return list(self.name_to_idx.keys())

    def values(self):
        return [self[name] for name in self.name_to_idx.keys()]

    def items(self):
        return [(name, self[name]) for name in self.name_to_idx.keys()]
