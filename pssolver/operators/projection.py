import torch
import torch.nn.functional as functional

from pssolver.core.numerics import (
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
)


class BasisAwareSpectralProjector:
    """Sharp tensor-product projector for the native FFT/DCT/DST bases.

    The cutoff is strict. For a fraction ``f`` the retained integer modes are
    ``abs(k) < f*N/2`` for FFT, ``m < f*N`` for DCT, and ``r < f*N`` for
    DST. Both filtering rules therefore remove the terminal DST mode, whose
    derivative would require the unavailable DCT mode ``m=N``.
    """

    def __init__(
        self,
        solver,
        rule=DEFAULT_DEALIAS_RULE,
        transform_execution=None,
    ):
        if rule not in DEALIAS_RULE_FRACTIONS:
            raise ValueError(
                f"Unknown dealias rule {rule!r}; expected one of "
                f"{tuple(DEALIAS_RULE_FRACTIONS)}."
            )
        if transform_execution is None:
            # With projection disabled there are no discarded modes to skip;
            # preserve the historical full transform automatically. An
            # explicit truncated+none request remains an error below.
            transform_execution = (
                "full"
                if rule == "none"
                else DEFAULT_PROJECTED_TRANSFORM_EXECUTION
            )
        if transform_execution not in PROJECTED_TRANSFORM_EXECUTION_MODES:
            raise ValueError(
                "transform_execution must be 'full' or 'truncated'."
            )
        if transform_execution == "truncated" and rule == "none":
            raise ValueError(
                "truncated projected transforms require enabled dealiasing."
            )
        self.rule = rule
        self.fraction = DEALIAS_RULE_FRACTIONS[rule]
        self.transform_execution = transform_execution
        self.physical_shape = tuple(solver.shape)
        self.transform_backend = solver.transform_backend
        self.shape = self.transform_backend.spectral_shape
        self.device = self.transform_backend.device
        self.real_dtype = self.transform_backend.real_dtype
        self._mask_cache = {}
        self._axis_mask_cache = {}
        self._retained_counts_cache = {}
        self._reduced_periodic_mask_cache = {}
        self._hermitian_plane_index_cache = {}
        self._hermitian_reverse_index_cache = {}

    def _validate_real_spectrum(self, spectral, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if len(boundary_conditions) != len(self.shape):
            raise ValueError(
                "Boundary-condition count must match the spectral dimension."
            )
        if tuple(spectral.shape[-len(self.shape) :]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        return boundary_conditions

    def _full_complex_reflection(self, spectral, boundary_conditions):
        """Return ``conj(F(-k))`` over every periodic spectral axis."""

        boundary_conditions = self._validate_real_spectrum(
            spectral,
            boundary_conditions,
        )
        tensor_axis_offset = spectral.ndim - len(self.shape)
        reflected = spectral
        for local_axis, boundary_condition in enumerate(boundary_conditions):
            if boundary_condition != "periodic":
                continue
            cache_key = (local_axis, spectral.device)
            reverse_index = self._hermitian_reverse_index_cache.get(cache_key)
            if reverse_index is None:
                reverse_index = torch.remainder(
                    -torch.arange(
                        self.physical_shape[local_axis],
                        device=spectral.device,
                        dtype=torch.long,
                    ),
                    self.physical_shape[local_axis],
                )
                self._hermitian_reverse_index_cache[cache_key] = reverse_index
            reflected = reflected.index_select(
                tensor_axis_offset + local_axis,
                reverse_index,
            )
        return reflected.conj()

    def _enforce_hermitian_full(self, spectral, boundary_conditions):
        """Project a full complex spectrum onto the real-field subspace."""

        reflected = self._full_complex_reflection(
            spectral,
            boundary_conditions,
        )
        return 0.5 * (spectral + reflected)

    def _enforce_hermitian_full_(self, spectral, boundary_conditions):
        """In-place counterpart of :meth:`_enforce_hermitian_full`."""

        spectral.copy_(
            self._enforce_hermitian_full(spectral, boundary_conditions)
        )
        return spectral

    def _enforce_hermitian_half_(self, spectral, boundary_conditions):
        """Project packed self-conjugate planes onto the real-field subspace.

        ``rfftn`` stores only the non-negative modes of the packed periodic
        axis.  Its zero plane, and its Nyquist plane for an even physical
        extent, still contain full spectra in every other periodic direction.
        Those planes must therefore obey ``F(k) = conj(F(-k))``.  An IMEX
        update performed directly in spectral space can accumulate an
        anti-Hermitian component there even though ``irfftn`` silently drops
        that component when producing the physical field.

        The pair average below is the orthogonal projection onto the valid
        Hermitian subspace.  Only the one or two constrained planes are
        materialized; the supplied spectral tensor is owned by the caller and
        updated in place, avoiding a second full-spectrum allocation.
        """

        backend = self.transform_backend
        if backend.spectral_storage != "hermitian_half":
            return spectral

        boundary_conditions = self._validate_real_spectrum(
            spectral,
            boundary_conditions,
        )
        packed_axis = backend.hermitian_axis
        if boundary_conditions[packed_axis] != "periodic":
            raise ValueError(
                "hermitian_half storage requires a periodic boundary "
                f"condition on axis {packed_axis}"
            )
        tensor_axis_offset = spectral.ndim - len(self.shape)
        packed_tensor_axis = tensor_axis_offset + packed_axis
        plane_indices = [0]
        packed_physical_size = self.physical_shape[packed_axis]
        if packed_physical_size % 2 == 0:
            plane_indices.append(packed_physical_size // 2)
        planes_index = self._hermitian_plane_index_cache.get(spectral.device)
        if planes_index is None:
            planes_index = torch.tensor(
                plane_indices,
                device=spectral.device,
                dtype=torch.long,
            )
            self._hermitian_plane_index_cache[spectral.device] = planes_index
        planes = spectral.index_select(packed_tensor_axis, planes_index)
        reflected = planes
        for local_axis, boundary_condition in enumerate(boundary_conditions):
            if boundary_condition != "periodic" or local_axis == packed_axis:
                continue
            cache_key = (local_axis, spectral.device)
            reverse_index = self._hermitian_reverse_index_cache.get(cache_key)
            if reverse_index is None:
                axis_size = self.physical_shape[local_axis]
                reverse_index = torch.remainder(
                    -torch.arange(
                        axis_size,
                        device=spectral.device,
                        dtype=torch.long,
                    ),
                    axis_size,
                )
                self._hermitian_reverse_index_cache[cache_key] = reverse_index
            reflected = reflected.index_select(
                tensor_axis_offset + local_axis,
                reverse_index,
            )
        constrained = 0.5 * (planes + reflected.conj())
        spectral.index_copy_(packed_tensor_axis, planes_index, constrained)
        return spectral

    @property
    def enabled(self):
        return self.fraction is not None

    def _axis_mode_numbers(self, axis, boundary_condition):
        key = (axis, boundary_condition)
        if key in self._axis_mask_cache:
            return self._axis_mask_cache[key]

        n = self.physical_shape[axis]
        if boundary_condition == "periodic":
            frequency = (
                torch.fft.rfftfreq
                if self.transform_backend.spectral_storage == "hermitian_half"
                and axis == self.transform_backend.hermitian_axis
                else torch.fft.fftfreq
            )
            mode_numbers = frequency(
                n,
                d=1.0 / n,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = n / 2.0
        elif boundary_condition == "neumann":
            mode_numbers = torch.arange(
                n,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        elif boundary_condition == "dirichlet":
            mode_numbers = torch.arange(
                1,
                n + 1,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        else:
            raise ValueError(
                f"Unsupported boundary condition {boundary_condition!r}."
            )

        keep = mode_numbers.abs() < self.fraction * nyquist_mode
        self._axis_mask_cache[key] = keep
        return keep

    def mask(self, boundary_conditions):
        if not self.enabled:
            raise RuntimeError("The spectral projector is disabled.")

        boundary_conditions = tuple(boundary_conditions)
        if len(boundary_conditions) != len(self.shape):
            raise ValueError(
                "Boundary-condition count must match the spectral dimension."
            )
        if boundary_conditions in self._mask_cache:
            return self._mask_cache[boundary_conditions]

        mask = torch.ones(self.shape, device=self.device, dtype=torch.bool)
        for axis, boundary_condition in enumerate(boundary_conditions):
            axis_keep = self._axis_mode_numbers(axis, boundary_condition)
            view_shape = [1] * len(self.shape)
            view_shape[axis] = axis_keep.shape[0]
            mask &= axis_keep.reshape(view_shape)

        self._mask_cache[boundary_conditions] = mask
        return mask

    def project(self, spectral, boundary_conditions):
        """Project spectral data in its trailing spatial dimensions."""
        if not self.enabled:
            return spectral
        if tuple(spectral.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        mask = self.mask(boundary_conditions)
        # Let PyTorch promote the Boolean mask inside the multiplication
        # kernel.  Materializing a full complex-valued copy of the mask adds
        # one allocation and one device copy at every projection while giving
        # exactly the same finite 0/1 multiplication.
        return spectral * mask

    def project_(self, spectral, boundary_conditions):
        """Project owned spectral storage in place without materializing a copy."""
        if not self.enabled:
            return spectral
        if tuple(spectral.shape[-len(self.shape) :]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        spectral.mul_(self.mask(boundary_conditions))
        return spectral

    def project_real_spectrum(self, spectral, boundary_conditions):
        """Project evolved coefficients into a valid real-field spectrum."""

        projected = self.project(spectral, boundary_conditions)
        storage = self.transform_backend.spectral_storage
        if storage == "full_complex":
            return self._enforce_hermitian_full(
                projected,
                boundary_conditions,
            )
        if projected is spectral:
            projected = spectral.clone()
        return self._enforce_hermitian_half_(
            projected,
            boundary_conditions,
        )

    def project_real_spectrum_(self, spectral, boundary_conditions):
        """Project owned evolved coefficients in place into real-field space."""

        self.project_(spectral, boundary_conditions)
        if self.transform_backend.spectral_storage == "full_complex":
            return self._enforce_hermitian_full_(
                spectral,
                boundary_conditions,
            )
        return self._enforce_hermitian_half_(spectral, boundary_conditions)

    def _retained_counts(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if boundary_conditions not in self._retained_counts_cache:
            self._retained_counts_cache[boundary_conditions] = tuple(
                (
                    self.shape[axis]
                    if bc == "periodic"
                    else int(self._axis_mode_numbers(axis, bc).sum().item())
                )
                for axis, bc in enumerate(boundary_conditions)
            )
        return self._retained_counts_cache[boundary_conditions]

    def _reduced_periodic_mask(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if boundary_conditions in self._reduced_periodic_mask_cache:
            return self._reduced_periodic_mask_cache[boundary_conditions]
        counts = self._retained_counts(boundary_conditions)
        mask = torch.ones(counts, device=self.device, dtype=torch.bool)
        for axis, bc in enumerate(boundary_conditions):
            if bc != "periodic":
                continue
            axis_keep = self._axis_mode_numbers(axis, bc)
            view_shape = [1] * len(self.shape)
            view_shape[axis] = self.shape[axis]
            mask &= axis_keep.reshape(view_shape)
        self._reduced_periodic_mask_cache[boundary_conditions] = mask
        return mask

    def forward_transform(self, tensor, boundary_conditions):
        """Transform and project, optionally avoiding discarded real modes."""
        boundary_conditions = tuple(boundary_conditions)
        if self.transform_execution == "full":
            spectral = self.transform_backend.forward(
                tensor,
                boundary_conditions,
            )
            if self.enabled:
                spectral = spectral * self.mask(boundary_conditions)
            # A transform of a real physical tensor already lies in the
            # Hermitian subspace.  Avoid paying for a redundant plane
            # projection on every forward transform; ``project`` and
            # ``project_dynamic_fields`` enforce the invariant after direct
            # spectral-space evolution, where drift can actually enter.
            return spectral

        counts = self._retained_counts(boundary_conditions)
        if any(count == 0 for count in counts):
            return torch.zeros(
                (*tensor.shape[: -len(self.shape)], *self.shape),
                device=tensor.device,
                dtype=self.transform_backend.spectral_dtype,
            )
        reduced = self.transform_backend.forward(
            tensor,
            boundary_conditions,
            retained_axis_counts=counts,
        )
        reduced.mul_(self._reduced_periodic_mask(boundary_conditions))
        padding = []
        for full_size, retained_count in reversed(
            tuple(zip(self.shape, counts))
        ):
            padding.extend((0, full_size - retained_count))
        return functional.pad(reduced, tuple(padding))

    def inverse_transform(self, spectral, boundary_conditions):
        """Invert coefficients whose high modes have already been projected."""
        boundary_conditions = tuple(boundary_conditions)
        if self.transform_execution == "full":
            return self.transform_backend.inverse(
                spectral,
                boundary_conditions,
            )
        counts = self._retained_counts(boundary_conditions)
        if any(count == 0 for count in counts):
            return torch.zeros(
                (*spectral.shape[: -len(self.shape)], *self.physical_shape),
                device=spectral.device,
                dtype=self.real_dtype,
            )
        return self.transform_backend.inverse(
            spectral,
            boundary_conditions,
            retained_axis_counts=counts,
        )

    def execution_metadata(self):
        """Return JSON-compatible provenance for projected transforms."""
        enabled = self.transform_execution == "truncated"
        spectral_storage = self.transform_backend.spectral_storage
        metadata = {
            "requested": self.transform_execution,
            "effective": self.transform_execution,
            "fallback_allowed": False,
            "fallback_reason": None,
            "truncated_real_basis_axes": enabled,
            "spectral_storage": spectral_storage,
            "backend_storage_shape_preserved": True,
            "full_spectral_storage_preserved": (
                spectral_storage == "full_complex"
            ),
        }
        if spectral_storage == "hermitian_half":
            metadata["hermitian_state_projection"] = (
                "self_conjugate_planes"
            )
        return metadata

    def computed_axis_sizes(self, boundary_conditions):
        """Return transform extents actually evaluated along each axis."""
        if self.transform_execution == "full":
            return self.shape
        return self._retained_counts(boundary_conditions)

    def project_dynamic_fields(self, fields, *, sync_spatial):
        """Project dynamic transform groups and optionally sync real fields."""
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            spectral = fields.select_spectral_group(group)
            self.project_real_spectrum_(spectral, boundary_conditions)
            if (
                fields.transform_group_indexing_metadata(group)["effective"]
                == "advanced"
            ):
                fields.store_spectral_group(group, spectral)
            if sync_spatial:
                fields.store_spatial_group(
                    group,
                    self.inverse_transform(spectral, boundary_conditions),
                )

    def refresh_dynamic_fields(self, fields, *, sync_spatial):
        """Rebuild projected dynamic spectra directly from spatial fields."""
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            spectral = self.forward_transform(
                fields.select_spatial_group(group),
                boundary_conditions,
            )
            fields.store_spectral_group(group, spectral)
            if sync_spatial:
                fields.store_spatial_group(
                    group,
                    self.inverse_transform(spectral, boundary_conditions),
                )

    def retained_axis_counts(self, boundary_conditions):
        if not self.enabled:
            return tuple(self.shape)
        return tuple(
            int(self._axis_mode_numbers(axis, bc).sum().item())
            for axis, bc in enumerate(boundary_conditions)
        )


def _forward_projected(
    backend,
    projector,
    tensor,
    boundary_conditions,
):
    if projector is None:
        return backend.forward(tensor, boundary_conditions)
    return projector.forward_transform(tensor, boundary_conditions)


def _inverse_projected(
    backend,
    projector,
    spectral,
    boundary_conditions,
):
    if projector is None:
        return backend.inverse(spectral, boundary_conditions)
    return projector.inverse_transform(spectral, boundary_conditions)
