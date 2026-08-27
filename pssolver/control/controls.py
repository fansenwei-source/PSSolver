from __future__ import annotations

import math

import torch


def _smooth_interval(coordinate, start, stop, width):
    if stop <= start:
        raise ValueError(f"Interval stop ({stop}) must exceed start ({start}).")
    if width <= 0:
        return ((coordinate >= start) & (coordinate < stop)).to(coordinate.dtype)
    return 0.5 * (
        torch.tanh((coordinate - start) / width)
        - torch.tanh((coordinate - stop) / width)
    )


def smooth_box_mask(shape, bounds, transition_width=1.5, device="cpu", dtype=torch.float32):
    """Create a differentiable, cell-centered soft box mask.

    ``bounds`` contains one ``(start, stop)`` pair per grid axis and is expressed
    in grid-index coordinates, matching the box controls used by the channel
    scripts. A smooth mask avoids the spectral ringing caused by a hard jump in
    spatially varying activity.
    """

    if len(shape) != len(bounds):
        raise ValueError("shape and bounds must have the same number of axes.")
    if isinstance(transition_width, (int, float)):
        transition_width = (float(transition_width),) * len(shape)
    if len(transition_width) != len(shape):
        raise ValueError("transition_width must be scalar or provide one value per axis.")

    mask = torch.ones(shape, device=device, dtype=dtype)
    for axis, (size, interval, width) in enumerate(zip(shape, bounds, transition_width)):
        coordinate = torch.arange(size, device=device, dtype=dtype) + 0.5
        axis_window = _smooth_interval(coordinate, interval[0], interval[1], float(width))
        view_shape = [1] * len(shape)
        view_shape[axis] = size
        mask = mask * axis_window.reshape(view_shape)
    return mask


def partition_mask_along_axis(
    mask,
    num_masks,
    *,
    axis=0,
    bounds=None,
    overlap=0.75,
    periodic=False,
):
    """Partition one spatial mask into normalized overlapping Gaussian windows.

    The returned masks are non-negative and sum pointwise to the input mask.
    ``overlap`` is the Gaussian standard deviation divided by the spacing
    between mask centers. Set ``periodic=True`` to measure axis distances with
    minimum-image wrapping.
    """

    mask = torch.as_tensor(mask)
    if mask.ndim < 1:
        raise ValueError("mask must have at least one spatial axis.")
    if not isinstance(num_masks, int) or isinstance(num_masks, bool) or num_masks <= 0:
        raise ValueError("num_masks must be a positive integer.")
    if axis < 0:
        axis += mask.ndim
    if axis < 0 or axis >= mask.ndim:
        raise IndexError(f"axis {axis} is outside a {mask.ndim}-dimensional mask.")
    if overlap <= 0:
        raise ValueError("overlap must be positive.")
    if mask.min().item() < -1e-7:
        raise ValueError("mask must be non-negative.")
    if num_masks == 1:
        return mask.unsqueeze(0)

    if bounds is None:
        reduction_axes = tuple(index for index in range(mask.ndim) if index != axis)
        profile = mask.amax(dim=reduction_axes) if reduction_axes else mask
        active = torch.nonzero(profile > 0.05 * profile.max(), as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError("Cannot partition an empty mask.")
        start = float(active[0].item())
        stop = float(active[-1].item() + 1)
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must contain (start, stop).")
        start, stop = (float(value) for value in bounds)
    if not 0 <= start < stop <= mask.shape[axis]:
        raise ValueError(
            f"bounds {(start, stop)} must lie inside axis extent [0, {mask.shape[axis]}]."
        )

    spacing = (stop - start) / num_masks
    centers = start + (torch.arange(
        num_masks,
        device=mask.device,
        dtype=mask.dtype,
    ) + 0.5) * spacing
    coordinate = torch.arange(
        mask.shape[axis],
        device=mask.device,
        dtype=mask.dtype,
    ) + 0.5
    distance = coordinate.unsqueeze(0) - centers.unsqueeze(1)
    if periodic:
        period = float(mask.shape[axis])
        distance = torch.remainder(distance + 0.5 * period, period) - 0.5 * period
    log_weights = -0.5 * (distance / (overlap * spacing)).square()
    weights = torch.softmax(log_weights, dim=0)
    view_shape = [num_masks] + [1] * mask.ndim
    view_shape[axis + 1] = mask.shape[axis]
    weights = weights.reshape(view_shape)
    return weights * mask.unsqueeze(0)


def partition_mask_rbf(
    mask,
    counts,
    *,
    bounds=None,
    overlap=0.75,
    periodic=None,
):
    """Partition a spatial mask into normalized tensor-grid Gaussian RBFs."""

    mask = torch.as_tensor(mask)
    ndim = mask.ndim
    counts = tuple(int(value) for value in counts)
    if len(counts) != ndim or any(value <= 0 for value in counts):
        raise ValueError("counts must provide one positive value per spatial axis.")
    if overlap <= 0:
        raise ValueError("overlap must be positive.")
    if mask.min().item() < -1e-7:
        raise ValueError("mask must be non-negative.")
    if periodic is None:
        periodic = (False,) * ndim
    periodic = tuple(bool(value) for value in periodic)
    if len(periodic) != ndim:
        raise ValueError("periodic must provide one flag per spatial axis.")

    if bounds is None:
        inferred = []
        for axis in range(ndim):
            reduction_axes = tuple(index for index in range(ndim) if index != axis)
            profile = mask.amax(dim=reduction_axes) if reduction_axes else mask
            active = torch.nonzero(profile > 0.05 * profile.max(), as_tuple=False).flatten()
            if active.numel() == 0:
                raise ValueError("Cannot partition an empty mask.")
            inferred.append((float(active[0]), float(active[-1] + 1)))
        bounds = tuple(inferred)
    else:
        bounds = tuple(tuple(float(value) for value in interval) for interval in bounds)
    if len(bounds) != ndim:
        raise ValueError("bounds must provide one (start, stop) pair per spatial axis.")

    log_weights = None
    for axis, (size, count, interval, is_periodic) in enumerate(
        zip(mask.shape, counts, bounds, periodic)
    ):
        start, stop = interval
        if not 0 <= start < stop <= size:
            raise ValueError(
                f"Axis {axis} bounds {(start, stop)} must lie inside [0, {size}]."
            )
        spacing = (stop - start) / count
        centers = start + (
            torch.arange(count, device=mask.device, dtype=mask.dtype) + 0.5
        ) * spacing
        coordinate = torch.arange(size, device=mask.device, dtype=mask.dtype) + 0.5
        distance = coordinate.unsqueeze(0) - centers.unsqueeze(1)
        if is_periodic:
            period = float(size)
            distance = torch.remainder(distance + 0.5 * period, period) - 0.5 * period
        axis_log_weight = -0.5 * (distance / (overlap * spacing)).square()
        view_shape = [1] * (2 * ndim)
        view_shape[axis] = count
        view_shape[ndim + axis] = size
        axis_log_weight = axis_log_weight.reshape(view_shape)
        log_weights = (
            axis_log_weight if log_weights is None else log_weights + axis_log_weight
        )

    num_masks = math.prod(counts)
    weights = torch.softmax(log_weights.reshape(num_masks, *mask.shape), dim=0)
    return weights * mask.unsqueeze(0)


class TemporalMaskControl(torch.nn.Module):
    """Bounded, piecewise-constant activity amplitudes on fixed spatial masks."""

    def __init__(
        self,
        masks,
        num_steps,
        block_size,
        batchsize=1,
        alpha_min=0.0,
        alpha_max=5.0,
        initial_alpha=2.5,
        baseline=0.0,
        mask_grid_shape=None,
    ):
        super().__init__()
        masks = torch.as_tensor(masks)
        if masks.ndim < 2:
            raise ValueError("masks must have shape (num_masks, *spatial_shape).")
        if block_size <= 0 or num_steps <= 0:
            raise ValueError("num_steps and block_size must be positive.")
        if alpha_max <= alpha_min:
            raise ValueError("alpha_max must exceed alpha_min.")

        mask_sum = masks.sum(dim=0)
        if masks.min().item() < -1e-7 or mask_sum.max().item() > 1.0 + 1e-6:
            raise ValueError("masks must be nonnegative and sum to at most one at every point.")

        self.register_buffer("masks", masks)
        self.num_steps = int(num_steps)
        self.block_size = int(block_size)
        self.num_blocks = math.ceil(num_steps / block_size)
        self.batchsize = int(batchsize)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.baseline = float(baseline)
        if mask_grid_shape is None:
            mask_grid_shape = (masks.shape[0],)
        self.mask_grid_shape = tuple(int(value) for value in mask_grid_shape)
        if any(value <= 0 for value in self.mask_grid_shape):
            raise ValueError("mask_grid_shape values must be positive.")
        if math.prod(self.mask_grid_shape) != masks.shape[0]:
            raise ValueError("mask_grid_shape product must equal the number of masks.")

        initial = torch.as_tensor(initial_alpha, device=masks.device, dtype=masks.dtype)
        if initial.ndim == 0:
            initial = initial.repeat(masks.shape[0])
        if initial.shape != (masks.shape[0],):
            raise ValueError("initial_alpha must be scalar or have one value per mask.")
        fraction = ((initial - alpha_min) / (alpha_max - alpha_min)).clamp(1e-5, 1 - 1e-5)
        initial_logits = torch.log(fraction / (1 - fraction))
        self.logits = torch.nn.Parameter(initial_logits.repeat(self.num_blocks, 1))

    @property
    def spatial_shape(self):
        return tuple(self.masks.shape[1:])

    def amplitudes(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.logits)

    def set_amplitudes(self, amplitudes):
        """Replace all time-block amplitudes while preserving the bounded parameterization."""

        amplitudes = torch.as_tensor(
            amplitudes, device=self.logits.device, dtype=self.logits.dtype
        )
        if amplitudes.shape != self.logits.shape:
            raise ValueError(
                f"Expected amplitudes with shape {tuple(self.logits.shape)}, "
                f"got {tuple(amplitudes.shape)}."
            )
        fraction = ((amplitudes - self.alpha_min) / (self.alpha_max - self.alpha_min)).clamp(
            1e-5, 1.0 - 1e-5
        )
        with torch.no_grad():
            self.logits.copy_(torch.log(fraction / (1.0 - fraction)))

    def field_for_step(self, step):
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"Control step {step} is outside [0, {self.num_steps}).")
        block = min(step // self.block_size, self.num_blocks - 1)
        amplitudes = self.amplitudes()[block]
        mask_sum = self.masks.sum(dim=0)
        field = self.baseline * (1.0 - mask_sum)
        field = field + torch.einsum("m,m...->...", amplitudes, self.masks)
        return field.unsqueeze(0).expand(self.batchsize, *self.spatial_shape)

    def temporal_regularization(self, weight):
        if weight == 0 or self.num_blocks < 2:
            return self.logits.sum() * 0.0
        differences = self.amplitudes()[1:] - self.amplitudes()[:-1]
        return 0.5 * float(weight) * differences.square().mean()

    def spatial_regularization(self, weight):
        if weight == 0 or self.masks.shape[0] < 2:
            return self.logits.sum() * 0.0
        amplitudes = self.amplitudes().reshape(self.num_blocks, *self.mask_grid_shape)
        penalties = []
        for axis, size in enumerate(self.mask_grid_shape, start=1):
            if size > 1:
                penalties.append(torch.diff(amplitudes, dim=axis).square().mean())
        if not penalties:
            return self.logits.sum() * 0.0
        return 0.5 * float(weight) * torch.stack(penalties).mean()
