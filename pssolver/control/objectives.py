from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as torch_functional

from ..models.active_nematics import Q_magnitude
from ..snapshots import load_q_snapshot


def load_q_target(path, device="cpu", dtype=torch.float32):
    """Load canonical ``Q_*.npy`` as ``(5, batch=1, Nx, Ny, Nz)``.

    The Q file must be colocated with schema-v1 active-nematic
    ``metadata.json`` declaring the sole canonical convention and S values.
    """

    data = load_q_snapshot(
        path,
        require_S_initial=True,
    ).values
    tensor = torch.as_tensor(data, device=device, dtype=dtype)
    return tensor.movedim(-1, 0).unsqueeze(1)


class QTrackingObjective:
    """Differentiable Q/core tracking with activity effort regularization."""

    def __init__(
        self,
        target_q,
        dt,
        spatial_mask=None,
        running_weight=0.0,
        terminal_weight=1.0,
        control_weight=0.0,
        reference_alpha=0.0,
        S_bulk=None,
        core_threshold_fraction=0.65,
        core_transition_fraction=0.08,
        target_core_q_weight=0.0,
        running_core_weight=0.0,
        terminal_core_weight=0.0,
        running_core_mass_weight=0.0,
        terminal_core_mass_weight=0.0,
        domain_lengths=None,
        running_moment_weight=0.0,
        terminal_moment_weight=0.0,
        running_aligned_shape_weight=0.0,
        terminal_aligned_shape_weight=0.0,
    ):
        self.target_q = torch.as_tensor(target_q)
        self.dt = float(dt)
        self.running_weight = float(running_weight)
        self.terminal_weight = float(terminal_weight)
        self.control_weight = float(control_weight)
        self.reference_alpha = float(reference_alpha)
        self.S_bulk = (
            None if S_bulk is None else float(S_bulk)
        )
        self.core_threshold_fraction = float(core_threshold_fraction)
        self.core_transition_fraction = float(core_transition_fraction)
        self.target_core_q_weight = float(target_core_q_weight)
        self.running_core_weight = float(running_core_weight)
        self.terminal_core_weight = float(terminal_core_weight)
        self.running_core_mass_weight = float(running_core_mass_weight)
        self.terminal_core_mass_weight = float(terminal_core_mass_weight)
        self.domain_lengths = (
            None if domain_lengths is None else tuple(float(value) for value in domain_lengths)
        )
        self.running_moment_weight = float(running_moment_weight)
        self.terminal_moment_weight = float(terminal_moment_weight)
        self.running_aligned_shape_weight = float(running_aligned_shape_weight)
        self.terminal_aligned_shape_weight = float(terminal_aligned_shape_weight)

        core_weights = (
            self.target_core_q_weight,
            self.running_core_weight,
            self.terminal_core_weight,
            self.running_core_mass_weight,
            self.terminal_core_mass_weight,
        )
        if any(weight < 0 for weight in core_weights):
            raise ValueError("Core-aware objective weights must be non-negative.")
        if self.running_moment_weight < 0 or self.terminal_moment_weight < 0:
            raise ValueError("Core-moment objective weights must be non-negative.")
        if (
            self.running_aligned_shape_weight < 0
            or self.terminal_aligned_shape_weight < 0
        ):
            raise ValueError("Aligned core-shape weights must be non-negative.")
        if self.S_bulk is not None and self.S_bulk <= 0:
            raise ValueError("S_bulk must be positive.")
        if not 0 < self.core_threshold_fraction < 1:
            raise ValueError("core_threshold_fraction must lie in (0, 1).")
        if self.core_transition_fraction <= 0:
            raise ValueError("core_transition_fraction must be positive.")
        if any(core_weights) and self.S_bulk is None:
            raise ValueError("S_bulk is required when core-aware weights are used.")
        if self.S_bulk is not None and self.target_q.shape[0] != 5:
            raise ValueError("Core-aware tracking requires five independent Q components.")

        if spatial_mask is None:
            spatial_mask = torch.ones(self.target_q.shape[2:], device=self.target_q.device)
        spatial_mask = torch.as_tensor(
            spatial_mask,
            device=self.target_q.device,
            dtype=self.target_q.dtype,
        )
        if tuple(spatial_mask.shape) != tuple(self.target_q.shape[2:]):
            raise ValueError("spatial_mask shape must match the target spatial shape.")
        self.spatial_mask = spatial_mask
        self.target_core_density = (
            self._core_density(self.target_q).detach()
            if self.S_bulk is not None
            else None
        )

    def _tracking_mse(self, q):
        target = self.target_q.to(device=q.device, dtype=q.dtype)
        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype)
        weights = mask.unsqueeze(0)
        if self.target_core_density is not None and self.target_core_q_weight:
            target_core = self.target_core_density.to(device=q.device, dtype=q.dtype)
            weights = weights * (1.0 + self.target_core_q_weight * target_core)
        if weights.shape[0] == 1 and q.shape[1] > 1:
            weights = weights.expand(q.shape[1], *weights.shape[1:])
        weighted_error = (q - target).square() * weights.unsqueeze(0)
        denominator = weights.sum(dtype=torch.float64).clamp_min(1e-12) * q.shape[0]
        return weighted_error.sum(dtype=torch.float64) / denominator

    def _core_density(self, q):
        magnitude = Q_magnitude(q.movedim(0, -1))
        magnitude = torch.sqrt(magnitude.square() + torch.finfo(q.dtype).eps)
        threshold = self.core_threshold_fraction * self.S_bulk
        transition = self.core_transition_fraction * self.S_bulk
        density = torch.sigmoid((threshold - magnitude) / transition)
        bulk_density = torch.sigmoid(torch.as_tensor(
            (threshold - self.S_bulk) / transition,
            device=q.device,
            dtype=q.dtype,
        ))
        normalized_deficit = (density - bulk_density) / (1.0 - bulk_density)
        return normalized_deficit.square()

    def _core_metrics(self, q):
        zero = q.sum(dtype=torch.float64) * 0.0
        if self.target_core_density is None:
            return zero, zero
        candidate = self._core_density(q)
        target = self.target_core_density.to(device=q.device, dtype=q.dtype)
        if target.shape[0] == 1 and candidate.shape[0] > 1:
            target = target.expand(candidate.shape[0], *target.shape[1:])
        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        if mask.shape[0] == 1 and candidate.shape[0] > 1:
            mask = mask.expand(candidate.shape[0], *mask.shape[1:])
        spatial_dims = tuple(range(1, target.ndim))
        target_mass = (target * mask).sum(dim=spatial_dims, dtype=torch.float64)
        shape_denominator = target_mass.sum().clamp_min(1e-12)
        shape_mse = (
            ((candidate - target).square() * mask).sum(dtype=torch.float64)
            / shape_denominator
        )
        candidate_mass = (candidate * mask).sum(
            dim=tuple(range(1, candidate.ndim)),
            dtype=torch.float64,
        )
        mass_ratio = candidate_mass / target_mass.clamp_min(1e-12)
        mass_penalty = (mass_ratio - 1.0).square().mean()
        return shape_mse, mass_penalty

    def components(self, q, alpha=None):
        core_mse, core_mass_penalty = self._core_metrics(q)
        if alpha is None:
            effort = q.sum(dtype=torch.float64) * 0.0
        else:
            effort = (alpha - self.reference_alpha).square().mean(dtype=torch.float64)
        return {
            "q_tracking_mse": self._tracking_mse(q),
            "core_tracking_mse": core_mse,
            "core_mass_penalty": core_mass_penalty,
            "control_effort": effort,
        }

    def stage_cost(self, q, alpha, step):
        del step
        components = self.components(q, alpha)
        cost = (
            self.running_weight * components["q_tracking_mse"]
            + self.running_core_weight * components["core_tracking_mse"]
            + self.running_core_mass_weight * components["core_mass_penalty"]
            + self.control_weight * components["control_effort"]
        )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q)
        cost = (
            self.terminal_weight * components["q_tracking_mse"]
            + self.terminal_core_weight * components["core_tracking_mse"]
            + self.terminal_core_mass_weight * components["core_mass_penalty"]
        )
        return 0.5 * cost


class CoreTranslationObjective(QTrackingObjective):
    """Track the differentiable defect-core center without matching its shape."""

    include_core_mass = True

    def __init__(self, *args, domain_lengths=None, **kwargs):
        if domain_lengths is None:
            raise ValueError("domain_lengths is required for core translation tracking.")
        if len(domain_lengths) != 3 or any(length <= 0 for length in domain_lengths):
            raise ValueError("domain_lengths must contain three positive values.")

        # These objectives deliberately exclude full-Q and core-shape matching.
        kwargs["target_core_q_weight"] = 0.0
        kwargs["running_core_weight"] = 0.0
        kwargs["terminal_core_weight"] = 0.0
        super().__init__(*args, **kwargs)
        if self.S_bulk is None:
            raise ValueError(
                "S_bulk is required for core translation tracking."
            )
        self.domain_lengths = tuple(float(length) for length in domain_lengths)
        self.target_core_center, self.target_core_mass = self._core_center_and_mass(
            self.target_q
        )
        self.target_core_center = self.target_core_center.detach()
        self.target_core_mass = self.target_core_mass.detach()

    def _core_center_and_mass(self, q):
        density = self._core_density(q)
        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        if mask.shape[0] == 1 and density.shape[0] > 1:
            mask = mask.expand(density.shape[0], *mask.shape[1:])
        weights = density * mask
        spatial_dims = tuple(range(1, weights.ndim))
        mass = weights.sum(dim=spatial_dims, dtype=torch.float64)
        safe_mass = mass.clamp_min(1e-12)

        axes = [
            (torch.arange(size, device=q.device, dtype=q.dtype) + 0.5)
            * length
            / size
            for size, length in zip(q.shape[2:], self.domain_lengths)
        ]

        theta_x = 2.0 * torch.pi * axes[0] / self.domain_lengths[0]
        weight_x = weights.sum(dim=(2, 3), dtype=torch.float64)
        sine = (weight_x * torch.sin(theta_x)).sum(dim=1)
        cosine = (weight_x * torch.cos(theta_x)).sum(dim=1)
        center_x = torch.remainder(
            torch.atan2(sine, cosine), 2.0 * torch.pi
        ) * self.domain_lengths[0] / (2.0 * torch.pi)

        weight_y = weights.sum(dim=(1, 3), dtype=torch.float64)
        weight_z = weights.sum(dim=(1, 2), dtype=torch.float64)
        center_y = (weight_y * axes[1]).sum(dim=1) / safe_mass
        center_z = (weight_z * axes[2]).sum(dim=1) / safe_mass
        return torch.stack((center_x, center_y, center_z), dim=1), mass

    def _translation_metrics(self, q):
        center, mass = self._core_center_and_mass(q)
        target_center = self.target_core_center.to(device=q.device, dtype=center.dtype)
        target_mass = self.target_core_mass.to(device=q.device, dtype=mass.dtype)
        if target_center.shape[0] == 1 and center.shape[0] > 1:
            target_center = target_center.expand(center.shape[0], -1)
            target_mass = target_mass.expand(mass.shape[0])

        delta = center - target_center
        theta_delta = 2.0 * torch.pi * delta[:, 0] / self.domain_lengths[0]
        periodic_dx = (
            torch.atan2(torch.sin(theta_delta), torch.cos(theta_delta))
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )
        delta = torch.cat((periodic_dx.unsqueeze(1), delta[:, 1:]), dim=1)
        position_penalty = delta.square().sum(dim=1).mean()
        mass_ratio = mass / target_mass.clamp_min(1e-12)
        mass_penalty = (mass_ratio - 1.0).square().mean()
        return position_penalty, mass_penalty

    def components(self, q, alpha=None):
        position_penalty, mass_penalty = self._translation_metrics(q)
        if alpha is None:
            effort = q.sum(dtype=torch.float64) * 0.0
        else:
            effort = (alpha - self.reference_alpha).square().mean(dtype=torch.float64)
        return {
            "core_position_penalty": position_penalty,
            "core_mass_penalty": mass_penalty,
            "control_effort": effort,
        }

    def stage_cost(self, q, alpha, step):
        del step
        components = self.components(q, alpha)
        cost = (
            self.running_weight * components["core_position_penalty"]
            + self.control_weight * components["control_effort"]
        )
        if self.include_core_mass:
            cost = cost + (
                self.running_core_mass_weight * components["core_mass_penalty"]
            )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q)
        cost = self.terminal_weight * components["core_position_penalty"]
        if self.include_core_mass:
            cost = cost + (
                self.terminal_core_mass_weight * components["core_mass_penalty"]
            )
        return 0.5 * cost


class CoreTranslationNoMassObjective(CoreTranslationObjective):
    """Track only the defect-core center, with no core-mass penalty."""

    include_core_mass = False


class CoreAwareMomentObjective(QTrackingObjective):
    """Q/core tracking augmented by a translation-invariant core moment loss."""

    def __init__(self, *args, domain_lengths=None, **kwargs):
        if domain_lengths is None:
            raise ValueError("domain_lengths is required for core moment tracking.")
        if len(domain_lengths) != 3 or any(length <= 0 for length in domain_lengths):
            raise ValueError("domain_lengths must contain three positive values.")
        super().__init__(*args, domain_lengths=domain_lengths, **kwargs)
        if self.S_bulk is None:
            raise ValueError("S_bulk is required for core moment tracking.")
        self.target_core_moment = self._core_moment(self.target_q).detach()

    def _core_moment(self, q):
        density = self._core_density(q)
        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        if mask.shape[0] == 1 and density.shape[0] > 1:
            mask = mask.expand(density.shape[0], *mask.shape[1:])
        weights = density * mask
        spatial_dims = tuple(range(1, weights.ndim))
        mass = weights.sum(dim=spatial_dims, dtype=torch.float64).clamp_min(1e-12)

        axes = [
            (torch.arange(size, device=q.device, dtype=q.dtype) + 0.5)
            * length
            / size
            for size, length in zip(q.shape[2:], self.domain_lengths)
        ]
        theta_x = 2.0 * torch.pi * axes[0] / self.domain_lengths[0]
        weight_x = weights.sum(dim=(2, 3), dtype=torch.float64)
        sine = (weight_x * torch.sin(theta_x)).sum(dim=1)
        cosine = (weight_x * torch.cos(theta_x)).sum(dim=1)
        center_theta = torch.atan2(sine, cosine)

        weight_y = weights.sum(dim=(1, 3), dtype=torch.float64)
        weight_z = weights.sum(dim=(1, 2), dtype=torch.float64)
        center_y = (weight_y * axes[1]).sum(dim=1) / mass
        center_z = (weight_z * axes[2]).sum(dim=1) / mass

        angle_delta = theta_x.unsqueeze(0) - center_theta.unsqueeze(1)
        dx = (
            torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )[:, :, None, None]
        dy = (axes[1].unsqueeze(0) - center_y.unsqueeze(1))[:, None, :, None]
        dz = (axes[2].unsqueeze(0) - center_z.unsqueeze(1))[:, None, None, :]
        deltas = (dx, dy, dz)

        rows = []
        for delta_i in deltas:
            rows.append(torch.stack([
                (weights * delta_i * delta_j).sum(
                    dim=spatial_dims, dtype=torch.float64
                ) / mass
                for delta_j in deltas
            ], dim=1))
        return torch.stack(rows, dim=1)

    def _moment_penalty(self, q):
        moment = self._core_moment(q)
        target = self.target_core_moment.to(device=q.device, dtype=moment.dtype)
        if target.shape[0] == 1 and moment.shape[0] > 1:
            target = target.expand(moment.shape[0], -1, -1)
        denominator = target.square().sum(dim=(1, 2)).clamp_min(1e-12)
        return ((moment - target).square().sum(dim=(1, 2)) / denominator).mean()

    def components(self, q, alpha=None):
        components = super().components(q, alpha)
        components["core_moment_penalty"] = self._moment_penalty(q)
        return components

    def stage_cost(self, q, alpha, step):
        del step
        components = self.components(q, alpha)
        cost = (
            self.running_weight * components["q_tracking_mse"]
            + self.running_core_weight * components["core_tracking_mse"]
            + self.running_core_mass_weight * components["core_mass_penalty"]
            + self.running_moment_weight * components["core_moment_penalty"]
            + self.control_weight * components["control_effort"]
        )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q)
        cost = (
            self.terminal_weight * components["q_tracking_mse"]
            + self.terminal_core_weight * components["core_tracking_mse"]
            + self.terminal_core_mass_weight * components["core_mass_penalty"]
            + self.terminal_moment_weight * components["core_moment_penalty"]
        )
        return 0.5 * cost


class CoreAwareAlignedShapeObjective(CoreAwareMomentObjective):
    """Core-aware tracking with translation-aligned full-density shape matching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_density = self.target_core_density
        self.target_core_center_x = self._core_center_x(target_density).detach()

    def _core_center_x(self, density):
        mask = self.spatial_mask.to(device=density.device, dtype=density.dtype)
        weights = density * mask.unsqueeze(0)
        nx = density.shape[1]
        x = (
            torch.arange(nx, device=density.device, dtype=density.dtype) + 0.5
        ) * self.domain_lengths[0] / nx
        theta = 2.0 * torch.pi * x / self.domain_lengths[0]
        weight_x = weights.sum(dim=(2, 3), dtype=torch.float64)
        sine = (weight_x * torch.sin(theta)).sum(dim=1)
        cosine = (weight_x * torch.cos(theta)).sum(dim=1)
        return (
            torch.remainder(torch.atan2(sine, cosine), 2.0 * torch.pi)
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )

    def _aligned_shape_penalty(self, q):
        density = self._core_density(q)
        target = self.target_core_density.to(device=q.device, dtype=q.dtype)
        if target.shape[0] == 1 and density.shape[0] > 1:
            target = target.expand(density.shape[0], *target.shape[1:])
        target_center = self.target_core_center_x.to(
            device=q.device, dtype=torch.float64
        )
        if target_center.shape[0] == 1 and density.shape[0] > 1:
            target_center = target_center.expand(density.shape[0])
        center = self._core_center_x(density)
        displacement = center - target_center
        displacement = (
            torch.atan2(
                torch.sin(2.0 * torch.pi * displacement / self.domain_lengths[0]),
                torch.cos(2.0 * torch.pi * displacement / self.domain_lengths[0]),
            )
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )

        frequencies = torch.fft.fftfreq(
            density.shape[1],
            d=self.domain_lengths[0] / density.shape[1],
            device=q.device,
            dtype=q.dtype,
        )
        phase = torch.exp(
            2j * torch.pi * displacement.to(q.dtype).unsqueeze(1) * frequencies
        )
        aligned = torch.fft.ifft(
            torch.fft.fft(density, dim=1) * phase[:, :, None, None],
            dim=1,
        ).real

        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        spatial_dims = tuple(range(1, density.ndim))
        candidate_mass = (aligned * mask).sum(
            dim=spatial_dims, dtype=torch.float64
        ).clamp_min(1e-12)
        target_mass = (target * mask).sum(
            dim=spatial_dims, dtype=torch.float64
        ).clamp_min(1e-12)
        aligned = aligned * (target_mass / candidate_mass).to(q.dtype)[:, None, None, None]
        numerator = ((aligned - target).square() * mask).sum(
            dim=spatial_dims, dtype=torch.float64
        )
        denominator = (target.square() * mask).sum(
            dim=spatial_dims, dtype=torch.float64
        ).clamp_min(1e-12)
        return (numerator / denominator).mean()

    def components(self, q, alpha=None):
        components = super().components(q, alpha)
        components["aligned_core_shape_penalty"] = self._aligned_shape_penalty(q)
        return components

    def stage_cost(self, q, alpha, step):
        del step
        components = self.components(q, alpha)
        cost = (
            self.running_weight * components["q_tracking_mse"]
            + self.running_core_weight * components["core_tracking_mse"]
            + self.running_core_mass_weight * components["core_mass_penalty"]
            + self.running_moment_weight * components["core_moment_penalty"]
            + self.running_aligned_shape_weight
            * components["aligned_core_shape_penalty"]
            + self.control_weight * components["control_effort"]
        )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q)
        cost = (
            self.terminal_weight * components["q_tracking_mse"]
            + self.terminal_core_weight * components["core_tracking_mse"]
            + self.terminal_core_mass_weight * components["core_mass_penalty"]
            + self.terminal_moment_weight * components["core_moment_penalty"]
            + self.terminal_aligned_shape_weight
            * components["aligned_core_shape_penalty"]
        )
        return 0.5 * cost


class CoreAwareComovingQTrajectoryObjective(CoreAwareAlignedShapeObjective):
    """Track a smooth translation while preserving the initial loop Q texture."""

    def __init__(
        self,
        *args,
        initial_q=None,
        num_steps=None,
        running_comoving_q_weight=0.0,
        terminal_comoving_q_weight=0.0,
        **kwargs,
    ):
        if initial_q is None:
            raise ValueError("initial_q is required for comoving-Q trajectory tracking.")
        if num_steps is None or int(num_steps) <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if running_comoving_q_weight < 0 or terminal_comoving_q_weight < 0:
            raise ValueError("Comoving-Q weights must be non-negative.")
        super().__init__(*args, **kwargs)
        self.initial_q = torch.as_tensor(
            initial_q,
            device=self.target_q.device,
            dtype=self.target_q.dtype,
        )
        if self.initial_q.shape != self.target_q.shape:
            raise ValueError("initial_q and target_q must have identical shapes.")
        self.num_steps = int(num_steps)
        self.running_comoving_q_weight = float(running_comoving_q_weight)
        self.terminal_comoving_q_weight = float(terminal_comoving_q_weight)
        self.initial_core_density = self._core_density(self.initial_q).detach()
        self.initial_translation_density = self._translation_density(
            self.initial_q
        ).detach()
        self.target_translation_density = self._translation_density(
            self.target_q
        ).detach()
        self.initial_core_center = self._core_center(
            self.initial_translation_density
        ).detach()
        self.target_trajectory_center = self._core_center(
            self.target_translation_density
        ).detach()
        self.initial_core_moment = self._core_moment(self.initial_q).detach()

    def _translation_density(self, q):
        """Use the full loop disturbance to resolve sub-grid translations."""

        uniform = torch.zeros_like(q)
        uniform[0] = self.S_bulk
        uniform[3] = -self.S_bulk / 2.0
        return (q - uniform).square().sum(dim=0)

    def _core_center(self, density):
        mask = self.spatial_mask.to(device=density.device, dtype=density.dtype)
        weights = density * mask.unsqueeze(0)
        mass = weights.sum(dim=(1, 2, 3), dtype=torch.float64).clamp_min(1e-12)
        axes = [
            (torch.arange(size, device=density.device, dtype=density.dtype) + 0.5)
            * length
            / size
            for size, length in zip(density.shape[1:], self.domain_lengths)
        ]
        theta_x = 2.0 * torch.pi * axes[0] / self.domain_lengths[0]
        weight_x = weights.sum(dim=(2, 3), dtype=torch.float64)
        center_x = (
            torch.remainder(
                torch.atan2(
                    (weight_x * torch.sin(theta_x)).sum(dim=1),
                    (weight_x * torch.cos(theta_x)).sum(dim=1),
                ),
                2.0 * torch.pi,
            )
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )
        center_y = (
            weights.sum(dim=(1, 3), dtype=torch.float64) * axes[1]
        ).sum(dim=1) / mass
        center_z = (
            weights.sum(dim=(1, 2), dtype=torch.float64) * axes[2]
        ).sum(dim=1) / mass
        return torch.stack((center_x, center_y, center_z), dim=1)

    def _candidate_center(self, q):
        return self._core_center(self._translation_density(q))

    def _periodic_x_delta(self, delta):
        angle = 2.0 * torch.pi * delta / self.domain_lengths[0]
        return (
            torch.atan2(torch.sin(angle), torch.cos(angle))
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )

    def _reference_center(self, step):
        tau = torch.as_tensor(
            min(max(float(step) / self.num_steps, 0.0), 1.0),
            device=self.target_q.device,
            dtype=torch.float64,
        )
        progress = tau.square() * (3.0 - 2.0 * tau)
        displacement = self.target_trajectory_center - self.initial_core_center
        displacement = torch.cat(
            (
                self._periodic_x_delta(displacement[:, :1]),
                displacement[:, 1:],
            ),
            dim=1,
        )
        center = self.initial_core_center + progress * displacement
        center_x = torch.remainder(center[:, :1], self.domain_lengths[0])
        return torch.cat((center_x, center[:, 1:]), dim=1)

    def _trajectory_position_penalty(self, q, step):
        center = self._candidate_center(q)
        reference = self._reference_center(step).to(
            device=q.device, dtype=center.dtype
        )
        if reference.shape[0] == 1 and center.shape[0] > 1:
            reference = reference.expand(center.shape[0], -1)
        delta = center - reference
        delta = torch.cat(
            (self._periodic_x_delta(delta[:, :1]), delta[:, 1:]),
            dim=1,
        )
        return delta.square().sum(dim=1).mean()

    def _align_x_to_initial(self, field, center_x):
        initial_center_x = self.initial_core_center[:, 0].to(
            device=field.device, dtype=center_x.dtype
        )
        if initial_center_x.shape[0] == 1 and field.shape[1] > 1:
            initial_center_x = initial_center_x.expand(field.shape[1])
        displacement = self._periodic_x_delta(center_x - initial_center_x)
        frequencies = torch.fft.fftfreq(
            field.shape[2],
            d=self.domain_lengths[0] / field.shape[2],
            device=field.device,
            dtype=field.dtype,
        )
        phase = torch.exp(
            2j * torch.pi * displacement.to(field.dtype).unsqueeze(1) * frequencies
        )
        spectrum = torch.fft.fft(field, dim=2)
        return torch.fft.ifft(
            spectrum * phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dim=2,
        ).real

    def _comoving_q_penalty(self, q):
        center_x = self._candidate_center(q)[:, 0]
        aligned = self._align_x_to_initial(q, center_x)
        reference = self.initial_q.to(device=q.device, dtype=q.dtype)
        if reference.shape[1] == 1 and q.shape[1] > 1:
            reference = reference.expand(-1, q.shape[1], -1, -1, -1)
        initial_density = self.initial_core_density.to(device=q.device, dtype=q.dtype)
        weights = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        weights = weights * (
            1.0 + self.target_core_q_weight * initial_density
        )
        numerator = ((aligned - reference).square() * weights).sum(
            dim=(0, 2, 3, 4), dtype=torch.float64
        )
        uniform = torch.zeros_like(reference)
        uniform[0] = self.S_bulk
        uniform[3] = -self.S_bulk / 2.0
        denominator = ((reference - uniform).square() * weights).sum(
            dim=(0, 2, 3, 4), dtype=torch.float64
        ).clamp_min(1e-12)
        return (numerator / denominator).mean()

    def _initial_moment_penalty(self, q):
        moment = self._core_moment(q)
        target = self.initial_core_moment.to(device=q.device, dtype=moment.dtype)
        if target.shape[0] == 1 and moment.shape[0] > 1:
            target = target.expand(moment.shape[0], -1, -1)
        denominator = target.square().sum(dim=(1, 2)).clamp_min(1e-12)
        return ((moment - target).square().sum(dim=(1, 2)) / denominator).mean()

    def components(self, q, alpha=None, step=None):
        if step is None:
            step = self.num_steps
        base = QTrackingObjective.components(self, q, alpha)
        base["trajectory_position_penalty"] = self._trajectory_position_penalty(
            q, step
        )
        base["core_moment_penalty"] = self._initial_moment_penalty(q)
        base["aligned_core_shape_penalty"] = self._aligned_shape_penalty(q)
        base["comoving_q_shape_penalty"] = self._comoving_q_penalty(q)
        return base

    def stage_cost(self, q, alpha, step):
        components = self.components(q, alpha, step=step)
        cost = (
            self.running_weight * components["trajectory_position_penalty"]
            + self.running_core_mass_weight * components["core_mass_penalty"]
            + self.running_moment_weight * components["core_moment_penalty"]
            + self.running_aligned_shape_weight
            * components["aligned_core_shape_penalty"]
            + self.running_comoving_q_weight
            * components["comoving_q_shape_penalty"]
            + self.control_weight * components["control_effort"]
        )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q, step=self.num_steps)
        cost = (
            self.terminal_weight * components["trajectory_position_penalty"]
            + self.terminal_core_mass_weight * components["core_mass_penalty"]
            + self.terminal_moment_weight * components["core_moment_penalty"]
            + self.terminal_aligned_shape_weight
            * components["aligned_core_shape_penalty"]
            + self.terminal_comoving_q_weight
            * components["comoving_q_shape_penalty"]
        )
        return 0.5 * cost


class FreePathShapePreservingObjective(QTrackingObjective):
    """Reach a terminal core center without prescribing the intervening path."""

    def __init__(
        self,
        *args,
        initial_q=None,
        position_tolerance=0.25,
        mass_tolerance=0.05,
        moment_tolerance=0.05,
        q_shape_tolerance=0.05,
        free_running_mass_weight=0.02,
        free_terminal_mass_weight=0.2,
        free_running_moment_weight=0.02,
        free_terminal_moment_weight=0.2,
        free_running_q_weight=0.05,
        free_terminal_q_weight=0.5,
        **kwargs,
    ):
        if initial_q is None:
            raise ValueError("initial_q is required for free-path shape preservation.")
        tolerances = (
            position_tolerance,
            mass_tolerance,
            moment_tolerance,
            q_shape_tolerance,
        )
        if any(float(value) <= 0 for value in tolerances):
            raise ValueError("Free-path normalization tolerances must be positive.")
        free_weights = (
            free_running_mass_weight,
            free_terminal_mass_weight,
            free_running_moment_weight,
            free_terminal_moment_weight,
            free_running_q_weight,
            free_terminal_q_weight,
        )
        if any(float(value) < 0 for value in free_weights):
            raise ValueError("Free-path objective weights must be non-negative.")
        super().__init__(*args, **kwargs)
        if self.domain_lengths is None:
            raise ValueError("domain_lengths is required for free-path tracking.")
        if self.S_bulk is None:
            raise ValueError("S_bulk is required for free-path tracking.")
        self.initial_q = torch.as_tensor(
            initial_q, device=self.target_q.device, dtype=self.target_q.dtype
        )
        if self.initial_q.shape != self.target_q.shape:
            raise ValueError("initial_q and target_q must have identical shapes.")
        self.position_tolerance = float(position_tolerance)
        self.mass_tolerance = float(mass_tolerance)
        self.moment_tolerance = float(moment_tolerance)
        self.q_shape_tolerance = float(q_shape_tolerance)
        self.free_running_mass_weight = float(free_running_mass_weight)
        self.free_terminal_mass_weight = float(free_terminal_mass_weight)
        self.free_running_moment_weight = float(free_running_moment_weight)
        self.free_terminal_moment_weight = float(free_terminal_moment_weight)
        self.free_running_q_weight = float(free_running_q_weight)
        self.free_terminal_q_weight = float(free_terminal_q_weight)

        self.initial_core_density = self._core_density(self.initial_q).detach()
        self.target_core_density_free = self._core_density(self.target_q).detach()
        self.initial_center, self.initial_mass = self._center_and_mass(
            self.initial_core_density
        )
        self.target_center, _ = self._center_and_mass(self.target_core_density_free)
        self.initial_center = self.initial_center.detach()
        self.target_center = self.target_center.detach()
        self.initial_mass = self.initial_mass.detach()
        self.initial_moment = self._central_moment(
            self.initial_core_density, self.initial_center, self.initial_mass
        ).detach()

    def _axes(self, tensor):
        return [
            (torch.arange(size, device=tensor.device, dtype=tensor.dtype) + 0.5)
            * length
            / size
            for size, length in zip(tensor.shape[1:], self.domain_lengths)
        ]

    def _periodic_x_delta(self, delta):
        angle = 2.0 * torch.pi * delta / self.domain_lengths[0]
        return (
            torch.atan2(torch.sin(angle), torch.cos(angle))
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )

    def _center_and_mass(self, density):
        # Core geometry is a state observable and must not depend on actuator support.
        weights = density
        mass = weights.sum(dim=(1, 2, 3), dtype=torch.float64).clamp_min(1e-12)
        x, y, z = self._axes(density)
        theta_x = 2.0 * torch.pi * x / self.domain_lengths[0]
        weight_x = weights.sum(dim=(2, 3), dtype=torch.float64)
        center_x = (
            torch.remainder(
                torch.atan2(
                    (weight_x * torch.sin(theta_x)).sum(dim=1),
                    (weight_x * torch.cos(theta_x)).sum(dim=1),
                ),
                2.0 * torch.pi,
            )
            * self.domain_lengths[0]
            / (2.0 * torch.pi)
        )
        center_y = (
            weights.sum(dim=(1, 3), dtype=torch.float64) * y
        ).sum(dim=1) / mass
        center_z = (
            weights.sum(dim=(1, 2), dtype=torch.float64) * z
        ).sum(dim=1) / mass
        return torch.stack((center_x, center_y, center_z), dim=1), mass

    def _central_moment(self, density, center, mass):
        weights = density
        x, y, z = self._axes(density)
        dx = self._periodic_x_delta(x.unsqueeze(0) - center[:, :1])[:, :, None, None]
        dy = (y.unsqueeze(0) - center[:, 1:2])[:, None, :, None]
        dz = (z.unsqueeze(0) - center[:, 2:3])[:, None, None, :]
        deltas = (dx, dy, dz)
        rows = []
        for delta_i in deltas:
            rows.append(
                torch.stack(
                    [
                        (weights * delta_i * delta_j).sum(
                            dim=(1, 2, 3), dtype=torch.float64
                        )
                        / mass
                        for delta_j in deltas
                    ],
                    dim=1,
                )
            )
        return torch.stack(rows, dim=1)

    def _align_q_to_initial(self, q, center):
        initial_center = self.initial_center.to(device=q.device, dtype=center.dtype)
        if initial_center.shape[0] == 1 and center.shape[0] > 1:
            initial_center = initial_center.expand(center.shape[0], -1)
        displacement = center - initial_center
        displacement_x = self._periodic_x_delta(displacement[:, 0])
        frequencies = torch.fft.fftfreq(
            q.shape[2],
            d=self.domain_lengths[0] / q.shape[2],
            device=q.device,
            dtype=q.dtype,
        )
        phase = torch.exp(
            2j * torch.pi * displacement_x.to(q.dtype).unsqueeze(1) * frequencies
        )
        aligned_x = torch.fft.ifft(
            torch.fft.fft(q, dim=2) * phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dim=2,
        ).real

        bulk = torch.zeros_like(aligned_x)
        bulk[0] = self.S_bulk
        bulk[3] = -self.S_bulk / 2.0
        disturbance = (aligned_x - bulk).permute(1, 0, 2, 3, 4)
        batch, _, nx, ny, nz = disturbance.shape
        grid_x = (torch.arange(nx, device=q.device, dtype=q.dtype) + 0.5) * 2.0 / nx - 1.0
        grid_y = (torch.arange(ny, device=q.device, dtype=q.dtype) + 0.5) * 2.0 / ny - 1.0
        grid_z = (torch.arange(nz, device=q.device, dtype=q.dtype) + 0.5) * 2.0 / nz - 1.0
        gx, gy, gz = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        sample_grid = torch.stack((gz, gy, gx), dim=-1).unsqueeze(0).expand(
            batch, nx, ny, nz, 3
        ).clone()
        sample_grid[..., 0] += (
            2.0 * displacement[:, 2].to(q.dtype) / self.domain_lengths[2]
        )[:, None, None, None]
        sample_grid[..., 1] += (
            2.0 * displacement[:, 1].to(q.dtype) / self.domain_lengths[1]
        )[:, None, None, None]
        aligned = torch_functional.grid_sample(
            disturbance,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(1, 0, 2, 3, 4)
        return aligned + bulk

    def _free_path_metrics(self, q):
        density = self._core_density(q)
        center, mass = self._center_and_mass(density)
        target_center = self.target_center.to(device=q.device, dtype=center.dtype)
        initial_mass = self.initial_mass.to(device=q.device, dtype=mass.dtype)
        if target_center.shape[0] == 1 and center.shape[0] > 1:
            target_center = target_center.expand(center.shape[0], -1)
            initial_mass = initial_mass.expand(mass.shape[0])
        center_delta = center - target_center
        center_delta = torch.cat(
            (self._periodic_x_delta(center_delta[:, :1]), center_delta[:, 1:]), dim=1
        )
        position = center_delta.square().sum(dim=1).mean() / self.position_tolerance**2
        mass_penalty = (
            (mass / initial_mass.clamp_min(1e-12) - 1.0).square().mean()
            / self.mass_tolerance**2
        )
        moment = self._central_moment(density, center, mass)
        initial_moment = self.initial_moment.to(device=q.device, dtype=moment.dtype)
        if initial_moment.shape[0] == 1 and moment.shape[0] > 1:
            initial_moment = initial_moment.expand(moment.shape[0], -1, -1)
        moment_scale = initial_moment.square().sum(dim=(1, 2)).clamp_min(1e-12)
        moment_penalty = (
            ((moment - initial_moment).square().sum(dim=(1, 2)) / moment_scale).mean()
            / self.moment_tolerance**2
        )

        aligned = self._align_q_to_initial(q, center)
        reference = self.initial_q.to(device=q.device, dtype=q.dtype)
        if reference.shape[1] == 1 and q.shape[1] > 1:
            reference = reference.expand(-1, q.shape[1], -1, -1, -1)
        initial_density = self.initial_core_density.to(device=q.device, dtype=q.dtype)
        mask = self.spatial_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        weights = mask * (0.05 + self.target_core_q_weight * initial_density)
        numerator = ((aligned - reference).square() * weights).sum(
            dim=(0, 2, 3, 4), dtype=torch.float64
        )
        bulk = torch.zeros_like(reference)
        bulk[0] = self.S_bulk
        bulk[3] = -self.S_bulk / 2.0
        denominator = ((reference - bulk).square() * weights).sum(
            dim=(0, 2, 3, 4), dtype=torch.float64
        ).clamp_min(1e-12)
        q_shape = (numerator / denominator).mean() / self.q_shape_tolerance**2
        return position, mass_penalty, moment_penalty, q_shape

    def components(self, q, alpha=None, step=None):
        del step
        position, mass, moment, q_shape = self._free_path_metrics(q)
        effort = (
            q.sum(dtype=torch.float64) * 0.0
            if alpha is None
            else (alpha - self.reference_alpha).square().mean(dtype=torch.float64)
        )
        return {
            "endpoint_position_penalty": position,
            "core_mass_penalty": mass,
            "core_moment_penalty": moment,
            "comoving_q_shape_penalty": q_shape,
            "control_effort": effort,
        }

    def stage_cost(self, q, alpha, step):
        components = self.components(q, alpha, step=step)
        cost = (
            self.free_running_mass_weight * components["core_mass_penalty"]
            + self.free_running_moment_weight * components["core_moment_penalty"]
            + self.free_running_q_weight * components["comoving_q_shape_penalty"]
            + self.control_weight * components["control_effort"]
        )
        return 0.5 * self.dt * cost

    def terminal_cost(self, q):
        components = self.components(q)
        cost = (
            self.terminal_weight * components["endpoint_position_penalty"]
            + self.free_terminal_mass_weight * components["core_mass_penalty"]
            + self.free_terminal_moment_weight * components["core_moment_penalty"]
            + self.free_terminal_q_weight * components["comoving_q_shape_penalty"]
        )
        return 0.5 * cost


LOSS_FUNCTIONS = {
    "core_aware_aligned_shape_tracking": CoreAwareAlignedShapeObjective,
    "core_aware_comoving_q_trajectory": CoreAwareComovingQTrajectoryObjective,
    "core_aware_moment_tracking": CoreAwareMomentObjective,
    "core_aware_q_tracking": QTrackingObjective,
    "core_translation": CoreTranslationObjective,
    "core_translation_no_mass": CoreTranslationNoMassObjective,
    "free_path_shape_preserving": FreePathShapePreservingObjective,
}


def available_loss_functions():
    """Return the registered DAL loss-function names."""

    return tuple(sorted(LOSS_FUNCTIONS))


def build_loss_function(name, **kwargs):
    """Construct a registered DAL objective implementing the loss interface."""

    try:
        objective_type = LOSS_FUNCTIONS[name]
    except KeyError as error:
        available = ", ".join(available_loss_functions())
        raise ValueError(
            f"Unknown loss function {name!r}. Available loss functions: {available}."
        ) from error
    objective = objective_type(**kwargs)
    required_methods = ("stage_cost", "terminal_cost", "components")
    missing = [
        method
        for method in required_methods
        if not callable(getattr(objective, method, None))
    ]
    if missing:
        raise TypeError(
            f"Loss function {name!r} is missing required methods: {', '.join(missing)}."
        )
    return objective


__all__ = [
    "CoreAwareAlignedShapeObjective",
    "CoreAwareComovingQTrajectoryObjective",
    "CoreAwareMomentObjective",
    "CoreTranslationNoMassObjective",
    "CoreTranslationObjective",
    "FreePathShapePreservingObjective",
    "LOSS_FUNCTIONS",
    "QTrackingObjective",
    "available_loss_functions",
    "build_loss_function",
    "load_q_target",
]
