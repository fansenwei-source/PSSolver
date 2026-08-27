import torch


def active_force_divergence(fields, alpha, beta):
    """Return ``div(beta * alpha * Q)`` for the five-component 3D Q tensor."""

    qxx = fields["Qxx"]
    qxy = fields["Qxy"]
    qxz = fields["Qxz"]
    qyy = fields["Qyy"]
    qyz = fields["Qyz"]
    qzz = -qxx - qyy

    force_x = beta * (
        fields.gradient("Qxx", axis=0, tensor=alpha * qxx)
        + fields.gradient("Qxy", axis=1, tensor=alpha * qxy)
        + fields.gradient("Qxz", axis=2, tensor=alpha * qxz)
    )
    force_y = beta * (
        fields.gradient("Qxy", axis=0, tensor=alpha * qxy)
        + fields.gradient("Qyy", axis=1, tensor=alpha * qyy)
        + fields.gradient("Qyz", axis=2, tensor=alpha * qyz)
    )
    force_z = beta * (
        fields.gradient("Qxz", axis=0, tensor=alpha * qxz)
        + fields.gradient("Qyz", axis=1, tensor=alpha * qyz)
        + fields.gradient("Qxx", axis=2, tensor=alpha * qzz)
    )
    return torch.stack((force_x, force_y, force_z))
