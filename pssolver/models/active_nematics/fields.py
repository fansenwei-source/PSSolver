"""Field names shared by active-nematic model components."""

Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")

CHANNEL_Q_BOUNDARY_CONDITIONS = (
    "periodic",
    "neumann",
    "neumann",
)

# Row-major parity of -L1 * (partial_i Q_ab)(partial_j Q_ab). Products of
# equal sine parities are Neumann; one y or z derivative contributes the
# corresponding Dirichlet parity.
CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS = (
    ("periodic", "neumann", "neumann"),
    ("periodic", "dirichlet", "neumann"),
    ("periodic", "neumann", "dirichlet"),
    ("periodic", "dirichlet", "neumann"),
    ("periodic", "neumann", "neumann"),
    ("periodic", "dirichlet", "dirichlet"),
    ("periodic", "neumann", "dirichlet"),
    ("periodic", "dirichlet", "dirichlet"),
    ("periodic", "neumann", "neumann"),
)

__all__ = [
    "CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS",
    "CHANNEL_Q_BOUNDARY_CONDITIONS",
    "Q_COMPONENTS",
]
