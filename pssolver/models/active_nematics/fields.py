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

STRESS_COMPONENTS = (
    "xx",
    "xy",
    "xz",
    "yx",
    "yy",
    "yz",
    "zx",
    "zy",
    "zz",
)


BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY = (
    "beris_edwards_molecular_field"
)


BERIS_EDWARDS_Q_GRADIENT_CAPABILITY = "beris_edwards_q_gradient"


BERIS_EDWARDS_STRESS_CAPABILITY = "beris_edwards_stress"


BERIS_EDWARDS_FORCE_CAPABILITY = "beris_edwards_force"


BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY = (
    "beris_edwards_velocity_gradient"
)


H_COMPONENTS = ("Hxx", "Hxy", "Hxz", "Hyy", "Hyz")


CARTESIAN_AXES = ("x", "y", "z")


Q_GRADIENT_COMPONENTS = tuple(
    f"d{component}_d{axis}"
    for axis in CARTESIAN_AXES
    for component in Q_COMPONENTS
)


ALGEBRAIC_STRESS_COMPONENTS = tuple(
    f"stress_algebraic_{component}" for component in STRESS_COMPONENTS
)


DISTORTION_STRESS_COMPONENTS = tuple(
    f"stress_distortion_{component}" for component in STRESS_COMPONENTS
)


NEMATIC_FORCE_COMPONENTS = ("force_x", "force_y", "force_z")


VELOCITY_COMPONENTS = ("ux", "uy", "uz")


VELOCITY_GRADIENT_COMPONENTS = tuple(
    f"d{component}_d{axis}"
    for axis in CARTESIAN_AXES
    for component in VELOCITY_COMPONENTS
)

__all__ = [
    "ALGEBRAIC_STRESS_COMPONENTS",
    "BERIS_EDWARDS_FORCE_CAPABILITY",
    "BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY",
    "BERIS_EDWARDS_Q_GRADIENT_CAPABILITY",
    "BERIS_EDWARDS_STRESS_CAPABILITY",
    "BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY",
    "CARTESIAN_AXES",
    "CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS",
    "CHANNEL_Q_BOUNDARY_CONDITIONS",
    "DISTORTION_STRESS_COMPONENTS",
    "H_COMPONENTS",
    "NEMATIC_FORCE_COMPONENTS",
    "Q_COMPONENTS",
    "Q_GRADIENT_COMPONENTS",
    "STRESS_COMPONENTS",
    "VELOCITY_COMPONENTS",
    "VELOCITY_GRADIENT_COMPONENTS",
]
