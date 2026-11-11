"""Internal complete-stress model preset for the legacy Periodic translation."""

from pssolver.core.terms import (
    ConstrainedField,
    Constraint,
    EvolvedField,
    Explicit,
    Linear,
    Model,
    PackageTerm,
    Parameter,
    Recipe,
)


class CompleteStressBerisEdwards(Model):
    """Declare the exact S-25 Beris--Edwards plan without importing tensors."""

    Q = EvolvedField(("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"))
    velocity = ConstrainedField(("ux", "uy", "uz"))
    pressure = ConstrainedField(("p",))
    alpha = Parameter()

    def __init__(self, *, material, ldg_l1, stokes, distortion_odd_boundary_conditions=None, **parameters):
        super().__init__(**parameters)
        self.material = material
        self.ldg_l1 = ldg_l1
        self.stokes = stokes
        self.distortion_odd_boundary_conditions = distortion_odd_boundary_conditions

    def terms(self, f, p):
        material = self.material
        return (
            Linear(
                f.Q,
                diagonal=Recipe(
                    "beris_edwards",
                    ldg_a=material["ldg_a"],
                    ldg_l1=self.ldg_l1,
                    rotational_viscosity=material["gamma"],
                ),
            ),
            Explicit(
                f.Q,
                PackageTerm("beris_edwards_q"),
                arguments={
                    "ldg_b": material["ldg_b"],
                    "ldg_c": material["ldg_c"],
                    "rotational_viscosity": material["gamma"],
                    "flow_alignment": material["flow_alignment"],
                },
            ),
            Constraint(
                "stokes",
                inputs=(f.Q,),
                outputs=(f.velocity, f.pressure),
                arguments={
                    "beta_value": material["beta"],
                    "friction": self.stokes["friction"],
                    "viscosity": self.stokes["viscosity"],
                    "ldg_a": material["ldg_a"],
                    "ldg_b": material["ldg_b"],
                    "ldg_c": material["ldg_c"],
                    "ldg_l1": self.ldg_l1,
                    "flow_alignment": material["flow_alignment"],
                    "zero_mode_policy": self.stokes["tangential_zero_mode_policy"],
                    **({"distortion_odd_boundary_conditions": self.distortion_odd_boundary_conditions}
                       if self.distortion_odd_boundary_conditions is not None else {}),
                },
                execution_options=("molecular_field_linear_space", "stress_divergence_sum_space"),
            ),
        )
