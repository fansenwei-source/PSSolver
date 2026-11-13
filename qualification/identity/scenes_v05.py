"""Map the sealed S-23 scenarios to S-28b declarations without changing their inputs."""

import numpy as np

import pssolver as pub


CSBE = dict(
    ldg_a=0.0,
    ldg_b=-0.3,
    ldg_c=0.3,
    ldg_l1=1 / 81,
    gamma=2.94,
    flow_alignment=0.3,
    activity=0.01,
    beta=-1.0,
    viscosity=2 / 3,
)
GENERATED = dict(
    seed=24,
    num_defect_pairs=6,
    defect_min_separation=10.0,
    defect_core_radius=1.5,
    background_angle=0.0,
    twist_amplitude=0.01,
    twist_modes=[1, 2, 3],
    initial_s=1 / 3,
)


def initial_values(scene):  # verbatim S-23 qualification/identity/runtime_case.py:initial_values
    shape = tuple(scene["shape"])
    grid = np.indices(shape, dtype=np.float64)
    wave = sum(np.cos(2 * np.pi * grid[i] / shape[i]) for i in range(3))
    values = np.stack([(i + 1) * 1e-4 * wave for i in range(5)], axis=-1)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    return values.astype(np.dtype(scene["dtype"]), copy=False)


def snapshot_dir(scene, root):
    path = root / "initial"
    if not (path / "Q_0.npy").exists():
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "Q_0.npy", initial_values(scene), allow_pickle=False)
    return path


def simulation_v05(scene, root, *, output_dir=None, steps=None, restart_from=None):
    """S-23 scene -> 0.5 public declaration (what qualification/identity/scenes_v05.py has to do)."""
    family, o, dtype = scene["family"], dict(scene["options"]), scene["dtype"]
    shape = tuple(scene["shape"])
    steps = scene["steps"] if steps is None else steps
    output_dir = root / "output" if output_dir is None else output_dir
    if family == "plane":
        model = pub.ShendrukPlaneBerisEdwards(
            activity_number=18.0,
            parameterization="fixed-k",
            frank_k=1 / 81,
            **{k: o[k] for k in ("zero_mode_policy", "friction_mode_fric") if k in o},
        )
        geometry = pub.PlaneSlab(shape=shape, lengths=(16.0, 16.0, 12.0))
        boundaries = {
            "Q": pub.neumann_q(),
            "velocity": pub.free_slip_velocity(),
            "pressure": pub.neumann_pressure_compatibility(),
        }
        numerics = pub.SpectralNumerics(
            dtype=dtype,
            dealias_rule=o.get("dealias_rule", "cubic_half"),
            projected_transform_execution=o.get("projected_transform_execution", "truncated"),
            spectral_storage="hermitian_half",
            hermitian_axis=1,
            transform_execution_order="real_first",
        )
        refresh = (
            {"steps": o["spectral_refresh_steps"]}
            if o.get("spectral_refresh_steps")
            else {"physical_time": 0.2}
        )
        time = pub.SemiImplicitEuler(dt=0.001, refresh=refresh)
        initial = pub.GeneratedInitialCondition("extruded_defect_gas", parameters=GENERATED)
        options = dict(
            tf32="off",
            molecular_field_linear_space=o.get("molecular_field_linear_space", "spectral"),
            stress_divergence_sum_space=o.get("stress_divergence_sum_space", "spectral"),
            pointwise_execution=o.get("pointwise_execution", "eager"),
            disable_q_gradient_reuse=o.get("disable_q_gradient_reuse", False),
        )
        diagnostics = False
    else:
        channel = family == "channel"
        model = pub.CompleteStressBerisEdwards(**CSBE)
        geometry = (pub.RectangularChannel if channel else pub.PeriodicBox)(
            shape=shape, lengths=(16.0, 12.0, 12.0)
        )
        boundaries = {
            "Q": pub.neumann_q(),
            "velocity": pub.no_slip_velocity() if channel else pub.free_slip_velocity(),
            "pressure": pub.neumann_pressure_compatibility(),
        }
        extra = {}
        if channel:
            extra["pressure_solver"] = {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": o.get("relative_tolerance", 1e-10),
                "max_iterations": 40,
                "fixed_iterations": o.get("fixed_iterations", 12),
                "warm_start": o.get("warm_start", True),
            }
        numerics = pub.SpectralNumerics(
            dtype=dtype,
            dealias_rule=o.get("dealias_rule", "cubic_half"),
            projected_transform_execution=o.get("projected_transform_execution", "truncated"),
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
            **extra,
        )
        time = pub.SemiImplicitEuler(dt=0.001)
        initial = pub.SnapshotInitialCondition(snapshot_dir(scene, root), step=0)
        options = dict(
            tf32="off",
            molecular_field_linear_space="spectral",
            stress_divergence_sum_space="physical" if channel else "spectral",
            pointwise_execution=o.get("pointwise_execution", "eager"),
            disable_q_gradient_reuse=o.get("disable_q_gradient_reuse", False),
        )
        diagnostics = True
    return pub.Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=numerics,
        time=time,
        initial_condition=initial,
        execution=pub.TorchSpectralExecution(device="cpu", options=options),
        output=pub.Output(
            directory=output_dir,
            steps=steps,
            save_interval=steps,
            diagnostic_interval=steps,
            save_start_step=0,
            diagnostics=diagnostics,
            save_hydrodynamics=family != "plane",
            restart_from=restart_from,
        ),
    )


def functional_plan(scene, root):
    from pssolver.functional import legacy_declarations as L
    from pssolver.functional.legacy_declarations.legacy_channel import translate_channel
    from pssolver.functional.legacy_declarations.legacy_periodic import translate_periodic
    from pssolver.functional.legacy_declarations.public_simulation_runner import (
        compile_public_simulation,
    )

    channel = scene["family"].endswith("channel")
    o = scene["options"]
    model = L.CompleteStressBerisEdwards(**CSBE)
    geometry = (L.RectangularChannel if channel else L.PeriodicBox)(
        shape=tuple(scene["shape"]), lengths=(16.0, 12.0, 12.0)
    )
    pressure = (
        {
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": o.get("relative_tolerance", 1e-10),
                "max_iterations": 40,
                "fixed_iterations": o.get("fixed_iterations", 12),
                "warm_start": o.get("warm_start", True),
            }
        }
        if channel
        else {}
    )
    sim = L.Simulation(
        model=model,
        geometry=geometry,
        boundaries=L.assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": L.neumann_q(),
                "velocity": L.no_slip_velocity() if channel else L.free_slip_velocity(),
                "pressure": L.neumann_pressure_compatibility(),
            },
        ),
        numerics=L.SpectralNumerics(
            dtype=scene["dtype"],
            dealias_rule=o.get("dealias_rule", "cubic_half"),
            projected_transform_execution=o.get("projected_transform_execution", "truncated"),
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
        ),
        time=L.TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=L.SnapshotInitialCondition(snapshot_dir(scene, root), step=0),
        execution=L.TorchSpectralExecution(
            runtime_path="channel_complete_stress" if channel else "periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical" if channel else "spectral",
                "pointwise_execution": o.get("pointwise_execution", "eager"),
                "disable_q_gradient_reuse": o.get("disable_q_gradient_reuse", False),
            },
        ),
        output=L.Output(
            directory=root / "output",
            steps=scene["steps"],
            save_interval=scene["steps"],
            diagnostic_interval=scene["steps"],
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=pressure,
    )
    spec = sim.specification
    if channel:
        plan = translate_channel(compile_public_simulation(spec).run_spec, flavour="functional_1_1")
        return getattr(plan, "plan", plan).identity_sha256
    return translate_periodic(
        spec, flavour="functional_1_1", load_initial=False
    ).plan.identity_sha256
