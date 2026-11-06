"""Explicit S-23 minimum option inventory, not a production dispatch registry."""

from __future__ import annotations

from itertools import product


def scenarios() -> list[dict]:
    """Return deterministic scenes; the installed release decides acceptance."""
    rows = []
    for family in ("periodic", "plane", "channel", "functional_periodic", "functional_channel"):
        shape = {"periodic": [16, 16, 16], "plane": [16, 16, 12], "channel": [16, 12, 12]}
        geometry = family.removeprefix("functional_")
        variants = [("default", {}), ("cpu_compile", {"pointwise_execution": "compile"})]
        if family == "plane":
            variants += [
                (
                    f"spaces_{linear}_{stress}",
                    {
                        "molecular_field_linear_space": linear,
                        "stress_divergence_sum_space": stress,
                    },
                )
                for linear, stress in product(("spectral", "physical"), repeat=2)
            ]
            variants += [("reuse_off", {"disable_q_gradient_reuse": True})]
            variants += [
                (f"projection_{value}", {"projected_transform_execution": value})
                for value in ("truncated", "full")
            ]
            variants += [("zero_mean", {"zero_mode_policy": "zero_mean"})]
            variants += [
                (f"friction_{value}", {"zero_mode_policy": "friction", "friction_mode_fric": value})
                for value in (0.1, 0.3)
            ]
            variants += [("refresh20", {"spectral_refresh_steps": 20})]
        elif not family.startswith("functional_"):
            variants += [("reuse_off", {"disable_q_gradient_reuse": True})]
        if geometry == "channel":
            variants += [
                ("fixed_work", {"fixed_iterations": 12}),
                ("relative_tolerance", {"fixed_iterations": None}),
                (
                    "relative_tolerance_float32_floor",
                    {"fixed_iterations": None, "relative_tolerance": 1e-6},
                ),
            ]
            if family.startswith("functional_"):
                variants += [("warm_start_false", {"warm_start": False})]
            else:
                variants += [("odd_grid", {"shape": [17, 13, 11]})]
        if not family.startswith("functional_"):
            variants += [
                (f"dealias_{rule}", {"dealias_rule": rule})
                for rule in ("none", "two_thirds", "cubic_half")
            ]
            variants += [
                (
                    "dealias_none_full",
                    {"dealias_rule": "none", "projected_transform_execution": "full"},
                )
            ]
        paths = ("legacy_production", "compiled_v2") if family == "plane" else (None,)
        for dtype, runtime_path, (label, options) in product(
            ("float64", "float32"), paths, variants
        ):
            rows.append(
                {
                    "id": f"{family}_{runtime_path or 'default'}_{dtype}_{label}",
                    "family": family,
                    "dtype": dtype,
                    "shape": options.get("shape", shape[geometry]),
                    "steps": 32 if family.startswith("functional_") else 64,
                    "restart_steps": [23] if family.startswith("functional_") else [23, 40],
                    "runtime_path": runtime_path,
                    "options": options,
                    "seed": 20261106,
                    "observe": True,
                }
            )
        if not family.startswith("functional_"):
            default = next(row for row in rows if row["family"] == family)
            rows.append({**default, "id": default["id"] + "_unobserved", "observe": False})
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate scene identity")
    return rows
