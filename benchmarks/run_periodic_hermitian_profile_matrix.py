#!/usr/bin/env python3
"""Run the frozen Periodic Hermitian A/B profile matrix without shell arrays.

The orchestrator deliberately builds every subprocess invocation as a Python
``list[str]``.  Optional physical-artifact arguments are appended only for the
four production-forward trial-one profiles, so an absent optional argument can
never become an empty or unbound Bash array expansion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

from benchmarks.analyze_periodic_hermitian_profiles import (
    FROZEN_BASE_ACTIVITY,
    FROZEN_DT,
    FROZEN_GRIDS,
    FROZEN_INPUT_SHA256,
    FROZEN_LENGTHS,
    FROZEN_PROFILE_STEPS,
    FROZEN_TRIALS,
    FROZEN_WARMUP_STEPS,
)
from benchmarks.profile_periodic_functional import PROFILE_ROLES


MATRIX_KIND = "periodic_hermitian_profile_matrix_plan"
MATRIX_SCHEMA_VERSION = 1
VARIANTS = ("baseline", "candidate")


@dataclass(frozen=True, slots=True)
class VariantRuntime:
    python: Path
    repository_root: Path
    commit: str


@dataclass(frozen=True, slots=True)
class MatrixProfile:
    sequence: int
    variant: str
    grid: str
    role: str
    trial: int
    output: Path
    physical_artifact: Path | None
    command: tuple[str, ...]
    cwd: Path

    @property
    def key(self) -> str:
        return f"{self.variant}_{self.grid}_{self.role}_trial{self.trial}"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_import_bootstraps(
    output_root: Path,
    repositories: dict[str, Path],
) -> dict[str, Path]:
    """Expose repository benchmarks without exposing repository PSSolver.

    Executing ``.../benchmarks/script.py`` directly puts the benchmarks
    directory, not its parent, on ``sys.path``.  Consequently absolute
    ``benchmarks.*`` imports fail.  Adding the whole repository root to
    ``PYTHONPATH`` would fix that import while also shadowing the installed
    ``pssolver`` wheel with the source checkout.  A tiny package shim gives
    ``benchmarks`` the required package path and deliberately exposes no
    other repository package.
    """

    roots: dict[str, Path] = {}
    records: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        repository_root = repositories[variant].expanduser().resolve()
        benchmarks_root = repository_root / "benchmarks"
        if not benchmarks_root.is_dir():
            raise FileNotFoundError(
                f"{variant} benchmarks directory is absent: {benchmarks_root}"
            )
        bootstrap_root = output_root / "import_bootstrap" / variant
        package_root = bootstrap_root / "benchmarks"
        package_root.mkdir(parents=True)
        initializer = package_root / "__init__.py"
        initializer.write_text(
            '"""Generated qualification-only benchmarks package shim."""\n'
            f"__path__ = [{json.dumps(str(benchmarks_root))}]\n",
            encoding="utf-8",
        )
        if (bootstrap_root / "pssolver").exists():
            raise RuntimeError("import bootstrap must not expose source pssolver")
        roots[variant] = bootstrap_root.resolve()
        records[variant] = {
            "bootstrap_root": str(roots[variant]),
            "benchmarks_source": str(benchmarks_root),
            "pssolver_source_exposed": False,
        }
    _atomic_json(
        output_root / "import_bootstrap.json",
        {
            "schema_version": 1,
            "kind": "periodic_hermitian_matrix_import_bootstrap",
            "strategy": "benchmarks_package_path_shim",
            "variants": records,
        },
    )
    return roots


def _child_environment(
    base: dict[str, str],
    bootstrap_root: Path,
) -> dict[str, str]:
    environment = dict(base)
    environment["PYTHONPATH"] = str(bootstrap_root)
    return environment


def _variant_order(cell_index: int, trial: int) -> tuple[str, str]:
    """Alternate the leading variant globally across the frozen matrix."""

    return (
        VARIANTS
        if (cell_index + trial) % 2 == 0
        else tuple(reversed(VARIANTS))
    )


def build_matrix_plan(
    *,
    baseline: VariantRuntime,
    candidate: VariantRuntime,
    inputs: dict[str, Path],
    output_root: Path,
    device: str = "cuda",
) -> tuple[MatrixProfile, ...]:
    """Return the complete frozen matrix as explicit subprocess commands."""

    if device != "cuda":
        raise ValueError("the profiler CLI device token must be cuda")
    runtimes = {"baseline": baseline, "candidate": candidate}
    profiles: list[MatrixProfile] = []
    sequence = 0
    cell_index = 0
    for grid, shape in FROZEN_GRIDS.items():
        input_path = inputs[grid].expanduser().resolve()
        for role in PROFILE_ROLES:
            for trial in range(1, FROZEN_TRIALS + 1):
                for variant in _variant_order(cell_index, trial):
                    sequence += 1
                    runtime = runtimes[variant]
                    output = output_root / "profiles" / (
                        f"{variant}_{grid}_{role}_trial{trial}.json"
                    )
                    artifact = (
                        output_root
                        / "physical_q"
                        / f"{variant}_{grid}_production_forward.npy"
                        if role == "production_forward" and trial == 1
                        else None
                    )
                    command = [
                        str(runtime.python),
                        str(
                            runtime.repository_root
                            / "benchmarks/profile_periodic_hermitian_qualification.py"
                        ),
                        "--variant",
                        variant,
                        "--role",
                        role,
                        "--trial",
                        str(trial),
                        "--shape",
                        *(str(value) for value in shape),
                        "--lengths",
                        *(str(value) for value in FROZEN_LENGTHS),
                        "--initial-q-path",
                        str(input_path),
                        "--device",
                        device,
                        "--dtype",
                        "float64",
                        "--dt",
                        str(FROZEN_DT),
                        "--base-activity",
                        str(FROZEN_BASE_ACTIVITY),
                        "--warmup-steps",
                        str(FROZEN_WARMUP_STEPS),
                        "--profile-steps",
                        str(FROZEN_PROFILE_STEPS),
                        "--repository-root",
                        str(runtime.repository_root),
                        "--output",
                        str(output),
                    ]
                    if artifact is not None:
                        command.extend(("--physical-artifact", str(artifact)))
                    profiles.append(
                        MatrixProfile(
                            sequence=sequence,
                            variant=variant,
                            grid=grid,
                            role=role,
                            trial=trial,
                            output=output,
                            physical_artifact=artifact,
                            command=tuple(command),
                            cwd=runtime.repository_root,
                        )
                    )
                cell_index += 1
    expected = 2 * len(FROZEN_GRIDS) * len(PROFILE_ROLES) * FROZEN_TRIALS
    if len(profiles) != expected:
        raise AssertionError(f"expected {expected} profiles, built {len(profiles)}")
    if len({profile.key for profile in profiles}) != expected:
        raise AssertionError("profile matrix contains duplicate keys")
    if any(not value for profile in profiles for value in profile.command):
        raise AssertionError("profile command contains an empty argument")
    return tuple(profiles)


def _plan_metadata(profiles: Iterable[MatrixProfile]) -> dict[str, object]:
    records = []
    for profile in profiles:
        record = asdict(profile)
        for name in ("output", "physical_artifact", "cwd"):
            value = record[name]
            record[name] = None if value is None else str(value)
        record["command"] = list(profile.command)
        record["key"] = profile.key
        records.append(record)
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "kind": MATRIX_KIND,
        "profile_count": len(records),
        "profiler_cli_device_token": "cuda",
        "expected_allocated_device_identity": "cuda:0",
        "profiles": records,
    }


def _validate_input(path: Path, grid: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{grid} input is not a regular file")
    if _sha256(path) != FROZEN_INPUT_SHA256[grid]:
        raise ValueError(f"{grid} input SHA-256 differs")


def _validate_profile(profile: MatrixProfile) -> None:
    try:
        value = json.loads(profile.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid profile output for {profile.key}: {error}") from error
    if value.get("variant") != profile.variant:
        raise RuntimeError(f"profile variant differs for {profile.key}")
    base = value.get("base_profile")
    if not isinstance(base, dict):
        raise RuntimeError(f"base profile is absent for {profile.key}")
    config = base.get("config")
    environment = base.get("environment")
    if not isinstance(config, dict) or not isinstance(environment, dict):
        raise RuntimeError(f"profile identity is absent for {profile.key}")
    expected_shape = list(FROZEN_GRIDS[profile.grid])
    if config.get("shape") != expected_shape:
        raise RuntimeError(f"profile shape differs for {profile.key}")
    if config.get("role") != profile.role or config.get("trial") != profile.trial:
        raise RuntimeError(f"profile role or trial differs for {profile.key}")
    if config.get("device") != "cuda:0" or environment.get("device") != "cuda:0":
        raise RuntimeError(f"allocated device identity differs for {profile.key}")


def execute_matrix(
    profiles: tuple[MatrixProfile, ...],
    *,
    output_root: Path,
    candidate: VariantRuntime,
    baseline_commit: str,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root {output_root}")
    (output_root / "profiles").mkdir(parents=True)
    (output_root / "physical_q").mkdir()
    (output_root / "logs").mkdir()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    import_bootstraps = _prepare_import_bootstraps(
        output_root,
        {
            variant: next(
                profile.cwd for profile in profiles if profile.variant == variant
            )
            for variant in VARIANTS
        },
    )
    for profile in profiles:
        completed = subprocess.run(
            list(profile.command),
            cwd=profile.cwd,
            env=_child_environment(
                environment,
                import_bootstraps[profile.variant],
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        (output_root / "logs" / f"{profile.key}.stdout").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (output_root / "logs" / f"{profile.key}.stderr").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"profile {profile.key} failed with exit code {completed.returncode}"
            )
        _validate_profile(profile)
    analysis_path = output_root / "analysis.json"
    analyzer = candidate.repository_root / "benchmarks/analyze_periodic_hermitian_profiles.py"
    command = [
        str(candidate.python),
        str(analyzer),
        "--profiles",
        str(output_root / "profiles"),
        "--baseline-commit",
        baseline_commit,
        "--candidate-commit",
        candidate.commit,
        "--output",
        str(analysis_path),
    ]
    completed = subprocess.run(
        command,
        cwd=candidate.repository_root,
        env=_child_environment(environment, import_bootstraps["candidate"]),
        capture_output=True,
        text=True,
        check=False,
    )
    (output_root / "logs" / "analyzer.stdout").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_root / "logs" / "analyzer.stderr").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"profile analyzer failed with exit code {completed.returncode}"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("passed") is not True:
        raise RuntimeError("profile analyzer did not report PASS")
    _atomic_json(
        output_root / "MATRIX_COMPLETE.json",
        {
            "schema_version": 1,
            "kind": "periodic_hermitian_profile_matrix_completion",
            "profile_count": len(profiles),
            "analysis": str(analysis_path),
            "passed": True,
        },
    )


def _runtime(python: str, root: str, commit: str) -> VariantRuntime:
    return VariantRuntime(
        python=Path(python).expanduser().resolve(),
        repository_root=Path(root).expanduser().resolve(),
        commit=commit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-python", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--r128-input", required=True)
    parser.add_argument("--r320-input", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    baseline = _runtime(
        args.baseline_python,
        args.baseline_root,
        args.baseline_commit,
    )
    candidate = _runtime(
        args.candidate_python,
        args.candidate_root,
        args.candidate_commit,
    )
    inputs = {
        "R128": Path(args.r128_input).expanduser().resolve(),
        "R320": Path(args.r320_input).expanduser().resolve(),
    }
    for grid, path in inputs.items():
        _validate_input(path, grid)
    profiles = build_matrix_plan(
        baseline=baseline,
        candidate=candidate,
        inputs=inputs,
        output_root=args.output_root.expanduser().resolve(),
        device=args.device,
    )
    _atomic_json(args.plan_output, _plan_metadata(profiles))
    if args.execute:
        execute_matrix(
            profiles,
            output_root=args.output_root.expanduser().resolve(),
            candidate=candidate,
            baseline_commit=args.baseline_commit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
