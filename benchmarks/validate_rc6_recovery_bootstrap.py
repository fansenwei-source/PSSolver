#!/usr/bin/env python3
"""Persist and enforce the RC6 recovery script/import-root contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig


SCHEMA = "pssolver.rc6.recovery_bootstrap_validation.v1"
PLAN_SCHEMA = "pssolver.rc6.h100_stage_plan.v1"
STAGE_COUNT = 11
REQUIRED_SCRIPTS = (
    "run_rc5_g1_checkpoint_tmp_preflight.py",
    "run_rc6_cuda_toolkit_preflight.py",
    "run_rc6_single_h100_stage_plan.py",
    "validate_rc6_recovery_bootstrap.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("bootstrap validation JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _predicate(identifier: str, name: str, expected: object, observed: object) -> dict[str, object]:
    return {
        "id": identifier,
        "name": name,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _safe_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def observe(args: argparse.Namespace) -> dict[str, object]:
    driver = args.driver_python.expanduser().absolute()
    script_root_input = args.script_root.expanduser().absolute()
    import_root_input = args.import_root.expanduser().absolute()
    script_root = script_root_input.resolve()
    import_root = import_root_input.resolve()
    stage_plan = args.stage_plan.expanduser().resolve()
    output_root = args.output_root.expanduser().absolute()
    control_root = args.control_root.expanduser().resolve()
    predicates: list[dict[str, object]] = []
    predicates.append(_predicate("P01", "driver_python_matches_current_interpreter", str(driver), str(Path(sys.executable).absolute())))
    predicates.append(_predicate("P02", "script_root_is_ordinary_directory", True, script_root_input.is_dir() and not script_root_input.is_symlink()))
    predicates.append(_predicate("P03", "import_root_is_ordinary_directory", True, import_root_input.is_dir() and not import_root_input.is_symlink()))
    predicates.append(_predicate("P04", "script_and_import_roots_are_distinct", True, script_root != import_root))
    predicates.append(_predicate("P05", "script_root_does_not_expose_pssolver", False, (script_root / "pssolver").exists()))
    predicates.append(_predicate("P06", "import_root_does_not_expose_pssolver", False, (import_root / "pssolver").exists()))
    required: dict[str, dict[str, object]] = {}
    for name in REQUIRED_SCRIPTS:
        path = script_root / "benchmarks" / name
        regular = _safe_file(path)
        required[name] = {
            "path": str(path),
            "regular_non_symlink": regular,
            "sha256": _sha256(path) if regular else None,
        }
    predicates.append(_predicate("P07", "required_scripts_are_regular", True, all(row["regular_non_symlink"] for row in required.values())))
    initializer = import_root / "benchmarks/__init__.py"
    expected_initializer = (
        '"""Generated source-free RC6 qualification bootstrap."""\n'
        f"__path__ = [{json.dumps(str(script_root / 'benchmarks'))}]\n"
    )
    predicates.append(_predicate("P08", "import_initializer_is_exact", expected_initializer, initializer.read_text(encoding="utf-8") if _safe_file(initializer) else None))
    predicates.append(_predicate("P09", "import_root_has_only_benchmarks_package", ["benchmarks"], sorted(path.name for path in import_root.iterdir()) if import_root.is_dir() else None))
    predicates.append(_predicate("P10", "stage_plan_is_regular", True, _safe_file(stage_plan)))
    plan: dict[str, object] | None = None
    plan_error: str | None = None
    try:
        plan = json.loads(stage_plan.read_text(encoding="utf-8"))
    except Exception as error:  # persisted as evidence before fail-closed exit
        plan_error = f"{type(error).__name__}: {error}"
    predicates.append(_predicate("P11", "stage_plan_schema", PLAN_SCHEMA, plan.get("schema") if isinstance(plan, dict) else None))
    stages = plan.get("stages") if isinstance(plan, dict) else None
    predicates.append(_predicate("P12", "stage_count", STAGE_COUNT, len(stages) if isinstance(stages, list) else None))
    bindings_pass = True
    binding_details: list[dict[str, object]] = []
    if not isinstance(stages, list):
        bindings_pass = False
    else:
        for stage in stages:
            bindings = stage.get("command_file_bindings") if isinstance(stage, dict) else None
            script_bindings = [row for row in bindings or [] if isinstance(row, dict) and row.get("kind") == "python_script"]
            if len(script_bindings) != 1:
                bindings_pass = False
                continue
            binding = script_bindings[0]
            path = Path(str(binding.get("path", "")))
            try:
                path.relative_to(script_root / "benchmarks")
                in_script_root = True
            except ValueError:
                in_script_root = False
            actual = _sha256(path) if _safe_file(path) else None
            passed = in_script_root and binding.get("sha256") == actual
            bindings_pass = bindings_pass and passed
            binding_details.append({
                "stage": stage.get("name"),
                "path": str(path),
                "in_script_root": in_script_root,
                "expected_sha256": binding.get("sha256"),
                "actual_sha256": actual,
                "passed": passed,
            })
    predicates.append(_predicate("P13", "stage_script_bindings_are_source_free_and_hash_exact", True, bindings_pass))
    predicates.append(_predicate("P14", "formal_output_root_absent", False, output_root.exists() or output_root.is_symlink()))
    predicates.append(_predicate("P15", "control_logs_directory_exists", True, (control_root / "logs").is_dir()))
    predicates.append(_predicate("P16", "pythonpath_matches_import_root", str(import_root), os.environ.get("PYTHONPATH")))

    import_identity: dict[str, object]
    try:
        pssolver = importlib.import_module("pssolver")
        benchmarks = importlib.import_module("benchmarks")
        purelib = Path(sysconfig.get_path("purelib")).resolve()
        pssolver_file = Path(str(pssolver.__file__)).resolve()
        benchmark_file = Path(str(benchmarks.__file__)).resolve()
        import_identity = {
            "pssolver_file": str(pssolver_file),
            "purelib": str(purelib),
            "pssolver_from_purelib": pssolver_file.is_relative_to(purelib),
            "benchmarks_file": str(benchmark_file),
            "benchmarks_paths": [str(Path(value).resolve()) for value in benchmarks.__path__],
            "benchmarks_initializer_matches": benchmark_file == initializer.resolve(),
            "benchmarks_path_matches_script_root": [str(Path(value).resolve()) for value in benchmarks.__path__] == [str((script_root / "benchmarks").resolve())],
        }
    except Exception as error:  # persisted as evidence before fail-closed exit
        import_identity = {"exception": f"{type(error).__name__}: {error}"}
    predicates.append(_predicate("P17", "pssolver_imports_from_installed_purelib", True, import_identity.get("pssolver_from_purelib")))
    predicates.append(_predicate("P18", "benchmarks_import_uses_only_source_free_shim", True, bool(import_identity.get("benchmarks_initializer_matches")) and bool(import_identity.get("benchmarks_path_matches_script_root"))))
    passed = all(row["passed"] for row in predicates)
    return {
        "schema": SCHEMA,
        "passed": passed,
        "failures": [row["id"] for row in predicates if not row["passed"]],
        "predicates": predicates,
        "driver_python": str(driver),
        "script_root": str(script_root),
        "import_root": str(import_root),
        "stage_plan": str(stage_plan),
        "stage_plan_sha256": _sha256(stage_plan) if _safe_file(stage_plan) else None,
        "stage_plan_error": plan_error,
        "formal_output_root": str(output_root),
        "control_root": str(control_root),
        "required_scripts": required,
        "stage_script_bindings": binding_details,
        "import_identity": import_identity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-python", type=Path, required=True)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--stage-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = observe(args)
    _atomic_json(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
