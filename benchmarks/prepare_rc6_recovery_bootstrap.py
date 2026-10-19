#!/usr/bin/env python3
"""Create the source-free import bootstrap used by RC6 recovery jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "pssolver.rc6.recovery_import_bootstrap.v1"
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
            raise RuntimeError("bootstrap record JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def initializer_text(script_root: Path) -> str:
    benchmark_root = script_root / "benchmarks"
    return (
        '"""Generated source-free RC6 qualification bootstrap."""\n'
        f"__path__ = [{json.dumps(str(benchmark_root))}]\n"
    )


def prepare(script_root: Path, import_root: Path) -> dict[str, object]:
    script_root_input = script_root.expanduser().absolute()
    import_root_input = import_root.expanduser().absolute()
    if not script_root_input.is_dir() or script_root_input.is_symlink():
        raise FileNotFoundError("ordinary script root required")
    script_root = script_root_input.resolve()
    import_root = import_root_input
    if (script_root / "pssolver").exists():
        raise ValueError("script root must not expose source pssolver")
    if import_root.exists() or import_root.is_symlink():
        raise FileExistsError(f"import root already exists: {import_root}")
    if import_root.resolve(strict=False) == script_root:
        raise ValueError("script root and import root must be distinct")
    scripts: dict[str, str] = {}
    for name in REQUIRED_SCRIPTS:
        path = script_root / "benchmarks" / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"regular qualification script required: {path}")
        scripts[name] = _sha256(path)
    package = import_root / "benchmarks"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text(initializer_text(script_root), encoding="utf-8")
    if (import_root / "pssolver").exists():
        raise RuntimeError("import root must not expose source pssolver")
    return {
        "schema": SCHEMA,
        "passed": True,
        "strategy": "distinct_source_free_script_root_and_benchmarks_path_shim",
        "script_root": str(script_root),
        "import_root": str(import_root),
        "script_root_contains_pssolver": False,
        "import_root_contains_pssolver": False,
        "benchmarks_source": str(script_root / "benchmarks"),
        "initializer": str(initializer),
        "initializer_sha256": _sha256(initializer),
        "required_scripts": scripts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, prepare(args.script_root, args.import_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
