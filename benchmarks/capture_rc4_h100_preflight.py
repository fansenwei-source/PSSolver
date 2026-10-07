#!/usr/bin/env python3
"""Capture and enforce the RC4 H100 preflight one predicate at a time."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any


SCHEMA = "pssolver.rc4_1_2.h100_preflight.v1"
PASS_CLASSIFICATION = "PASS_RC4_1_2_H100_PREFLIGHT"
FAIL_CLASSIFICATION = "FAIL_RC4_1_2_H100_PREFLIGHT"
_IMPORT = object()


class PreflightFailure(RuntimeError):
    """Raised only after the complete predicate report has been persisted."""


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace *path* with strict JSON and verify the result."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _import_or_record(
    name: str,
    supplied: object,
) -> tuple[object | None, dict[str, str] | None]:
    if supplied is not _IMPORT:
        return supplied, None
    try:
        return importlib.import_module(name), None
    except Exception as error:  # pragma: no cover - exercised through fakes
        return None, _exception_record(error)


def _module_path(module: object | None) -> str | None:
    if module is None:
        return None
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        return None
    return str(Path(value).resolve())


def _read_flag(getter: Callable[[], object]) -> dict[str, Any]:
    try:
        return {"value": bool(getter()), "exception": None}
    except Exception as error:
        return {"value": None, "exception": _exception_record(error)}


def _set_flag(setter: Callable[[], None]) -> dict[str, Any]:
    try:
        setter()
        return {"applied": True, "exception": None}
    except Exception as error:
        return {"applied": False, "exception": _exception_record(error)}


def _evaluate(
    *,
    predicate_id: str,
    expression: str,
    expected: object,
    observe: Callable[[], object],
    compare: Callable[[object], bool],
) -> dict[str, Any]:
    try:
        observed = observe()
        passed = bool(compare(observed))
        exception = None
    except Exception as error:
        observed = None
        passed = False
        exception = _exception_record(error)
    return {
        "id": predicate_id,
        "expression": expression,
        "expected": expected,
        "observed": observed,
        "exception": exception,
        "status": "PASS" if passed else "FAIL",
    }


def capture_preflight(
    *,
    q_root: Path,
    output: Path,
    enforce: bool = True,
    torch_module: object = _IMPORT,
    pssolver_module: object = _IMPORT,
    diagnostic_module: object = _IMPORT,
) -> dict[str, Any]:
    """Persist raw state, set TF32 policy, and checkpoint P01--P08."""

    q_root = q_root.resolve()
    output = output.resolve()
    if not q_root.is_dir():
        raise NotADirectoryError(q_root)
    if output.exists():
        raise FileExistsError(output)

    torch_module, torch_error = _import_or_record("torch", torch_module)
    pssolver_module, pssolver_error = _import_or_record(
        "pssolver", pssolver_module
    )
    diagnostic_module, diagnostic_error = _import_or_record(
        "benchmarks.diagnose_plane_nyquist_storage", diagnostic_module
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": "IN_PROGRESS",
        "preflight_complete": False,
        "enforcement_requested": enforce,
        "q_root": str(q_root),
        "environment": {
            "cwd": str(Path.cwd().resolve()),
            "python_executable": sys.executable,
            "python_executable_resolved": str(Path(sys.executable).resolve()),
            "python": platform.python_version(),
            "torch": getattr(torch_module, "__version__", None),
            "cuda_runtime": getattr(
                getattr(torch_module, "version", None), "cuda", None
            ),
            "variables": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "LD_LIBRARY_PATH",
                    "LOADEDMODULES",
                    "NVIDIA_TF32_OVERRIDE",
                    "PYTHONNOUSERSITE",
                    "PYTHONPATH",
                )
            },
            "imports": {
                "pssolver": {
                    "path": _module_path(pssolver_module),
                    "exception": pssolver_error,
                },
                "diagnostic": {
                    "path": _module_path(diagnostic_module),
                    "exception": diagnostic_error,
                },
                "torch": {"exception": torch_error},
            },
        },
        "tf32_policy": {
            "before": {},
            "assignments": {},
            "after": {},
        },
        "predicates": [],
        "progress": {"completed_predicate_ids": []},
    }
    write_json_atomic(output, report)

    def torch_required() -> object:
        if torch_module is None:
            raise RuntimeError("torch import failed")
        return torch_module

    report["tf32_policy"]["before"] = {
        "cuda_matmul_allow_tf32": _read_flag(
            lambda: torch_required().backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": _read_flag(
            lambda: torch_required().backends.cudnn.allow_tf32
        ),
    }
    write_json_atomic(output, report)

    def disable_matmul_tf32() -> None:
        torch_required().backends.cuda.matmul.allow_tf32 = False

    def disable_cudnn_tf32() -> None:
        torch_required().backends.cudnn.allow_tf32 = False

    report["tf32_policy"]["assignments"] = {
        "torch.backends.cuda.matmul.allow_tf32 = False": _set_flag(
            disable_matmul_tf32
        ),
        "torch.backends.cudnn.allow_tf32 = False": _set_flag(
            disable_cudnn_tf32
        ),
    }
    report["tf32_policy"]["after"] = {
        "cuda_matmul_allow_tf32": _read_flag(
            lambda: torch_required().backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": _read_flag(
            lambda: torch_required().backends.cudnn.allow_tf32
        ),
    }
    write_json_atomic(output, report)

    def imported_path(module: object | None, name: str) -> str:
        path = _module_path(module)
        if path is None:
            raise RuntimeError(f"{name} import has no file path")
        return path

    predicates = (
        (
            "P01",
            "Path(pssolver.__file__).resolve().is_relative_to(q_root)",
            f"path under {q_root}",
            lambda: imported_path(pssolver_module, "pssolver"),
            lambda value: Path(str(value)).is_relative_to(q_root),
        ),
        (
            "P02",
            "Path(diagnostic.__file__).resolve().is_relative_to(q_root)",
            f"path under {q_root}",
            lambda: imported_path(diagnostic_module, "diagnostic"),
            lambda value: Path(str(value)).is_relative_to(q_root),
        ),
        (
            "P03",
            "torch.cuda.is_available()",
            True,
            lambda: bool(torch_required().cuda.is_available()),
            lambda value: value is True,
        ),
        (
            "P04",
            "torch.cuda.device_count() == 1",
            1,
            lambda: int(torch_required().cuda.device_count()),
            lambda value: value == 1,
        ),
        (
            "P05",
            'str(torch.empty((), device="cuda").device) == "cuda:0"',
            "cuda:0",
            lambda: str(torch_required().empty((), device="cuda").device),
            lambda value: value == "cuda:0",
        ),
        (
            "P06",
            '"H100" in torch.cuda.get_device_name(0)',
            "device name containing H100",
            lambda: str(torch_required().cuda.get_device_name(0)),
            lambda value: "H100" in str(value),
        ),
        (
            "P07",
            "not torch.backends.cuda.matmul.allow_tf32",
            False,
            lambda: bool(
                torch_required().backends.cuda.matmul.allow_tf32
            ),
            lambda value: value is False,
        ),
        (
            "P08",
            "not torch.backends.cudnn.allow_tf32",
            False,
            lambda: bool(torch_required().backends.cudnn.allow_tf32),
            lambda value: value is False,
        ),
    )
    for predicate_id, expression, expected, observe, compare in predicates:
        result = _evaluate(
            predicate_id=predicate_id,
            expression=expression,
            expected=expected,
            observe=observe,
            compare=compare,
        )
        report["predicates"].append(result)
        report["progress"]["completed_predicate_ids"].append(predicate_id)
        write_json_atomic(output, report)

    failed = [
        predicate for predicate in report["predicates"]
        if predicate["status"] != "PASS"
    ]
    report["preflight_complete"] = True
    report["classification"] = (
        PASS_CLASSIFICATION if not failed else FAIL_CLASSIFICATION
    )
    report["enforcement"] = {
        "passed": not failed,
        "failed_predicate_ids": [item["id"] for item in failed],
    }
    write_json_atomic(output, report)

    if enforce and failed:
        details = "; ".join(
            f"{item['id']} observed={item['observed']!r} "
            f"expected={item['expected']!r} exception={item['exception']!r}"
            for item in failed
        )
        raise PreflightFailure(f"RC4 H100 preflight failed: {details}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = capture_preflight(
            q_root=args.q_root,
            output=args.output,
            enforce=not args.no_enforce,
        )
    except PreflightFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
