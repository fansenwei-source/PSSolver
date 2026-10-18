#!/usr/bin/env python3
"""Capture and compare the frozen RC5/RC6 functional provider identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
from types import ModuleType


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RC5_COMMIT = "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
RC6_COMMIT = "ea801ec08f5b2465661334e1c126133e55b68e96"
CONSUMER_COMMIT = "c14e6b56c88081e7f9a5642e65ffd722eec7445b"
CONSUMER_ARCHIVE_SHA256 = (
    "3b66240d6de16956c8958268e75b79e3e250efca61f21ff51ec44609ab93728e"
)
FIXTURE_SHA256 = {
    "periodic_runtime.py": (
        "6bc103bb7ab28fe297c5847fa715a5479f2a4aeef2af7c07ea956fa725e3ca79"
    ),
    "channel_runtime.py": (
        "15958068d17cf472aa7fc2b323434eb665f1f1e90ead6c7a43d880d2d3d2a8d9"
    ),
}
EXPECTED_RC5_DIGESTS = {
    "periodic_canonical": (
        "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d"
    ),
    "channel_canonical_fixed_iteration": (
        "b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905"
    ),
    "channel_tolerance": (
        "b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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
        _require(
            json.loads(path.read_text(encoding="utf-8")) == expected,
            "JSON round-trip differs",
        )
    finally:
        temporary.unlink(missing_ok=True)


def _installed_identity(
    *, expected_version: str, expected_commit: str
) -> dict[str, object]:
    import pssolver

    package_file = Path(pssolver.__file__).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    source_package = (SOURCE_ROOT / "pssolver").resolve()
    _require(package_file.is_relative_to(purelib), "pssolver is outside purelib")
    _require(
        not package_file.is_relative_to(source_package),
        "source shadow import detected",
    )
    _require(
        importlib.metadata.version("pssolver") == expected_version,
        "installed distribution version differs",
    )
    _require(pssolver.__version__ == expected_version, "public version differs")
    return {
        "qualified_commit": expected_commit,
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "purelib": str(purelib),
        "pssolver_version": pssolver.__version__,
        "pssolver_import": str(package_file),
        "source_package": str(source_package),
        "source_shadow_import": False,
        "python_isolated_mode": bool(sys.flags.isolated),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _load_fixture(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider_identity(
    fixture_name: str,
    periodic: ModuleType,
    channel: ModuleType,
    root: Path,
) -> tuple[dict[str, object], str]:
    from pssolver.functional import (
        build_channel_activity_functional_runtime,
        build_functional_runtime,
        channel_activity_functional_request,
        periodic_activity_functional_request,
    )

    if fixture_name == "periodic_canonical":
        simulation = periodic.simulation(root)
        request = periodic_activity_functional_request(simulation.specification)
        runtime = build_functional_runtime(request)
    else:
        pressure_solver = (
            None
            if fixture_name == "channel_canonical_fixed_iteration"
            else channel.TOLERANCE_PRESSURE_SOLVER
        )
        simulation = channel.simulation(root, pressure_solver=pressure_solver)
        request = channel_activity_functional_request(simulation.specification)
        runtime = build_channel_activity_functional_runtime(request)
    metadata = runtime.identity().to_metadata()
    digest = runtime.identity().canonical_sha256()
    _require(
        digest == _canonical_sha256(metadata),
        f"canonical identity digest differs for {fixture_name}",
    )
    _require(metadata["api_version"] == "1.0", "functional API version differs")
    return metadata, digest


def capture(
    *,
    output_root: Path,
    output: Path,
    fixture_root: Path,
    fixture_archive: Path,
    label: str,
    expected_version: str,
    expected_commit: str,
) -> None:
    installed = _installed_identity(
        expected_version=expected_version,
        expected_commit=expected_commit,
    )
    fixture_root = fixture_root.expanduser().resolve()
    fixture_archive = fixture_archive.expanduser().resolve()
    _require(fixture_root.is_dir(), "consumer fixture root is absent")
    _require(fixture_archive.is_file(), "consumer fixture archive is absent")
    _require(
        _sha256(fixture_archive) == CONSUMER_ARCHIVE_SHA256,
        "consumer fixture archive SHA-256 differs",
    )
    fixture_paths = {
        name: fixture_root / name for name in FIXTURE_SHA256
    }
    for name, expected in FIXTURE_SHA256.items():
        _require(fixture_paths[name].is_file(), f"fixture is absent: {name}")
        _require(
            _sha256(fixture_paths[name]) == expected,
            f"fixture SHA-256 differs: {name}",
        )

    periodic = _load_fixture("rc6_identity_periodic_fixture", fixture_paths["periodic_runtime.py"])
    channel = _load_fixture("rc6_identity_channel_fixture", fixture_paths["channel_runtime.py"])
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    fixtures: dict[str, object] = {}
    for fixture_name in EXPECTED_RC5_DIGESTS:
        identities = []
        digests = []
        for replica in ("first_root", "second_root"):
            metadata, digest = _provider_identity(
                fixture_name,
                periodic,
                channel,
                output_root / fixture_name / replica,
            )
            identities.append(metadata)
            digests.append(digest)
        _require(
            identities[0] == identities[1],
            f"identity is path dependent: {fixture_name}",
        )
        _require(
            digests[0] == digests[1],
            f"identity digest is path dependent: {fixture_name}",
        )
        fixtures[fixture_name] = {
            "provider_identity": identities[0],
            "provider_canonical_sha256": digests[0],
            "canonical_equals_compact_json_sha": True,
            "path_independent": True,
            "replica_count": 2,
        }
    result = {
        "schema": "pssolver.rc6.provider_identity_capture.v1",
        "label": label,
        "qualified_commit": expected_commit,
        "installed_identity": installed,
        "consumer_fixture": {
            "qualified_source_commit": CONSUMER_COMMIT,
            "archive": str(fixture_archive),
            "archive_sha256": CONSUMER_ARCHIVE_SHA256,
            "root": str(fixture_root),
            "files": FIXTURE_SHA256,
        },
        "fixtures": fixtures,
    }
    _atomic_json(output, result)


def _differences(left: object, right: object, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(_differences(a, b, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def analyze(
    *,
    baseline_path: Path,
    candidate_path: Path,
    historical_path: Path,
    output: Path,
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    _require(baseline["qualified_commit"] == RC5_COMMIT, "RC5 commit differs")
    _require(candidate["qualified_commit"] == RC6_COMMIT, "RC6 commit differs")
    _require(
        baseline["consumer_fixture"] == candidate["consumer_fixture"],
        "consumer fixture identities differ",
    )
    results: dict[str, object] = {}
    for fixture_name, expected_digest in EXPECTED_RC5_DIGESTS.items():
        old = historical["fixtures"][fixture_name]
        rc5 = baseline["fixtures"][fixture_name]
        rc6 = candidate["fixtures"][fixture_name]
        _require(
            old["provider_canonical_sha256"] == expected_digest,
            f"historical digest differs: {fixture_name}",
        )
        _require(
            rc5["provider_canonical_sha256"] == expected_digest,
            f"rebuilt RC5 digest differs: {fixture_name}",
        )
        _require(
            old["provider_identity"] == rc5["provider_identity"],
            f"rebuilt RC5 metadata differs from frozen evidence: {fixture_name}",
        )
        changed_paths = _differences(
            rc5["provider_identity"], rc6["provider_identity"]
        )
        digest_equal = (
            rc5["provider_canonical_sha256"]
            == rc6["provider_canonical_sha256"]
        )
        _require(not changed_paths, f"provider identity changed: {fixture_name}")
        _require(digest_equal, f"provider digest changed: {fixture_name}")
        results[fixture_name] = {
            "rc5_canonical_sha256": rc5["provider_canonical_sha256"],
            "rc6_canonical_sha256": rc6["provider_canonical_sha256"],
            "full_metadata_equal": True,
            "canonical_sha256_equal": True,
            "changed_paths": changed_paths,
            "changed_path_count": len(changed_paths),
            "rc5_path_independent": rc5["path_independent"],
            "rc6_path_independent": rc6["path_independent"],
        }
    result = {
        "schema": "pssolver.rc6.provider_identity_impact_analysis.v1",
        "classification": (
            "PASS_RC6_PROVIDER_IDENTITY_UNCHANGED_NO_CONTROL_IDENTITY_MIGRATION_REQUIRED"
        ),
        "baseline_commit": RC5_COMMIT,
        "candidate_commit": RC6_COMMIT,
        "consumer_fixture_commit": CONSUMER_COMMIT,
        "fixture_count": len(results),
        "changed_fixture_count": 0,
        "changed_path_count": 0,
        "fixtures": results,
        "functional_api_version": "1.0",
        "provider_identity_impact_review_complete": True,
        "provider_identity_changed": False,
        "pssolver_control_identity_migration_required": False,
        "pssolver_control_installed_wheel_qualification_complete": False,
        "pssolver_control_guard_order_work_is_separate": True,
        "production_default_changed": False,
        "scientific_timestep_executed": False,
    }
    _atomic_json(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output-root", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--fixture-root", type=Path, required=True)
    capture_parser.add_argument("--fixture-archive", type=Path, required=True)
    capture_parser.add_argument("--label", required=True)
    capture_parser.add_argument("--expected-version", required=True)
    capture_parser.add_argument("--expected-commit", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--baseline", type=Path, required=True)
    analyze_parser.add_argument("--candidate", type=Path, required=True)
    analyze_parser.add_argument("--historical", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "capture":
        capture(
            output_root=arguments.output_root,
            output=arguments.output,
            fixture_root=arguments.fixture_root,
            fixture_archive=arguments.fixture_archive,
            label=arguments.label,
            expected_version=arguments.expected_version,
            expected_commit=arguments.expected_commit,
        )
    else:
        analyze(
            baseline_path=arguments.baseline,
            candidate_path=arguments.candidate,
            historical_path=arguments.historical,
            output=arguments.output,
        )


if __name__ == "__main__":
    main()
