#!/usr/bin/env python3
"""Installed-wheel A/A/B trajectories and four PCG paths, CPU or H100.

All roles reuse precisely the same calculation path sequentially. Results and
compiler caches live elsewhere. Frozen S-18/S-09 files are imported unchanged;
the PCG harness function is rebound to an explicit device-aware fixture in a
private globals dictionary, never patched in the runtime or frozen module.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import types
import uuid
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._artifact_io import atomic_write_bytes, atomic_write_json, atomic_write_text, file_sha256, runtime_environment
from benchmarks._s19_compare import HardQualificationError, compare_field, compare_trace, require_execution, require_finite_payload
from benchmarks.plan_v0_4_s19_h100 import REFRESH_INTERVALS, ROLES, build_plan

PUBLIC_PATHS = ('legacy_production', 'compiled_v2', 'legacy_channel',
                'compiled_channel_v2', 'periodic_spectral', 'channel_complete_stress')


def _public_dispatch(args):
    """Exercise compile_simulation/run_simulation, not the runtime factories."""
    import numpy as np
    import torch
    from pssolver import compile_simulation, run_simulation
    from benchmarks.run_rc6_consumer import _execution_on_device
    from tests import test_phase7_p7711_public_entry_equivalence as legacy
    from tests import test_phase8_p82_periodic_application as periodic
    from tests import test_phase8_p83_channel_complete_stress_application as channel
    traces, terminals = [], {}
    for index, (name, steps, checkpoint, restart) in enumerate((
        ('continuous', 4, None, None), ('segment', 2, 1, None),
        ('resume', 2, None, args.work / 'segment/checkpoint_2'),
    )):
        if args.family in PUBLIC_PATHS[:4]:
            application = 'plane' if args.family in PUBLIC_PATHS[:2] else 'channel'
            simulation = legacy._simulation(application, args.work / name, args.family,
                                             steps=steps, checkpoint_interval=checkpoint, restart_from=restart)
        else:
            module = periodic if args.family == 'periodic_spectral' else channel
            simulation = module._simulation(args.work, output_name=name, steps=steps,
                                             checkpoint_interval=checkpoint, restart_from=restart)
        simulation = replace(simulation, execution=_execution_on_device(simulation.execution, args.device))
        compiled = compile_simulation(simulation)
        identity = compiled.to_metadata()
        if identity['fallback_allowed'] is not False:
            raise HardQualificationError('public compilation permits fallback')
        torch.manual_seed(1234)
        result = run_simulation(compiled)
        metadata = result.to_metadata()
        atomic_write_json(args.output.parent / f'{name}_run.json', metadata, overwrite=False)
        files = sorted((args.work / name).rglob('*.npy'))
        if not files:
            raise HardQualificationError('public entry produced no arrays')
        fields = {}
        for file in files:
            array = np.load(file, allow_pickle=False)
            relative = str(file.relative_to(args.work / name))
            fields[relative] = _persist_tensor(args.output.parent / 'arrays' / name / relative,
                                                torch.from_numpy(array.copy()))
        traces.append(dict(step=index, fields=fields, runtime_identity=identity,
                           configuration_identity=simulation.canonical_sha256(), progress={'run': name, 'steps': steps}))
        terminals[name] = {}
        for prefix in ('Q_', 'u_', 'p_'):
            values = sorted((args.work / name).glob(prefix + '*.npy'))
            if not values:
                raise HardQualificationError('missing public Q/u/p output')
            terminals[name][prefix] = np.load(values[-1], allow_pickle=False)
    restart_metrics = [compare_field(terminals['continuous'][field], terminals['resume'][field],
                                     field=field, step=4) for field in ('Q_', 'u_', 'p_')]
    return dict(kind='public_dispatch', family=args.family, interval='default', traces=traces,
                restart_comparison=restart_metrics,
                classification='N' if any(x['classification'] == 'N' for x in restart_metrics) else 'PASS')


def _persist_tensor(path: Path, tensor, *, pcg_record=None) -> dict[str, object]:
    import numpy as np
    array = tensor.detach().contiguous().cpu().numpy()
    finite = require_finite_payload(array, pcg_record=pcg_record)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    atomic_write_bytes(path, stream.getvalue(), overwrite=False)
    return {"path": str(path), "sha256": file_sha256(path), "shape": list(array.shape), "dtype": str(array.dtype), "finite": finite}


def _installed_identity(role: str, wheel: Path) -> dict[str, object]:
    import pssolver
    import torch
    plan = build_plan()
    expected = plan["provider"]["baseline" if role.startswith("baseline") else "candidate"]
    if file_sha256(wheel) != expected["wheel_sha256"]:
        raise HardQualificationError("input wheel SHA-256 mismatch")
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if not Path(pssolver.__file__).resolve().is_relative_to(purelib):
        raise HardQualificationError("Provider is not in this venv site-packages")
    if importlib.metadata.version("pssolver") != expected["version"]:
        raise HardQualificationError("Provider version mismatch")
    if torch.__version__ != "2.5.1+cu121":
        raise HardQualificationError("torch version differs from the C/B contract")
    matched = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(".py"):
                expected_sha = hashlib.sha256(archive.read(name)).hexdigest()
                actual = file_sha256(purelib / name)
                if actual != expected_sha:
                    raise HardQualificationError(f"installed runtime byte mismatch: {name}")
                matched[name] = actual
    if not matched:
        raise HardQualificationError("empty wheel runtime inventory")
    return dict(expected, imported=str(Path(pssolver.__file__).resolve()), purelib=str(purelib),
                runtime_file_sha256=matched, environment=runtime_environment())


def _prepare_child(args):
    import torch
    from benchmarks.run_rc5_g1_numerical import _tf32_off
    # Check before touching any computation or compiler in the child.
    caches = [Path(os.environ[key]) for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")]
    if any(not path.is_dir() or path.is_symlink() or list(path.iterdir()) for path in caches):
        raise HardQualificationError("child compiler caches must initially be empty")
    identity = _installed_identity(args.role, args.wheel)
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise HardQualificationError("exactly one CUDA device required")
        if "H100" not in torch.cuda.get_device_name(0) or str(torch.empty((), device="cuda").device) != "cuda:0":
            raise HardQualificationError("allocated CUDA device is not cuda:0 H100")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise HardQualificationError("CPU rehearsal must hide CUDA")
    _tf32_off()
    return identity


def _trajectory(args) -> dict[str, object]:
    from benchmarks.profile_rc458_runtime_family import _build_case
    from benchmarks.run_rc5_g1_numerical import _config
    from benchmarks.verify_v0_4_s18_r4_step_equivalence import _progress_record
    config = replace(_config(args.family, "candidate", "0" * 40, "0" * 64),
                     device=args.device, warmup_steps=0, profile_steps=20, post_flush_rewarm_steps=0)
    case_root = args.work / "case"
    case_root.mkdir(parents=True, exist_ok=False)
    case = _build_case(case_root, config)
    interval = json.loads(args.interval)
    if interval != "default":
        case.solver.integrator.set_spectral_refresh_interval(interval)
    traces = []
    for step in range(21):
        state = tuple(case.state())
        if any(str(value.device) != ("cuda:0" if args.device == "cuda" else "cpu") for value in state):
            raise HardQualificationError("trajectory tensor device mismatch")
        identity = case.runtime_identity
        if identity.get("fallback_used") is not False:
            raise HardQualificationError("trajectory fallback or missing fallback report")
        fields = {f"state_{index}": _persist_tensor(args.output.parent / "arrays" / f"step{step:02d}_state{index}.npy", value)
                  for index, value in enumerate(state)}
        traces.append({"step": step, "fields": fields, "runtime_identity": identity,
                       "configuration_identity": case.configuration_identity, "progress": _progress_record(case)})
        if step < 20:
            case.step()
    if int(case.completed_steps()) != 20:
        raise HardQualificationError("trajectory completed-step count differs")
    return {"kind": "trajectory", "family": args.family, "interval": interval, "traces": traces}


def _pcg(args) -> dict[str, object]:
    import torch
    from benchmarks import verify_v0_4_s18_r5_pcg_equivalence as frozen
    from pssolver import SpectralSolver
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver
    from pssolver.functional.channel_pressure import ChannelPressureTransposeOperator
    records = []
    case_device = args.device

    def build_solver(case):
        shape = tuple(case["shape"])
        spectral = SpectralSolver(shape, L=tuple(float(case["length_scale"]) * (1 + .37 * x) for x in shape),
                                  device=case_device, dtype=getattr(torch, case["dtype"]))
        solver = ChannelNoSlipModalStokesSolver(
            spectral.transform_backend, friction=case["friction"], viscosity=case["viscosity"],
            pressure_relative_tolerance=case["relative_tolerance"], pressure_max_iterations=case["max_iterations"],
            pressure_fixed_iterations=case["fixed_iterations"])
        return spectral, solver

    def rhs(solver, case, *, seed_offset=0):
        generator = torch.Generator().manual_seed(case["seed"] + seed_offset)
        real = torch.randn((1, *case["shape"]), generator=generator, dtype=getattr(torch, case["dtype"]))
        imag = torch.randn((1, *case["shape"]), generator=generator, dtype=getattr(torch, case["dtype"]))
        return ChannelPressureTransposeOperator(solver)._project_gauge(torch.complex(real, imag).to(case_device))

    for case in frozen.CASES:
        for path in frozen.PATHS:
            # The oracle public contract is CPU-only; never bypass its guard.
            case_device = "cpu" if path == "oracle" else args.device
            fields = {}
            names = iter(("result", "gradient", "warm_pressure", "warm_operator_action")
                         if path in ("implicit", "oracle") else
                         ("result", "warm_pressure", "warm_operator_action"))

            def tensor_record(tensor):
                key = next(names)
                expected_device = "cuda:0" if case_device == "cuda" else "cpu"
                if str(tensor.device) != expected_device:
                    raise HardQualificationError("PCG tensor device mismatch")
                fields[key] = _persist_tensor(args.output.parent / "arrays" / f"{case['name']}_{path}_{key}.npy", tensor,
                                              pcg_record=(case["name"], path, key))
                return frozen._tensor_record(tensor)

            namespace = dict(frozen._run_path.__globals__)
            namespace.update(_build_solver=build_solver, _rhs=rhs, _tensor_record=tensor_record)
            run = types.FunctionType(frozen._run_path.__code__, namespace, frozen._run_path.__name__)
            result = run(case, path)
            records.append({"case": case, "path": path, "record": result, "fields": fields, "execution_device": case_device})
    return {"kind": "pcg", "cases": records, "fixture_binding": "unchanged S-18 CASES/PATHS/_run_path; only explicit device and tensor persistence adapters"}


def child(args) -> None:
    identity = _prepare_child(args)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    data = {'trajectory': _trajectory, 'pcg': _pcg, 'public_dispatch': _public_dispatch}[args.kind](args)
    nonfinite = [[row["case"]["name"], row["path"], name]
                 for row in data.get("cases", []) for name, value in row["fields"].items() if not value["finite"]]
    atomic_write_json(args.output, dict(data, identity=identity, role=args.role,
                                       execution_device="cuda:0" if args.device == "cuda" else "cpu",
                                       finite=not nonfinite, nonfinite_records=nonfinite,
                                       fallback_used=False, installed_wheel_only=True,
                                       cache_initially_empty=True, returncode=0), overwrite=False)


def _read_array(record):
    import numpy as np
    path = Path(record["path"])
    if not path.is_file() or path.is_symlink() or file_sha256(path) != record["sha256"]:
        raise HardQualificationError("comparison payload hash mismatch")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
        raise HardQualificationError("comparison payload schema mismatch")
    return array


def _compare(left, right):
    if left["kind"] != right["kind"]:
        raise HardQualificationError("comparison kind differs")
    if left["kind"] in ("trajectory", "public_dispatch"):
        if (left["family"], left["interval"]) != (right["family"], right["interval"]):
            raise HardQualificationError("trajectory case differs")
        traces = []
        for report in (left, right):
            traces.append([dict(step, fields={name: _read_array(row) for name, row in step["fields"].items()})
                           for step in report["traces"]])
        report = compare_trace(*traces)
        if left['kind'] == 'public_dispatch' and any(row['classification'] == 'N' for row in (left, right)):
            report.update(classification='N', qualification_passed=False)
        return report
    if len(left["cases"]) != len(right["cases"]):
        raise HardQualificationError("PCG case coverage differs")
    fields, records = [], []
    for index, (a, b) in enumerate(zip(left["cases"], right["cases"])):
        if (a["case"] != b["case"] or a["path"] != b["path"] or list(a["fields"]) != list(b["fields"])
                or a.get("execution_device") != b.get("execution_device")):
            raise HardQualificationError("PCG fixture identity differs")
        for key in a["fields"]:
            fields.append(compare_field(_read_array(a["fields"][key]), _read_array(b["fields"][key]),
                                        field=f"{a['case']['name']}/{a['path']}/{key}", step=index,
                                        pcg_record=(a['case']['name'], a['path'], key)))
        records.append({"case": a["case"]["name"], "path": a["path"], "exact": a["record"] == b["record"],
                        "reference": a["record"], "observed": b["record"]})
    passed = all(row["bitwise_equal"] for row in fields) and all(row["exact"] for row in records)
    return {"classification": "PASS" if passed else "N", "qualification_passed": passed,
            "continue_control_and_profiles": True, "fields": fields, "records": records}


def execute(args) -> dict[str, object]:
    plan = build_plan()
    support = args.support_root.resolve()
    if (support / "pssolver").exists() or (support / "pssolver_control").exists():
        raise HardQualificationError("support shadows an installed package")
    output = args.output_root.absolute()
    output.mkdir(parents=True, exist_ok=False)
    work = args.work.absolute()
    if work.name != "s19_shared_calculation" or work.exists() or work.is_symlink():
        raise HardQualificationError("designated initially absent work directory required")
    if output.is_relative_to(work) or work.is_relative_to(output):
        raise HardQualificationError("calculation and retained outputs must be separate")
    pythons = {"baseline_a": args.baseline_python, "baseline_b": args.baseline_python, "candidate": args.candidate_python}
    wheels = {"baseline_a": args.baseline_wheel, "baseline_b": args.baseline_wheel, "candidate": args.candidate_wheel}
    cases = [("trajectory", family, interval) for family in plan["numerical"]["families"] for interval in REFRESH_INTERVALS]
    cases.append(("pcg", "four_paths", "default"))
    cases.extend(("public_dispatch", path, "default") for path in PUBLIC_PATHS)
    summary, launches = [], []
    for number, (kind, family, interval) in enumerate(cases):
        reports = {}
        for role in ROLES:
            owner = uuid.uuid4().hex
            work.mkdir(parents=True, exist_ok=False)
            marker = work / ".s19_owner"
            atomic_write_text(marker, owner, overwrite=False)
            cache = output / "compiler_cache" / f"{number:02d}_{role}"
            for name in ("triton", "inductor"):
                (cache / name).mkdir(parents=True, exist_ok=False)
            destination = output / "children" / f"{number:02d}_{role}" / "result.json"
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", NVIDIA_TF32_OVERRIDE="0",
                       TRITON_CACHE_DIR=str(cache / "triton"), TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
            if args.device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            command = [str(pythons[role]), "-I", "-B", str(Path(__file__).resolve()), "--internal-child",
                       "--role", role, "--device", args.device, "--kind", kind, "--family", family,
                       "--interval", json.dumps(interval), "--work", str(work), "--wheel", str(wheels[role]),
                       "--output", str(destination)]
            result = subprocess.run(command, cwd=support, env=env, capture_output=True, text=True, check=False)
            atomic_write_text(output / "logs" / f"{number:02d}_{role}.stdout", result.stdout)
            atomic_write_text(output / "logs" / f"{number:02d}_{role}.stderr", result.stderr)
            launch = {"command": command, "returncode": result.returncode, "same_work_path": str(work), "compile_cache": str(cache)}
            launches.append(launch)
            atomic_write_json(output / "launches" / f"{number:02d}_{role}.json", launch, overwrite=False)
            if result.returncode != 0:
                raise HardQualificationError(f"{kind}/{family}/{interval}/{role} crashed; see retained logs")
            report = json.loads(destination.read_text())
            require_execution(report, {"role": role, "kind": kind, "execution_device": "cuda:0" if args.device == "cuda" else "cpu"})
            expected = plan["provider"]["baseline" if role.startswith("baseline") else "candidate"]
            if any(report["identity"].get(key) != value for key, value in expected.items()):
                raise HardQualificationError("child runtime identity mismatch")
            reports[role] = report
            # Only delete the new directory created by THIS loop, after its
            # outputs and successful report have been persisted elsewhere.
            if work.is_symlink() or marker.read_text() != owner:
                raise HardQualificationError("owned work directory changed")
            shutil.rmtree(work)
        comparisons = {}
        for a, b in (("baseline_a", "baseline_b"), ("baseline_a", "candidate"), ("baseline_b", "candidate")):
            comparisons[f"{a}_vs_{b}"] = _compare(reports[a], reports[b])
        record = {"kind": kind, "family": family, "interval": interval, "comparisons": comparisons,
                  "classification": "N" if any(v["classification"] == "N" for v in comparisons.values()) else "PASS"}
        atomic_write_json(output / "comparisons" / f"{number:02d}.json", record, overwrite=False)
        summary.append(record)
    result = {"classification": "N" if any(row["classification"] == "N" for row in summary) else "PASS",
              "formal_h100_contract": args.device == "cuda", "execution_device": args.device,
              "same_absolute_work_path": str(work), "launch_count": len(launches), "cases": summary,
              "continue_control_and_profiles": True, "nonzero_tolerance_authorized": False}
    atomic_write_json(output / "summary.json", result, overwrite=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    for name in ("work", "output", "wheel", "support-root", "output-root", "baseline-python", "candidate-python", "baseline-wheel", "candidate-wheel"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--kind", choices=("trajectory", "pcg", "public_dispatch"))
    parser.add_argument("--family")
    parser.add_argument("--interval")
    args = parser.parse_args()
    if args.internal_child:
        child(args)
    else:
        execute(args)
    return 0  # N is retained and evaluated by the final adjudicator, never retried.


if __name__ == "__main__":
    raise SystemExit(main())
