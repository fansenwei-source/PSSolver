"""Run the preregistered S-29 C1 risk matrix against one immutable tree."""

import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


KERNELS = (
    "bulk_molecular_field",
    "bulk_molecular_field_with_static_lift",
    "algebraic_stress",
    "algebraic_stress_with_static_lift",
    "distortion_stress",
    "q_nonlinear",
    "q_nonlinear_with_static_lift",
)
COUNTS = (5, 5, 9, 9, 9, 5, 5)
GRIDS = (
    (16, 16, 16, 1),
    (16, 16, 12, 1),
    (16, 12, 12, 1),
    (17, 13, 11, 1),
    (128, 64, 64, 1),
    (16, 16, 16, 3),
)
SCENES = {
    "periodic": "periodic_default_float64_default",
    "plane": "plane_legacy_production_float64_default",
    "channel": "channel_default_float64_default",
}
SEEDS = tuple(range(2026111300, 2026111308))
BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def readback(out, s23):
    import torch
    import torch._inductor.config as config

    from pssolver.api.lowering import lower_simulation
    from qualification.identity.scenarios import scenarios
    from qualification.identity.scenes_v05 import simulation_v05

    rows = {}
    for group, name in SCENES.items():
        frozen = s23 / "results" / "matrix_baseline_first" / (name + ".json")
        source = json.loads(frozen.read_bytes())["scene"]
        (scene,) = (row for row in scenarios() if row["id"] == name)
        assert scene == source, (scene, source)
        simulation = simulation_v05(scene, out / "parameter_inputs" / group)
        problem, construction = lower_simulation(simulation)
        model = problem.model
        material = {**model.material, "ldg_l1": model.ldg_l1, "alpha": model.alpha.default}
        expected = {
            "ldg_a": 0.0,
            "ldg_b": -0.3,
            "ldg_c": 0.3,
            "gamma": 2.94,
            "flow_alignment": 0.3,
            "beta": -1.0,
        }
        assert all(material[key] == value for key, value in expected.items())
        if group == "plane":
            assert simulation.model.activity_number == 18.0
            assert simulation.geometry.domain.lengths[2] == 12.0
            assert simulation.model.parameterization == "fixed-k"
            assert simulation.model.frank_k == 1 / 81
            assert (simulation.model.coefficient_min, simulation.model.coefficient_max) == (
                0.01,
                0.05,
            )
            assert (material["ldg_l1"], material["alpha"]) == (
                construction.plane_preset.ldg_l1,
                construction.plane_preset.zeta,
            )
        else:
            assert (material["ldg_l1"], material["alpha"]) == (1 / 81, 0.01)
        rows[group] = {
            "id": name,
            "values": {k: float(v).hex() for k, v in material.items()},
            "frozen_record": str(frozen),
            "frozen_record_sha256": sha(frozen),
        }
    write(out / "parameters.json", rows)
    compiler = {}
    for tool in dict.fromkeys(
        (
            os.environ.get("CC", "gcc"),
            os.environ.get("CXX", "g++"),
            *(v for v in config.cpp.cxx if v),
        )
    ):
        path = Path(shutil.which(tool) or tool).resolve(strict=True)
        compiler[tool] = {
            "path": str(path),
            "sha256": sha(path),
            "version": subprocess.check_output([path, "--version"], text=True),
        }
    (out / "cpuinfo.txt").write_bytes(Path("/proc/cpuinfo").read_bytes())
    write(
        out / "fingerprint.json",
        {
            "python": sys.version,
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "torch": {
                "version": str(torch.__version__),
                "file": torch.__file__,
                "sha256": sha(Path(torch.__file__)),
                "build": torch.__config__.show(),
            },
            "inductor": {k: repr(v) for k, v in config.save_config_portable().items()},
            "compiler": compiler,
            "cpuinfo_sha256": sha(out / "cpuinfo.txt"),
            "environment": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                    "CC",
                    "CXX",
                )
            },
            "protected_kernel_sha256": sha(
                Path("pssolver/models/active_nematics/beris_edwards.py")
            ),
        },
    )


def inputs(torch, index, dtype, shape, seed, m):
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000 * (index + 1))

    def values(count):
        return tuple(
            torch.randn(shape, dtype=dtype, device="cpu", generator=generator) for _ in range(count)
        )

    def gradients(count):
        return tuple(values(count) for _ in range(3))

    if index < 2:
        args = tuple(values(5) for _ in range(index + 1))
        kwargs = {key: m[key] for key in ("ldg_a", "ldg_b", "ldg_c")}
    elif index < 4:
        args = tuple(values(5) for _ in range(index))
        kwargs = dict(
            flow_alignment=m["flow_alignment"],
            active_prefactor=m["beta"] * m["alpha"],
            q_dot_h=None,
        )
    elif index == 4:
        args = (gradients(5),)
        kwargs = {"ldg_l1": m["ldg_l1"]}
    else:
        args = tuple(values(5) for _ in range(index - 4))
        args += (values(3), gradients(5), gradients(3))
        kwargs = dict(
            ldg_b_over_gamma=m["ldg_b"] / m["gamma"],
            ldg_c_over_gamma=m["ldg_c"] / m["gamma"],
            flow_alignment=m["flow_alignment"],
        )
    return args, kwargs


def tensor_bytes(tensor):
    return tensor.detach().contiguous().numpy().tobytes()


def describe_tree(value):
    if isinstance(value, tuple):
        return [describe_tree(item) for item in value]
    return {
        "sha256": digest(tensor_bytes(value)),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
    }


def compare(torch, eager, compiled, dtype):
    a, b = eager.to(torch.float64), compiled.to(torch.float64)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    same_contract = eager.shape == compiled.shape and eager.dtype == compiled.dtype
    eager_raw, compiled_raw = tensor_bytes(eager), tensor_bytes(compiled)
    amplitude = float(a.abs().max()) if finite else None
    delta = float((a - b).abs().max()) if finite and same_contract else None
    denominator = torch.finfo(dtype).eps * amplitude if amplitude is not None else None
    ratio = delta / denominator if delta is not None and denominator else None
    zero_equal = eager_raw == compiled_raw
    return {
        "finite": finite,
        "same_contract": same_contract,
        "eager_sha256": digest(eager_raw),
        "compiled_sha256": digest(compiled_raw),
        "max_abs_eager": amplitude,
        "max_abs_delta": delta,
        "denominator": denominator,
        "r": ratio if ratio is None or math.isfinite(ratio) else None,
        "r_display": None if ratio is None else format(ratio, ".3g"),
        "zero_exact": zero_equal if amplitude == 0 else None,
        "passed": finite
        and same_contract
        and (zero_equal if amplitude == 0 else ratio is not None and ratio <= 16),
    }


def inventory(directory):
    return [
        {"path": str(path.relative_to(directory)), "size": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def measure(out, name, index, dtype_name, grid, group):
    start = time.time()
    caches = {
        key: Path(os.environ[key])
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR")
    }
    before = {key: inventory(path) for key, path in caches.items()}
    assert not any(before.values()), before

    import torch
    from torch._dynamo.utils import counters

    from pssolver.models.active_nematics.beris_edwards import BerisEdwardsPointwiseKernels

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    assert tuple(BerisEdwardsPointwiseKernels._EAGER_KERNELS) == KERNELS
    eager = BerisEdwardsPointwiseKernels("eager")
    compiled = BerisEdwardsPointwiseKernels("compile")
    metadata = compiled.metadata()
    assert metadata["effective"] == "compile" and not metadata["fallback_allowed"]
    assert metadata["fallback_reason"] is None
    config = metadata["compile"]
    assert (
        config["enabled"],
        config["backend"],
        config["mode"],
        config["fullgraph"],
        config["dynamic"],
    ) == (True, "inductor", "default", True, False)
    materials = json.loads((out / "parameters.json").read_bytes())
    m = {key: float.fromhex(value) for key, value in materials[group]["values"].items()}
    dtype = getattr(torch, dtype_name)
    key = KERNELS[index]
    rows = []
    first_compile = None
    with torch.no_grad():
        for seed in SEEDS:
            args, kwargs = inputs(torch, index, dtype, (grid[-1], *grid[:-1]), seed, m)
            descriptions = describe_tree(args)
            a = eager._kernels[key](*args, **kwargs)
            b = compiled._kernels[key](*args, **kwargs)
            assert descriptions == describe_tree(args), (name, seed)
            assert len(a) == len(b) == COUNTS[index]
            rows.append(
                {
                    "seed": seed,
                    "inputs": descriptions,
                    "input_sha256": digest(json.dumps(descriptions, sort_keys=True).encode()),
                    "outputs": [compare(torch, u, v, dtype) for u, v in zip(a, b, strict=True)],
                }
            )
            if first_compile is None:
                first_compile = {
                    "counters": {cat: dict(values) for cat, values in counters.items()},
                    "inductor": inventory(caches["TORCHINDUCTOR_CACHE_DIR"]),
                }
                assert counters["stats"]["unique_graphs"] >= 1, first_compile
                assert any(entry["path"].endswith(".so") for entry in first_compile["inductor"])
    assert not torch.cuda.is_initialized()
    write(
        out / "cells" / name / "result.json",
        {
            "name": name,
            "pid": os.getpid(),
            "metadata": metadata,
            "grid": grid,
            "group": group,
            "kernel": key,
            "dtype": dtype_name,
            "wall_seconds": time.time() - start,
            "kwargs": kwargs,
            "rows": rows,
            "cache_before": before,
            "first_compile": first_compile,
            "cache_after": {key: inventory(path) for key, path in caches.items()},
        },
    )


def aggregate(out, specs, codes, head):
    exceedances, other, determinism = [], [], []
    rows = 0
    rounds = []
    for round_index in range(len(codes)):
        seen = {}
        for i, spec in enumerate(specs):
            name = f"r{round_index + 1}-{i:03d}"
            code = codes[round_index][i]
            if code:
                other.append({"cell": name, "spec": spec, "exit": code})
                continue
            result = json.loads((out / "cells" / name / "result.json").read_bytes())
            for row in result["rows"]:
                seed = row["seed"]
                seen[i, seed] = (
                    row["input_sha256"],
                    [item["compiled_sha256"] for item in row["outputs"]],
                )
                for output_index, output in enumerate(row["outputs"]):
                    rows += 1
                    if not output["passed"]:
                        violation = {
                            "cell": name,
                            "spec": spec,
                            "seed": seed,
                            "output_index": output_index,
                            **output,
                        }
                        (
                            exceedances if output["r"] is not None and output["r"] > 16 else other
                        ).append(violation)
        rounds.append(seen)
    if len(rounds) == 2:
        for key in sorted(rounds[0].keys() & rounds[1].keys()):
            lhs, rhs = rounds[0][key], rounds[1][key]
            if lhs[0] != rhs[0]:
                other.append({"cross_round_input_difference": key})
            if lhs[1] != rhs[1]:
                determinism.append({"cell_and_seed": key, "round1": lhs[1], "round2": rhs[1]})
    expected_rows = sum(COUNTS[index] for index, *_ in specs) * len(SEEDS) * len(codes)
    write(
        out / "summary.json",
        {
            "baseline": BASE,
            "target": head,
            "cold_processes": sum(map(len, codes)),
            "zero_exit_processes": sum(code == 0 for group in codes for code in group),
            "rows": rows,
            "expected_rows": expected_rows,
            "exceedances": exceedances,
            "other_failures": other,
            "compile_hash_differences": determinism,
            "passed": rows == expected_rows and not exceedances and not other,
        },
    )


def run(out, scratch, s23, workers, mode, head, risk=None):
    sys.path.insert(0, str(Path.cwd()))
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == head
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    readback(out, s23)
    rounds = 2
    if mode == "formal":
        assert risk is not None
        reference = json.loads((risk / "verified.json").read_bytes())
        assert reference["passed"] and reference["rows"] == 27072
        assert (risk / "parameters.json").read_bytes() == (out / "parameters.json").read_bytes()

        def environment(path):
            data = json.loads((path / "fingerprint.json").read_bytes())
            # cpuinfo includes transient CPU frequency and bogomips. Stable
            # model, flags and affinity are captured in the full fingerprint.
            del data["cpuinfo_sha256"]
            return data

        same = (
            environment(out) == environment(risk)
            and (out / "full_fingerprint.json").read_bytes()
            == (risk / "full_fingerprint.json").read_bytes()
        )
        write(out / "risk_fingerprint_comparison.json", {"equal": same, "rounds": 1 if same else 2})
        rounds = 1 if same else 2
    specs = list(itertools.product(range(len(KERNELS)), ("float64", "float32"), GRIDS, SCENES))
    assert len(specs) == 252
    if mode == "preflight":
        specs = [specs[i] for i in range(0, len(specs), 36)]
        assert len(specs) == 7
        rounds = 1
    write(out / "matrix.json", specs)
    (out / "cells").mkdir()

    def launch(task):
        round_number, (i, (index, dtype, grid, group)) = task
        name = f"r{round_number}-{i:03d}"
        result_dir = out / "cells" / name
        result_dir.mkdir()
        work = scratch / name
        work.mkdir()
        env = os.environ.copy()
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
            path = work / key
            path.mkdir()
            env[key] = str(path)
        argv = [
            sys.executable,
            "-I",
            "-B",
            __file__,
            "cell",
            str(out),
            name,
            str(index),
            dtype,
            ",".join(map(str, grid)),
            group,
        ]
        write(
            result_dir / "command.json",
            {
                "argv": argv,
                "cwd": str(Path.cwd()),
                "cache_dirs": {k: v for k, v in env.items() if v.startswith(str(work))},
            },
        )
        with (
            (result_dir / "stdout").open("wb") as stdout,
            (result_dir / "stderr").open("wb") as stderr,
        ):
            rc = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, check=False).returncode
        if rc:
            write(result_dir / "failed_cache_inventory.json", inventory(work))
        (result_dir / "exit").write_text(f"{rc}\n")
        shutil.rmtree(work)
        print(name, rc, flush=True)
        return rc

    codes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for round_number in range(1, rounds + 1):
            tasks = list(itertools.product((round_number,), enumerate(specs)))
            # The first legitimate cell doubles as the approved cold compile probe.
            if round_number == 1:
                first = launch(tasks[0])
                if first:
                    raise RuntimeError("first cold compile probe failed; see r1-000")
                codes.append([first, *executor.map(launch, tasks[1:])])
            else:
                codes.append(list(executor.map(launch, tasks)))
    aggregate(out, specs, codes, head)
    if mode == "formal":
        differences = []
        for i, spec in enumerate(specs):
            name = f"r1-{i:03d}"
            if not (out / "cells" / name / "result.json").is_file():
                continue
            before, after = [
                json.loads((root / "cells" / name / "result.json").read_bytes())["rows"]
                for root in (risk, out)
            ]
            for a, b in zip(before, after, strict=True):
                assert (a["seed"], a["inputs"]) == (b["seed"], b["inputs"]), (name, spec)
                if a["outputs"] != b["outputs"]:
                    differences.append({"cell": name, "seed": a["seed"]})
        write(out / "risk_output_comparison.json", {"different": differences})


if __name__ == "__main__":
    if sys.argv[1] == "cell":
        sys.path.insert(0, str(Path.cwd()))
        measure(
            Path(sys.argv[2]),
            sys.argv[3],
            int(sys.argv[4]),
            sys.argv[5],
            tuple(map(int, sys.argv[6].split(","))),
            sys.argv[7],
        )
    elif sys.argv[1] == "run":
        run(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            int(sys.argv[5]),
            sys.argv[6],
            sys.argv[7],
            Path(sys.argv[8]) if len(sys.argv) > 8 else None,
        )
        raise SystemExit(
            int(not json.loads((Path(sys.argv[2]) / "summary.json").read_bytes())["passed"])
        )
    else:
        raise SystemExit("expected run or cell")
