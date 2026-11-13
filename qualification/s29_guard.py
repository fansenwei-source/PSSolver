"""Generate and check the S-29 design boundary from the accepted Git tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import tomllib


BASIS = "7267dcd00d58c11069a1ee205c181fa3f62ba745"
OWNERS = {
    "pssolver/engine/linear.py": (
        "C4/C5",
        ("DiagonalOperator", "LinearSolve", "MatrixFreeSolve", "SolveReport", "ModalBlockOperator"),
    ),
    "pssolver/engine/imex.py": ("C6", ("ImexScheme", "SemiImplicitEuler")),
    "pssolver/engine/program.py": ("C5/C6", ("advance",)),
    "pssolver/engine/derivatives.py": ("C3.2", ("StepDerivativeTable",)),
    "pssolver/engine/terms.py": ("C3.2", ("pointwise_explicit_term",)),
    "pssolver/engine/engine.py": (
        "C3.2/C4/C5/C6",
        (
            "Engine.__init__",
            "Engine._initialize",
            "Engine._for_reference",
            "Engine._generic_linear",
            "Engine._generic_explicit",
        ),
    ),
    "pssolver/planning/engine_plan.py": ("C4/C6", ("LinearTermPlan", "IntegratorPlan")),
    "pssolver/io/engine_state.py": ("C0.3/C6", ("read_header", "read", "_reject_test_integrator")),
    "pssolver/runtime/simulation_run.py": ("C0.3", ("run_assembled_simulation",)),
    "pssolver/experimental/representations.py": (
        "C3.3",
        ("_RepresentationPair", "_tensor_version", "AlgebraicRepresentationCache", "__all__"),
    ),
}
PATHS = {
    "qualification/s29_authority.json",
    "qualification/s29_guard.py",
    "qualification/s29_optimizations.json",
    "qualification/s29_preflight.py",
    "qualification/s29_trial.py",
    "qualification/s29_compare.py",
    "qualification/s29_t2.py",
    "qualification/s29_k2.py",
    "tests/categories.toml",
    "tests/test_v0_5_s29_boundary.py",
    "tests/test_v0_5_s29_registry.py",
    "tests/test_v0_5_s29_derivatives.py",
    "tests/test_v0_5_s29_linear_imex.py",
    "tests/oracles/s29_diffusion.py",
    "tests/s29_reference.py",
    "ARCHITECTURE.md",
}
PROTECTED = {
    "pssolver/models/active_nematics/beris_edwards.py",
    "pssolver/models/active_nematics/q_tensor.py",
    "pssolver/__init__.py",
    "pssolver/cli/plane.py",
    "pssolver/api/simulation.py",
    "pssolver/api/declarations.py",
    "pssolver/planning/assemble.py",
    "pssolver/engine/members.py",
    "pssolver/engine/state.py",
    "pssolver/engine/constraints.py",
    "MANIFEST.in",
    "setup.py",
    "pyproject.toml",
    "tests/test_projected_transform_truncation.py",
    "tests/test_phase4_scalar_sbdf2_reference.py",
    "tests/test_phase4_modal_block_operator.py",
    "tests/test_phase4_combined_sbdf2_modal_block.py",
    "tests/test_v0_5_s26_allen_cahn.py",
    "tests/test_v0_5_s26_assembly.py",
    "tests/test_v0_5_s28a_solver_state.py",
    "tests/v0_3_s03_successor.py",
    "qualification/s28b_tables.json",
    "qualification/s28b_guard.py",
    "qualification/s28b_frozen_callers.json",
    "qualification/s28b_test_disposition.json",
    "qualification/historical_successor_order.py",
    "qualification/historical_stage_audit.py",
    "qualification/historical_s26_audit.py",
    "qualification/historical_s27_audit.py",
    "qualification/historical_s28b_audit.py",
    "qualification/identity/scenarios.py",
    "qualification/s28b_trajectory.py",
    "qualification/k2/dispatch_counter.py",
    "CHANGELOG.md",
}
FROZEN_PREFIXES = (
    "pssolver/functional/",
    "qualification/stages/",
    "benchmarks/",
    "scripts_plane/",
    "scripts_channel/",
    "jobs/",
    "notes/",
)
RETIRE = {"_RepresentationPair", "_tensor_version", "AlgebraicRepresentationCache"}
REPRESENTATIONS = "pssolver/experimental/representations.py"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def files(root, commit):
    return {
        path.decode(): meta.split()[2].decode()
        for entry in git(root, "ls-tree", "-rz", commit).split(b"\0")
        if entry
        for meta, path in (entry.split(b"\t", 1),)
    }


def source(root, path):
    return git(root, "show", f"{BASIS}:{path}")


def units(raw, prefix=""):
    def walk(nodes, parent):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = parent + node.name
                yield name, node
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, name + ".")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        yield parent + target.id, node

    return dict(walk(ast.parse(raw, feature_version=(3, 10)).body, prefix))


def unit_sha(raw, node):
    start = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))])
    return digest(b"".join(raw.splitlines(keepends=True)[start - 1 : node.end_lineno]))


def layer_edges(raw):
    edges = set()
    for node in ast.walk(ast.parse(raw, feature_version=(3, 10))):
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            edges.add(module)
        elif isinstance(node, ast.Import):
            edges.update(alias.name for alias in node.names)
    return sorted(edges)


def generate(root):
    tracked = files(root, BASIS)
    required = PROTECTED | set(OWNERS)
    if missing := sorted(required - tracked.keys()):
        raise ValueError(f"unresolved baseline paths: {missing}")
    nodes = {}
    for path, (owner, editable) in OWNERS.items():
        raw = source(root, path)
        present = units(raw)
        if (
            set(editable)
            - present.keys()
            - {
                "_reject_test_integrator",
                "Engine._initialize",
                "Engine._for_reference",
                "LinearSolve",
                "MatrixFreeSolve",
                "SolveReport",
                "ModalBlockOperator",
            }
        ):
            raise ValueError(f"unresolved baseline units: {path}: {editable}")
        nodes[path] = {
            "owner": owner,
            "editable": {
                name: present[name].lineno if name in present else None for name in editable
            },
            "fixed": {
                name: unit_sha(raw, node)
                for name, node in present.items()
                if name not in editable
                and not any(
                    name.startswith(key + ".") or key.startswith(name + ".") for key in editable
                )
            },
            "blob": tracked[path],
            "allowed_import_modules": sorted(
                set(layer_edges(raw))
                | {
                    "pssolver/engine/linear.py": {"typing", "collections.abc"},
                    "pssolver/engine/imex.py": {"collections.abc"},
                    "pssolver/engine/derivatives.py": {"collections.abc"},
                    "pssolver/engine/engine.py": {".derivatives"},
                }.get(path, set())
            ),
        }
    frozen = {
        path: sha
        for path, sha in tracked.items()
        if path in PROTECTED or path.startswith(FROZEN_PREFIXES)
    }
    candidates = set(PATHS) | {"notes/PSSolver_v0_3_0_s03_successor_149_s29_interfaces.json"}
    return {
        "basis": BASIS,
        "basis_tree": git(root, "rev-parse", f"{BASIS}^{{tree}}").decode().strip(),
        "changed_units": nodes,
        "allowed_other_paths": sorted(candidates),
        "frozen_blobs": frozen,
        "guard_sha256": digest(Path(__file__).read_bytes()),
        "six_cuda_deselect": tomllib.loads(source(root, "tests/categories.toml").decode())[
            "gpu_prerequisites"
        ]["nodeids"],
    }


def check(root, table, *, final):
    problems = {
        key: []
        for key in ("authority", "paths", "frozen", "owners", "edges", "version", "retirement")
    }
    if table != generate(root):
        problems["authority"].append("authority differs from authenticated basis/tool")
    tracked = files(root, BASIS)
    note = "notes/PSSolver_v0_3_0_s03_successor_149_s29_interfaces.json"
    changes = set(git(root, "diff", "--name-only", BASIS).decode().splitlines())
    changes.update(git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    if final:
        problems["paths"].extend(sorted(set(OWNERS) - changes))
    problems["paths"].extend(sorted(changes - set(OWNERS) - set(table["allowed_other_paths"])))
    for path, blob in table["frozen_blobs"].items():
        path_obj = root / path
        if path not in changes and path_obj.is_file() and not path_obj.is_symlink():
            continue
        if (
            not path_obj.is_file()
            or path_obj.is_symlink()
            or git(root, "hash-object", path).decode().strip() != blob
        ):
            problems["frozen"].append(path)
    for path in changes & set(OWNERS):
        raw = (root / path).read_bytes()
        current = units(raw)
        rule = table["changed_units"][path]
        problems["edges"].extend(
            f"{path}:{name}"
            for name in layer_edges(raw)
            if name not in rule["allowed_import_modules"]
        )
        base = units(source(root, path))
        allowed = set(rule["editable"])
        for name, sha in rule["fixed"].items():
            if name not in current or unit_sha(raw, current[name]) != sha:
                problems["owners"].append(f"{path}:{name}")
        for name in current.keys() - base.keys():
            if name not in allowed and not any(name.startswith(key + ".") for key in allowed):
                problems["owners"].append(f"{path}:{name}")
    for path in sorted((root / "pssolver").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative == "pssolver/models/active_nematics/beris_edwards.py":
            continue
        if final or relative != REPRESENTATIONS:
            for node in ast.walk(ast.parse(path.read_bytes(), feature_version=(3, 10))):
                if isinstance(node, ast.Attribute) and node.attr == "_version":
                    problems["version"].append(f"{relative}:{node.lineno}")
    if final:
        actual = units((root / REPRESENTATIONS).read_bytes())
        problems["retirement"].extend(sorted(RETIRE & actual.keys()))
        if note not in changes or note in tracked:
            problems["retirement"].append("missing forward successor 149")
    return {key: sorted(value) for key, value in problems.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "probe", "final"))
    parser.add_argument("root", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("report", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "generate":
        args.table.write_text(json.dumps(generate(args.root), indent=2, sort_keys=True) + "\n")
        return 0
    report = check(args.root, json.loads(args.table.read_bytes()), final=args.mode == "final")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(any(report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
