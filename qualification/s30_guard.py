"""Check the S-30 import contracts without importing the numerical package."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from tools._source import module_name, read_toml
from tools.structure_metrics import cycles


PILOT_OWNER = "pssolver.operators:__getattr__"
PILOT_RESOLVER = "pssolver.core._compatibility:resolve_deprecated_facade_name"
PILOT_TARGET = "pssolver.experimental.boundary_pilots"
PILOT_EXPORTS = (
    "CellCenteredRobinEigenbasisOperator",
    "FRESH_INITIAL_CONDITIONING_SCHEMA_VERSION",
    "FreshInitialConditioningPolicy",
    "LEGACY_WHOLE_SLAB_SINE",
    "LOCALIZED_QUINTIC_WALL_TRANSITION",
    "MaterializedLinearLiftCorrection",
    "PlaneStaticLiftingOperator",
    "STRICT_NO_AUTOMATIC_CONDITIONING",
    "legacy_whole_slab_sine_conditioning",
    "localized_quintic_wall_transition",
    "materialize_cell_centered_robin_eigenbasis",
    "materialize_plane_static_lifting",
    "strict_no_automatic_conditioning",
)
FROZEN_LEGACY = "pssolver.functional.legacy_declarations"
ENTRY_POINTS = {
    "importlib.import_module",
    "importlib.util.spec_from_file_location",
    "runpy.run_path",
    "builtins.__import__",
    "__import__",
    PILOT_RESOLVER.replace(":", "."),
}


@dataclass(frozen=True)
class ImportUse:
    """A resolved import or a call requiring a literal source of destinations."""

    owner: str
    line: int
    target: str
    top_level: bool = False


def target_parts(text):
    return text.partition(":")[0].split(".")


def package_name(module, is_package):
    return module if is_package else module.rpartition(".")[0]


def literal_assignment(tree, name):
    (value,) = (
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return value


def export_dictionary(tree, name):
    return ast.literal_eval(literal_assignment(tree, name))


def name_tuple(tree, name):
    value = literal_assignment(tree, name)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
    ):
        value = value.args[0]
    return tuple(ast.literal_eval(value))


def simple_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{simple_name(node.value)}.{node.attr}"
    return ""


def constructor_export_table(tree, table, names, module):
    node = literal_assignment(tree, table)
    if not isinstance(node, ast.DictComp):
        raise ValueError(f"{table} must be a literal-derived map")
    if ast.dump(node.key) != ast.dump(ast.parse("name", mode="eval").body) or ast.dump(
        node.value
    ) != ast.dump(ast.parse(f"({module!r}, name)", mode="eval").body):
        raise ValueError(f"{table} key or destination changed")
    (generator,) = node.generators
    if generator.ifs or generator.is_async or generator.target.id != "name":
        raise ValueError(f"{table} comprehension changed")
    if names and simple_name(generator.iter) != names:
        raise ValueError(f"{table} names source changed")
    return name_tuple(tree, names) if names else tuple(ast.literal_eval(generator.iter))


def facade_destinations(parsed):
    operators = parsed["pssolver.operators"]
    functional = parsed["pssolver.functional"]
    operator_names = constructor_export_table(operators, "_PILOT_EXPORTS", None, PILOT_TARGET)
    if operator_names != PILOT_EXPORTS:
        raise ValueError("operators compatibility names changed")
    functional_names = constructor_export_table(
        functional, "_COMPATIBILITY_EXPORTS", "_STABLE_API_EXPORT_NAMES", "pssolver.functional.api"
    )
    (update,) = (
        node.value
        for node in functional.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and simple_name(node.value.func) == "_COMPATIBILITY_EXPORTS.update"
    )
    extra = ast.literal_eval(update.args[0])
    targets = {f"pssolver.functional.api:{name}" for name in functional_names}
    targets |= {f"{module}:{name}" for module, name in extra.values()}
    return {
        PILOT_OWNER: ("_PILOT_EXPORTS", {f"{PILOT_TARGET}:{name}" for name in operator_names}),
        "pssolver.functional:__getattr__": ("_COMPATIBILITY_EXPORTS", targets),
    }


def alias_targets(node, module, is_package, modules):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        base = importlib.util.resolve_name(
            "." * node.level + base, package_name(module, is_package)
        )
    return [
        candidate if candidate in modules else f"{base}:{alias.name}"
        for alias in node.names
        for candidate in (f"{base}.{alias.name}",)
    ]


def uses(tree, module, is_package, modules):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for alias, path in zip(
                node.names, alias_targets(node, module, is_package, modules), strict=True
            ):
                aliases[alias.asname or alias.name] = path.replace(":", ".")

    def visit(node, parent, top_level):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = (*parent, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_level = False
        owner = module + ":" + ".".join(parent) if parent else module
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for target in alias_targets(node, module, is_package, modules):
                yield ImportUse(owner, node.lineno, target, top_level), None
        elif isinstance(node, ast.Call):
            head, dot, tail = simple_name(node.func).partition(".")
            target = aliases.get(head, head) + dot + tail
            if target in ENTRY_POINTS:
                yield ImportUse(owner, node.lineno, target, top_level), node
        if (
            isinstance(node, ast.If)
            and simple_name(node.test).rsplit(".", 1)[-1] == "TYPE_CHECKING"
        ):
            for child in node.body:
                yield from visit(child, parent, False)
            for child in node.orelse:
                yield from visit(child, parent, top_level)
        else:
            for child in ast.iter_child_nodes(node):
                yield from visit(child, parent, top_level)

    yield from visit(tree, (), True)


def capability_destinations(parsed):
    rows = literal_assignment(parsed["pssolver.planning.capabilities"], "CAPABILITIES")
    return {
        ast.literal_eval(node.args[2])
        for node in ast.walk(rows)
        if isinstance(node, ast.Call) and simple_name(node.func) == "Capability"
    }


def legacy_destinations(parsed):
    module = FROZEN_LEGACY + ".application_registry"
    rows = literal_assignment(parsed[module], "APPLICATIONS")
    return {
        f"{FROZEN_LEGACY}.{old.split('.')[-2]}:{old.split('.')[-1]}"
        for member in rows.elts
        for entry in (literal_assignment(parsed[module], member.id),)
        for old in (ast.literal_eval(entry.args[3]),)
    }


def resolve_call(use, node, parsed, facades):
    owner = use.owner
    module = owner.partition(":")[0]
    if use.target == PILOT_RESOLVER.replace(":", "."):
        table, destinations = facades[owner]
        (argument,) = (row.value for row in node.keywords if row.arg == "exports")
        if simple_name(argument) != table:
            raise ValueError("facade no longer uses the reviewed export table")
        return destinations
    if not node.args:
        raise ValueError("dynamic import has no positional target")
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return {first.value}
    if owner == "pssolver:__getattr__" and simple_name(first) == "module":
        return {
            f"{value}:{name}"
            for name, value in export_dictionary(parsed[module], "_EXPORTS").items()
        }
    if owner == "pssolver.engine.engine:_resolve" and simple_name(first) == "module":
        return capability_destinations(parsed)
    if (
        owner == FROZEN_LEGACY + ".public_simulation_runner:compile_public_simulation"
        and simple_name(first) == "module"
    ):
        return legacy_destinations(parsed)
    if owner in {PILOT_RESOLVER} and simple_name(first) == "module_name":
        return set().union(*(names for _, names in facades.values()))
    if (
        module in {"pssolver.operators", "pssolver.functional", "pssolver.runtime"}
        and owner.endswith(":__getattr__")
        and ast.dump(first) == ast.dump(ast.parse('f"{__name__}.{name}"', mode="eval").body)
    ):
        return {
            f"{module}.{name}" for name in name_tuple(parsed[module], "_IMPLICIT_SUBMODULE_NAMES")
        }
    raise ValueError("dynamic target is not backed by a recognized literal source")


def forbidden(contracts, owner, target):
    src, dst = target_parts(owner), target_parts(target)
    layer = src[1] if len(src) > 1 else ""
    if dst[:1] == ["torch"]:
        return "torch" in contracts.get(layer, ())
    if len(dst) < 2 or dst[0] != "pssolver":
        return False
    return dst[1] in contracts.get(layer, ()) or (
        dst[1] in contracts["outside_experimental"] and layer != "experimental"
    )


def import_contract(root: Path, replacements=None):
    """Read the whole package and return exact allowed edges, calls, SCCs and failures."""
    config = read_toml(root / "pyproject.toml")["tool"]["pssolver"]
    contracts = config["architecture-contracts"]
    registered = {
        (owner, target) for owner, paths in config["architecture-edges"].items() for target in paths
    }
    expected_facades = {
        PILOT_OWNER: {
            "resolver": PILOT_RESOLVER,
            "target": PILOT_TARGET,
            "exports": list(PILOT_EXPORTS),
        }
    }
    paths = {module_name(path, root): path for path in sorted((root / "pssolver").rglob("*.py"))}
    replacements = replacements or {}
    parsed = {
        name: ast.parse(replacements.get(name, path.read_text()), filename=name)
        for name, path in paths.items()
    }
    graph = {module: set() for module in paths}
    violations, dynamic, legacy, observed = [], [], [], set()
    if config["architecture-facades"] != expected_facades:
        violations.append("changed facade authorization")
    try:
        facades = facade_destinations(parsed)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        facades = {}
        violations.append(str(exc))
    for module, path in paths.items():
        for use, call in uses(parsed[module], module, path.name == "__init__.py", paths):
            if call is not None:
                try:
                    targets = resolve_call(use, call, parsed, facades)
                    if not targets:
                        raise ValueError("empty destination set")
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    violations.append(f"{use.owner}:{use.line}: {exc}")
                    continue
                dynamic.append((use.owner, use.line, use.target, sorted(targets)))
            else:
                targets = {use.target}
            for target in sorted(targets):
                name = target.partition(":")[0]
                if use.top_level and name in paths and name != module:
                    graph[module].add(name)
                if module.startswith(FROZEN_LEGACY + "."):
                    legacy.append((use.owner, use.line, target))
                    continue
                edge = (use.owner, target)
                if edge in registered:
                    observed.add(edge)
                if not forbidden(contracts, use.owner, target) or edge in registered:
                    continue
                # The generic resolver is not a second owner of the dependencies.
                # Its possible destinations are enumerated above; the actual
                # edge is checked independently at each facade's call site.
                if call is not None and use.owner == PILOT_RESOLVER:
                    continue
                if (
                    call is not None
                    and use.owner == PILOT_OWNER
                    and (
                        use.target == PILOT_RESOLVER.replace(":", ".")
                        and target in {f"{PILOT_TARGET}:{name}" for name in PILOT_EXPORTS}
                    )
                ):
                    continue
                violations.append(f"{use.owner}:{use.line}: forbidden {target}")
    frozen_cycles, active_cycles = [], []
    for group in cycles(graph):
        (
            frozen_cycles
            if all(name.startswith(FROZEN_LEGACY + ".") for name in group)
            else active_cycles
        ).append(group)
    violations.extend(
        f"registered edge absent: {owner} -> {target}"
        for owner, target in sorted(registered - observed)
    )
    return dict(
        violations=violations,
        dynamic=dynamic,
        frozen_legacy=legacy,
        cycles=active_cycles,
        frozen_cycles=frozen_cycles,
        exact_edges=sorted(observed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = import_contract(args.root)
    print(json.dumps(report, sort_keys=True, indent=2))
    return int(bool(report["violations"] or report["cycles"]))


if __name__ == "__main__":
    raise SystemExit(main())
