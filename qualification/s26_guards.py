"""Static planning closure and scoped tensor-construction observations for S-26."""

import ast
import sys
from contextlib import contextmanager
from unittest.mock import patch

from qualification.s25_import_scan import imports


PLANNING_ROOTS = (
    "pssolver.core.terms",
    "pssolver.planning.assemble",
    "pssolver.planning.capabilities",
    "pssolver.planning.term_analysis",
    "pssolver.presets.beris_edwards",
)
COMPONENTS = frozenset(
    {
        "engine",
        "models",
        "presets",
        "geometries",
        "boundaries",
        "integrators",
        "backends",
        "operators",
        "linear_solvers",
        "runtime",
        "io",
        "functional",
        "api",
        "applications",
        "workflows",
        "adapters",
        "configuration",
        "execution",
        "systems",
        "experimental",
        "control",
    }
)


def planning_closure(root):
    """Inspect nested imports transitively, without importing the package root."""
    pending, seen, violations, exceptions = list(PLANNING_ROOTS), set(), [], []
    while pending:
        module = pending.pop()
        if module in seen or module == "pssolver":
            continue
        seen.add(module)
        path = root / (module.replace(".", "/") + ".py")
        if not path.is_file():
            path = root / module.replace(".", "/") / "__init__.py"
        tree = ast.parse(path.read_text())
        late_lines = set()
        if module == "pssolver.planning.assemble":
            cls = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AssembledProblem"
            )
            method = next(
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "engine"
            )
            late_lines = {n.lineno for n in method.body if isinstance(n, ast.ImportFrom)}
        for line, base, name in imports(tree, module, path.name == "__init__.py", nested=True):
            edge = dict(module=module, line=line, target=base, name=name)
            if (
                module == "pssolver.planning.assemble"
                and line in late_lines
                and base == "pssolver.engine.engine"
                and name == "Engine"
            ):
                exceptions.append(dict(edge, owner="AssembledProblem.engine"))
                continue
            if base.split(".")[0] in {"torch", "numpy", "scipy"} or (
                base.startswith("pssolver.") and base.split(".")[1] in COMPONENTS
            ):
                violations.append(edge)
                continue
            if base.startswith("pssolver."):
                full = base + "." + name if name else base
                target = full if (root / (full.replace(".", "/") + ".py")).is_file() else base
                pending.append(target)
    if len(exceptions) != 1:
        violations.append(dict(reason="expected exactly one named deferred Engine import"))
    return dict(modules=sorted(seen), exceptions=exceptions, violations=violations)


@contextmanager
def tensor_constructors():
    """Count both Tensor.__new__ and direct calls to the immutable C base constructor.

    CPython's c_call profile hook observes explicit _TensorBase.__new__ calls;
    Tensor's Python attribute is wrapped separately because type-call slots do not
    emit c_call. Factory allocations are covered by the accompanying dispatch mode.
    """
    import torch

    original = torch.Tensor.__new__
    base_new = torch._C._TensorBase.__new__
    previous = sys.getprofile()
    counts = dict(tensor_new=0, base_new=0)

    def constructor(cls, *args, **kwargs):
        counts["tensor_new"] += 1
        return original(cls, *args, **kwargs)

    def profile(frame, event, arg):
        if event == "c_call" and arg is base_new:
            counts["base_new"] += 1
        if previous is not None:
            previous(frame, event, arg)

    with patch.object(torch.Tensor, "__new__", staticmethod(constructor)):
        sys.setprofile(profile)
        try:
            yield counts
        finally:
            sys.setprofile(previous)
