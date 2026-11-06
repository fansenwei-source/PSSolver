"""Check the currently enforced subset of ARCHITECTURE section 3."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


if __package__:
    from ._source import imports, module_name, read_toml
else:
    from _source import imports, module_name, read_toml


def check(root: Path) -> list[dict]:
    contracts = read_toml(root / "pyproject.toml")["tool"]["pssolver"]["import-contracts"]
    violations = []
    for layer, forbidden in contracts.items():
        for path in sorted((root / "pssolver" / layer).rglob("*.py")):
            module = module_name(path, root)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for line, target in imports(tree, module, path.name == "__init__.py"):
                for prefix in forbidden:
                    if target == prefix or target.startswith(prefix + "."):
                        violations.append(
                            {
                                "file": path.relative_to(root).as_posix(),
                                "line": line,
                                "target": target,
                                "forbidden": prefix,
                            }
                        )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    violations = check(args.root)
    print(json.dumps({"passed": not violations, "violations": violations}, indent=2))
    return bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
