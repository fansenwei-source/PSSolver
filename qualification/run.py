"""Explicit entry point for independent A-group frozen stage tools only."""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "_pssolver_qualification_legacy_imports", Path(__file__).with_name("_legacy_imports.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()
    sys.argv = [str(args.script), *args.arguments]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
