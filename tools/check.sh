#!/usr/bin/env bash
# Select an isolated tooling installation with RUFF/MYPY; never install implicitly.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" -B tools/check.py "$@"
