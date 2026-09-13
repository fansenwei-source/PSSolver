"""CLI for the Stage L production-versus-shadow trajectory gate."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pssolver.experimental.shadow_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
