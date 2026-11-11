"""Compatibility CLI for the supported Plane Beris--Edwards application.

The application implementation lives in
``pssolver.cli.plane``.  This historical path remains
available for benchmark scripts and archived run commands.
"""

from pssolver.cli.plane import main


if __name__ == "__main__":
    raise SystemExit(main())
