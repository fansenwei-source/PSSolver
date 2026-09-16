"""Compatibility CLI for the supported Plane Beris--Edwards application.

The application implementation lives in
``pssolver.applications.plane_beris_edwards``.  This historical path remains
available for benchmark scripts and archived run commands.
"""

from pssolver.applications.plane_beris_edwards import main


if __name__ == "__main__":
    raise SystemExit(main())
