"""Profile the opt-in Stage M Plane shadow runtime on H100."""

from pssolver.experimental.h100_shadow_qualification import profile_main


if __name__ == "__main__":
    raise SystemExit(profile_main())
