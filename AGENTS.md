# PSSolver development instructions

Policy ID: `pssolver-python-engineering-v2`. Read `ARCHITECTURE.md` first; it is the
map every change must fit.

## Skills
- For substantive Python design, implementation or testing, load `python-design-patterns`,
  `python-project-structure`, `python-code-style` and `python-testing-patterns` before
  deciding; add `refactor` for structural clean-ups. Name the skills used at the start.
- Use `sympy` for manufactured solutions and symbolic checks. Derivation scripts live in
  `tests/mms/`, never in the package. Record the SymPy version and expression hashes.
- Project rules below override generic skill examples. A skill is guidance, not evidence.

## Project rules that override generic advice
- Old and new paths stay **bitwise identical** (CPU, float64 and float32, N steps
  with restart) unless a recorded project decision says otherwise. Keep existing
  tolerances; never loosen a gate; never add skip/xfail.
- Independent references (analytic solutions, MMS sources, gradient oracles) are
  deliberately duplicated. Do not merge them with production kernels.
- Source changes alter provenance hashes. Register forward successors; never rewrite
  frozen records or fake old hashes.
- No hard limits on function length or coverage. Split by responsibility and
  mathematical step, preserving floating-point order and performance.
- Formatting: repository ruff configuration, 100 columns, Python 3.10, NumPy-style
  docstrings for scientific public APIs (shape, axis order, dtype/device, units,
  boundaries, mutability, errors).
- Process artifacts (stage plans, qualification analyses, diagnostics) go to
  `qualification/`, not into the package.
- Library code and distributed documents contain no personal paths and no private
  research details.

## Before finishing a change
- State which `ARCHITECTURE.md` section the change touches; update the file if a layer
  boundary moved.
- Report: whether a parallel path or copy was added (and its retirement plan), which
  existing mechanism was reused, net lines added/removed (package and tests separately),
  and how a reader gets from the entry point to the mathematics.
- Run `tools/check.sh` (ruff on changed files, import contracts, scoped mypy, CPU quick
  regression) plus the tests for the affected behaviour. For packaging or public entry
  changes, test the installed wheel outside the source tree. A CPU check does not
  establish GPU/H100, long-run or release qualification.
- Report the actual commands, results and checks not run.

## Collaboration
- Confirm the branch and commit first; this project uses several worktrees.
- Follow the current task and the established ownership: design and review by one
  agent, implementation by another, as assigned. These instructions grant no
  publishing, compute-job or delegation authority.
- Commit only explicit paths; preserve unrelated edits.

For a package-root export, keep its Google-style docstring and complete
annotations with the actual definition. Describe accepted inputs and failure
conditions without adding imports merely to resolve annotation names. A
documentation change does not authorize a signature, default or behavior change.
