# PSSolver development instructions

Policy ID: `pssolver-python-engineering-v1`.

## Use the engineering skills before writing code

For each substantive Python design, implementation, refactoring, or testing task,
load these installed skills before making design decisions or editing Python:

- `python-design-patterns`: assign responsibilities and choose the simplest design.
- `python-project-structure`: place modules, define dependencies and public APIs.
- `python-code-style`: write readable names, types, docstrings, and consistent code.
- `python-testing-patterns`: decide the correctness evidence and tests alongside the design.

Apply all four throughout the task, rather than waiting for a final review.
Use Codex's skill loading/reading mechanism or Claude Code's Skill tool.
If discovery is unavailable, read the installed `SKILL.md` files under
`~/.codex/skills/<name>/` or `~/.claude/skills/<name>/`; follow relevant references.
Report a missing skill honestly and continue independent authorized work.
At the start, briefly name the skills used; for a small fix, a short inline plan
is enough. Documentation-only or administrative tasks need not load all four.

Project-specific rules below and explicit user requirements take precedence over
generic skill examples. A skill is guidance, not proof that checks have passed.

## Design and structure

- Before implementation, identify the public contract, module ownership, numerical
  assumptions, compatibility impact, and how the change will be verified.
- Use simple cohesive functions and composition. Introduce a new abstraction only
  for a demonstrated need; avoid speculative plugin systems and class hierarchies.
- Separate mathematical kernels from configuration, runtime orchestration,
  persistence, diagnostics, and command-line handling. Keep entry points thin.
- Reuse existing mechanisms after inspecting the current checkout. Preserve
  deliberately independent analytical references and numerical oracles.
- Distinguish supported public APIs, compatibility facades, and experimental code.
  Preserve supported imports and checkpoint formats; use explicit migrations or
  deprecations for intentional changes. `__all__` documents exports, not privacy.
- Confirm the branch/commit before work: this project has multiple worktrees.
  Follow the checkout's capability catalog and release scope; do not promise
  arbitrary PDE/geometry support or reorganize unrelated modules.

## Readable scientific Python

- Use descriptive English identifiers at public boundaries and conventional
  `snake_case` / `PascalCase`. Mathematical kernels may retain standard symbols
  such as `Q`, `dt`, or `k` when their meaning is documented.
- Give new/changed public interfaces useful type annotations. Prefer concrete
  types or small Protocols to `object` when attributes/methods are required.
  Do not hide type problems with blanket `Any` or broad ignores.
- Prefer NumPy-style docstrings for new scientific public APIs, while preserving
  an established local convention. Document shape and axis order, dtype/device,
  units or nondimensionalization, boundary conditions, mutability, and errors
  where relevant. Explain mathematical choices rather than narrating syntax.
- Follow repository formatting configuration. If absent, target 100 columns for
  new code and keep surrounding style consistent; avoid unrelated reformatting.
  Respect the Python minimum in package metadata (currently 3.10), rather than
  copying a skill's example Python version or strictness setting.
- Make physical terms and data flow understandable. Splitting expressions must
  preserve required floating-point order, compilation behavior, and performance.

## Tests and evidence during implementation

- Define expected behavior before or alongside implementation. Add focused tests
  for new behavior and bug fixes, including relevant invalid inputs and edge cases;
  do not add tests that merely repeat the implementation or assert formatting.
- Choose independent numerical evidence where applicable: manufactured solutions,
  invariants, spatial/time convergence, finite-difference gradient checks,
  restart equivalence, and relevant dtype/backend cases. State tolerance rationale.
- Preserve physical conventions, independent oracles, numerical tolerances,
  checkpoint identities, exception semantics, and performance contracts unless
  the task explicitly calls for changing them. Never loosen a gate to obtain PASS.
- Distinguish ordinary functional/numerical tests from historical evidence audits.
  Some audit tests need complete Git history and frozen records. Report missing
  prerequisites separately; do not silently skip required release evidence or
  rewrite historical records to accommodate a formatting change.
- Before completion, run relevant Ruff checks, scoped mypy checks, and pytest
  regression tests for the affected behavior. Prefer project configuration and
  the project's scientific Python environment; the engineering tool environment
  alone may not include Torch/NumPy. Tools may be installed under
  `~/.codex/skill-envs/python-engineering/bin/`.
- If configuration is absent, make the limited rule set/type-check scope explicit.
  Separate pre-existing diagnostics and missing dependencies from new failures.
  Avoid whole-repository auto-fixes or blanket strict typing during a small task.
- For packaging/public entry changes, test the installed wheel outside the source
  tree. For hot numerical paths, include the applicable performance validation.
- Report the actual commands, results, limitations, and checks not run. A local
  CPU check does not establish GPU/H100, long-run, NFS, or release qualification.

## Existing collaboration and change scope

Apply these skills within the current assigned role. Where the local PSSolver
handoff workflow is in use, preserve its confirmed decisions and ownership:
Claude normally handles design/analysis and Codex handles implementation/execution.
The current explicit user task determines the authorized work; these instructions
do not grant new publishing, compute-job, or delegation authority, and do not add
an extra approval step to already authorized routine work.

Keep changes focused and preserve unrelated user edits. Follow applicable project
handoff/verification rules when a task involves the registry, frozen evidence,
release process, or shared resources. Keep personal paths and private research
details out of library interfaces and distributable project documentation.
