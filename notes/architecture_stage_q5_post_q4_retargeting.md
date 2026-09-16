# Architecture Stage Q.5: post-Q.4 retargeting

Stage Q.5 closes the rejected Q.4 preallocated-workspace experiment and
selects the scope of a possible Q.6 diagnostic.  It is an analysis-only
continuation of the Stage Q performance investigation.  It is not Stage R and
does not consume or rename any part of the previously reserved R--W
architecture roadmap.

The frozen Stage Q.5 boundaries are:

- reuse only the completed O.4.3.1, O.4.3.3, O.4.3.4, Q.2, Q.3, and Q.4
  reports;
- distinguish allocator reuse from tensor data movement;
- keep rejected or neutral mechanisms closed unless new evidence explicitly
  reopens them;
- authorize at most a Q.6 diagnostic design, never a candidate
  implementation or production promotion;
- do not execute a solver, read large trajectory arrays, or change numerical
  behavior.

## Roadmap namespace

The architecture namespace after this correction is:

1. Q.5: post-Q.4 evidence closure and retargeting;
2. optional Q.6.x: bounded diagnostics or performance experiments selected by
   Q.5;
3. R--S: resume the original Plane production-integration roadmap;
4. pause after S and before T;
5. T--W: resume the original cross-geometry and framework-completion roadmap
   after the pause.

The S/T boundary is the intentional pause point because Plane production
integration and entry-point consolidation should be complete before Channel
or other geometry migration begins.  Work performed during the pause must use
separate branches/worktrees and must not silently alter the frozen Plane
production baseline.

## Current decision

Q.4 showed that preallocated batch workspaces preserve numerical equivalence
and correct tensor lifetimes, but do not improve R320 throughput and increase
peak allocated memory.  Stage Q.5 therefore keeps that route closed.  Its
primary diagnostic target is end-to-end algebraic data movement and consumer
layout; allocation reuse by itself is not a candidate mechanism.
