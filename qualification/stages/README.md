# Frozen stage sources (S-24 / R-0073)

All 35 Python files were moved without changing bytes. `manifest.json` binds
their original paths, SHA-256 and disposition. These are process artifacts,
not package modules; neither wheel nor sdist includes them.

Group A (7): stage_m_plan, stage_n_plan, stage_n1_plan, stage_n2_plan,
stage_n3_plan, stage_n4_plan, stage_o4_plan. Only these names are mapped by the
explicit test/qualification loader, under their original fully-qualified names.

Group B (28): every other file in the manifest. Frozen, **not importable in
0.5**; runnable in the v0.4.0 source checkout
`4c12228aba01e8483e80d28a1471ab72384d5fda`. They depend on retired implementations.
Do not add them to the loader or copy their dependencies into the current package.
Their historical audits are retained through `historical_stage_audit.py` and its
exact file/blob/nodeid manifest. Frozen benchmark and scripts_plane consumers
remain byte-identical and historical-only pending S-30 disposition.
