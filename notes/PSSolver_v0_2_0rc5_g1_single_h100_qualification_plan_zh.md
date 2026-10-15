# PSSolver v0.2.0rc5 G-1 单次 H100 累计资格计划

## 当前结论

PSSolver-Control 的 rc5 Periodic identity migration 与 installed-wheel CPU 资格证据通过独立复核：三提交职责分离，`src/` 没有变化，RC4.5.7 两份历史记录未改写，外部归档 `469/469` checksum 通过，独立定向复跑 `36 passed`。

Control qualified source 是 `c14e6b56c88081e7f9a5642e65ffd722eec7445b`；`2474592fa64d1b3243277c7439a16104f2cdeb9f` 只是 evidence-only commit。Provider qualified source 仍是 `674a8121779be72c95aa5dc9e2babca0ce7b93a9`。

因此可以冻结 H100 合同，但现在还不能提交 H100：两个分支尚未满足远端身份前置条件，rc5 专用 qualification runner 也尚未实现和完成 CPU synthetic/preflight 验证。

## 单次 H100 的累计范围

唯一正式 Job 必须按顺序完成：

1. 固定 commits、archives、wheels、旧 CPU 证据和 manifests；
2. fresh installed-wheel 环境以及逐项持久化的 H100 preflight；
3. 七个精确 CUDA-only tests；
4. 六种 caller state × 八类 runtime 的 48 项 TF32 状态恢复矩阵；
5. 当前 checkpoint 的 CPU↔CUDA 双向 restart，以及 rc1–rc4 的 13 项已发布 checkpoint disposition matrix；
6. 八类 runtime 的数值执行门禁；
7. rc4/rc5 两角色、八类 runtime、两个网格、三次 trial，共 96 个 profiles；
8. installed PSSolver-Control 的完整 H100 suite、rc5 identity、checkpoint、gradient gates；
9. fail-closed analyzers、checksum 和最后的 `COMPLETE`。

性能与显存门限沿用已资格化的 3% group / 5% paired 合同，不得放宽。

## 数值比较的重要修正

不能把 rc4 与 rc5 所有 final spectral state 都要求逐字节一致。rc5 明确修复了 Periodic stored-plane Hermitian 语义，这一变化已经进入 provider identity。

- 对语义未改变的 Plane/Channel 路径，versioned analyzer 可以继续要求逐字节一致；
- 对 Periodic repaired path，比较 physical Q/u/p，relative-L2 与 Linf 都不得宽于 `1e-12`；
- 同时要求 rc5 stored-plane Hermitian violation 精确为零；
- 不允许用旧的 global final-state SHA equality 错误否决正确修复。

## 当前停止点

下一步只允许实现并在 CPU 上验证 versioned rc5 qualification support，范围限定为 `benchmarks/`、`jobs/`、qualification `tests/` 和 forward-only `notes/`。不得修改 `pssolver/` 数值实现、Control source、nematics3d、阈值、skip/xfail、默认值或 schema。

当前没有授权 push、H100、自动重试、merge、tag、release 或 PyPI。

H100 PASS 只会使 rc5 进入 release integration review；不会自动 merge/publish，且 post-release verification 仍需在发布后完成。
