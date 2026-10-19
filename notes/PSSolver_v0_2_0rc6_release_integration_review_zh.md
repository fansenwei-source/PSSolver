# PSSolver v0.2.0rc6 发布集成审查

日期：2026-10-19

分类：`PASS_RC6_RELEASE_INTEGRATION_REVIEW_READY_FOR_FINAL_PACKAGE_PREPARATION`

## 结论

RC6 的运行时候选 `ea801ec08f5b2465661334e1c126133e55b68e96` 已完成累计单 H100 资格验证。资格支持提交为 `3228d3d723393818ee6149e9cf56021ffaaed411`，累计关闭记录提交为 `5fbf556c142a29896fe41d7cd991dc65a8149e1e`。

从运行时候选到累计关闭记录的 77 个改动文件只位于 `benchmarks/`、`jobs/`、`notes/` 和 `tests/`。这一段没有修改 `pssolver/`、版本文件、构建元数据、生产默认值、checkpoint schema 或数值实现。因此，经验证的运行时候选没有在资格阶段之后发生漂移。

## 累计 H100 证据

- Job：`10901236`，`COMPLETED`，`0:0`；
- 11/11 阶段完成；
- 192/192 profiles 有限，且 192/192 profile child TF32 bootstrap 通过；
- 16/16 性能与显存组通过；
- Consumer：344 passed，1 个已授权 skip，0 failed；
- 控制归档：24/24 checksum PASS；
- 正式归档：4763/4763 checksum PASS；
- `qualification_complete = true`；
- `rc6_complete = true`。

## 冻结的公共合同

- 包版本：`0.2.0rc6`；
- functional API：`1.0`；
- models / geometries / boundary policies：2 / 3 / 8；
- qualified combinations / runtime paths：4 / 6；
- functional batch size：仅 `[1]`；
- production default 未改变；
- compiled runtime 未提升为默认；
- checkpoint schema 未改变；
- lifting 未重新开启；
- PSSolver-Control 未并入 Provider；
- nematics3d 未修改。

## 授权边界

本审查只确认 RC6 已具备分支推送和最终发布包准备的技术条件。它没有执行或授权：

- push；
- merge；
- tag；
- GitHub prerelease；
- PyPI 发布；
- 自动默认提升。

最终 tag 目标、sdist/wheel 哈希及 installed-wheel CPU smoke 仍为空或 pending。下一门禁是：用户明确授权 push 后，从已推送、已审查的来源构建并验证最终 RC6 sdist 和 wheel。
