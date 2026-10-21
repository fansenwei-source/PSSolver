# PSSolver 0.3 S-01 登记表迁移

本步骤依据用户确认的 `D-0010`，只迁移问题登记目标，不修改运行时。

- 登记表主副本改为 `dev/0.3`；
- 20 个仍为 `registered` 的条目统一以 `0.3` 为目标；
- `A2-21/S2-12` 明确归入 `0.3` 的设计先行步骤 `S-05`；
- `B30` 明确归入文档批次 `S-07`；
- `NEW-14` 的目标改为 `0.3 GPU verification`；
- `V-17` 继续保持 `verified`，其修复提交、复核编号和历史证据均不变。

R-0003、R-0004 和 R-0005 的历史记录没有改写。对应测试现在验证这些记录在产生时
保存的旧登记表哈希，而不再错误要求活动登记表永久保持旧字节。

本步骤没有修改 `pssolver/`、打包输入、数值实现、checkpoint schema、provider identity、
生产默认值、tag、Release、资产、PSSolver-Control 或 nematics3d，也没有运行 HPCC、GPU 或 H100。
完成本地门禁后将创建 `R-0006`（`plan_step = S-01`），推送仍需另行授权。
