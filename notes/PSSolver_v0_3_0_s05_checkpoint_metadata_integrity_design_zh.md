# PSSolver v0.3.0 S-05 checkpoint metadata 完整性设计草案

状态：`pending_user_confirmation`

本草案只做设计，不修改 runtime、checkpoint 字节、schema、provider identity、数值实现或
生产默认值。路线图的总体选择已经由 `D-0011` 确认：v0.3 默认写带摘要的新格式，继续只读
加载全部正式 v0.2 checkpoint，并要求 restart 等价。当前需要确认的是具体实施边界。

## 问题与现状

`A2-21/S2-12` 的核心不是单个计数器缺少约束，而是 Plane format 2 与 legacy Channel
format 2 的 `checkpoint.json` 没有覆盖整个 metadata 的摘要。攻击或误编辑如果同时修改
`completed_steps` 与 integrator 计数，使字段仍自洽，单靠结构检查无法发现。

进一步审计发现，finite-Q anchoring format 1 和 experimental shadow format 1 也只校验
identity/tensor 文件，没有把进度和完整 tensor manifest 一起密封。Periodic、complete-stress
Channel、当前两个 Functional carrier 已使用 `metadata_sha256`；reference 与 modal-block
格式通过 `COMPLETE` 中的 manifest SHA-256 覆盖全部进度，无需改格式。

## 推荐合同

1. 复用 `seal_checkpoint_metadata()` / `verify_checkpoint_metadata()`：对完整 canonical JSON
   计算 SHA-256，只排除 `metadata_sha256` 字段自身。
2. Plane 2→3、legacy Channel 2→3、finite-Q 1→2、experimental shadow 1→2。
3. 新格式必须先验证 metadata seal，再打开任何 tensor；失败统一为
   `CheckpointIntegrityError`，且不得修改 target、启动 timestep 或创建输出目录。
4. v0.3 公共 writer 只写最新密封格式，不继续生成新的无摘要 checkpoint。
5. 旧格式按精确版本分支只读加载，源目录不变；加载结果标记为 `unsealed_legacy`，不能
   宣称旧文件在历史上未被修改。
6. 提供非原地的 carrier upgrade；升级只是密封“已经按旧格式验证到的当前内容”，不把
   SHA-256 描述成签名、MAC 或真实性证明。
7. checkpoint carrier 版本变化不进入科学/兼容 identity；provider identity、Functional API
   版本、数值状态和生产默认值保持不变。

## 必须通过的门禁

- 对每个新格式分别篡改单个进度字段、全部自洽进度字段、tensor record、backend restart、
  identity、摘要本身以及 key 集合；全部须在 payload load 与 target mutation 前拒绝。
- 未修改的新格式必须 checkpoint/restart 与连续运行一致。
- v0.2 冻结 fixture 的原始 tree hash 不变，全部可读并满足已有 continuation oracle。
- 运行既有 checkpoint identity、migration、NFS、并发、故障注入、完整 CPU 和 installed-wheel
  CPU 门禁；不增加 skip/xfail，不放宽阈值。
- metadata-only 逻辑不单独申请 H100；跨设备 checkpoint/restart 留在 S-09 集中验证。

## 需要用户决定的问题

### S05-Q01：覆盖范围

- **A（建议）**：覆盖 Plane、legacy Channel、finite-Q 与 experimental shadow 四个未密封 writer。
- B：只覆盖三个 public workflow writer，把 experimental shadow 留到 S-07。
- C：只覆盖原报告点名的 Plane 与 legacy Channel。

### S05-Q02：旧格式写入

- **A（建议）**：v0.3 公共 writer 只接受最新密封格式；旧格式只能来自冻结 fixture/test helper。
- B：允许显式 `legacy=True` 写旧格式。
- C：保留当前低层 writer 可写旧格式的行为。

### S05-Q03：旧格式读取

- **A（建议）**：按精确格式版本自动分流、只读、不改源，并标为 `unsealed_legacy`。
- B：每次读取都必须显式传 `allow_unsealed_legacy=True`。
- C：读取前必须先显式迁移。

### S05-Q04：完整性状态可见性

- **A（建议）**：header/checkpoint 暴露稳定只读状态 `sealed_sha256` 或 `unsealed_legacy`，但不进入
  compatibility identity。
- B：只写文档，不提供程序化状态。
- C：每次旧格式读取发 warning。

### S05-Q05：迁移入口

- **A（建议）**：四个受影响格式都提供非原地 source→target upgrade。
- B：只给 Plane 与 legacy Channel 提供 upgrade。
- C：不提供 upgrade，用户只能恢复后在未来步骤重新写 checkpoint。

建议五项均选 A。用户确认前不开始实现，也不创建 Claude review request。
