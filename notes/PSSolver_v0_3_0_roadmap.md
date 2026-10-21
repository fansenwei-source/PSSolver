# PSSolver v0.3.0 目标与完成标准草稿

日期：2026-10-21
路线图步骤：S-02
状态：等待用户决定

## 总目标

v0.3.0 应尽可能解决当前已登记的已知问题，同时保持科学正确性、旧 checkpoint
可读取、公共接口可审计、PSSolver-Control 独立兼容以及失败关闭语义。

建议采用 **issue-first**：先依次完成 S-03 至 S-09，除修复登记问题所必需的工作外，
暂不扩展新的公开几何、边界、lifting 或模型组合。S-10 再决定 D-0007 的三个
延期/接受项。

## 实施顺序

1. S-03：统一公共异常和 metadata 状态；
2. S-04：修复 checkpoint/NFS/权限/清理竞态，并重跑 N-1 矩阵和真实 NFS；
3. S-05：设计并实现带摘要的 checkpoint 进度完整性，同时兼容读取 v0.2；
4. S-06：修复压力容差、异常 pickle 和 float32 fixed-work；
5. S-07：清理 wheel、补齐规范化 metadata、诊断与文档；
6. S-08：补历史 fixture 和 CPU 覆盖；
7. S-09：集中完成 CUDA 长时程 Hermitian、TF32 backward 和相关 H100 资格；
8. S-10：复核 lifting、架构冗余和 generation label 三项决定。

## v0.3.0 完成标准

- 登记表中 S-03 至 S-09 的问题全部获得独立 verified/closed 证据；S-10 有用户
  确认决定；不存在未分流的新问题。
- 公共执行、restart、checkpoint 和收敛错误使用一致的公共异常与有限、可序列化
  metadata；异常退出不会遗留 `running` 状态。
- v0.2 正式 fixture 可只读加载且续算符合冻结合同；新格式不会改写旧源；NFS、并发、
  tamper、权限和故障注入全部 fail-closed 且无残留。
- 数值有限性、压力残差、transform count、Hermitian、梯度/Taylor、restart 和确定性
  门禁不放宽。
- 源码与 installed-wheel CPU suite、wheel/sdist 内容、公开导入、命令行入口、fixture
  身份和文档绑定全部通过。
- 使用预注册的既有平台门限完成 Plane、Periodic、Channel 的配对性能/显存非回退。
- PSSolver-Control 继续只依赖公开 API，并通过 installed-wheel CPU 与受影响 H100 路径。
- S-09 在一次集中、预注册、逐次授权的资格任务中覆盖 CUDA 长时程、TF32 backward、
  restart、梯度、显存和非回退。
- 冻结 `v0.3.0rc1` 后完成独立只读验收；没有发布阻断项才发布 v0.3.0。

## 需要用户决定的问题

### Q-01：已知问题与新功能谁优先？

- **A（建议）**：S-09 前只处理已知问题，不加入无关的新公开功能。
- B：S-04 后允许不扩大资格面的窄范围新功能。
- C：新功能与问题修复全面并行。

建议 A：可以避免修复过程中接口和资格面持续移动。

### Q-02：0.3 是否引入带摘要的新 checkpoint schema？

- **A（建议）**：新 writer 使用带摘要格式，同时显式兼容只读加载所有正式 v0.2
  checkpoint。
- B：新格式仅 opt-in，0.3 默认仍写旧格式。
- C：推迟到 0.3 之后。

建议 A：协调一致的进度篡改无法仅靠旧格式可靠检测，0.3 是合适的兼容边界。

### Q-03：性能和显存资格采用什么强度？

- **A（建议）**：预注册并复用最近一次已接受的、按平台和路径区分的门限；更改门限
  必须另有科学说明和用户决定。
- B：要求没有任何可测变慢。
- C：只记录，不作为发布门禁。

建议 A：既能阻止静默回退，也不会在实现后临时创造更严或更松的标准。

### Q-04：Control 兼容是否作为 PSSolver v0.3.0 发布门禁？

- **A（建议）**：要求 Control installed-wheel CPU 和受影响的 S-09 H100 路径通过。
- B：只要求 Control CPU。
- C：Control 不参与 PSSolver 发布门禁。

建议 A：Control 是 stable functional API 最重要的独立消费者，但仍保持两个项目分离。

### Q-05：checkpoint/provider identity 的兼容承诺是什么？

- **A（建议）**：读取全部已有正式 v0.2 fixture、不改写源、要求 restart 等价，identity
  变化必须显式迁移。
- B：只兼容 v0.2.2 及以后。
- C：0.3 与 v0.2 明确不兼容。

建议 A：最大限度保护已有模拟数据，同时允许 0.3 写出更安全的新格式。

### Q-06：S-10 如何处理 D-0007 三项？

- **A（建议）**：S-09 后，如果能独立验证，则把 R1-R9 加入 0.3 最后清理；lifting
  仍等到明确重开时再处理；A2-11 保持已接受设计限制。
- B：三项全部纳入 v0.3.0。
- C：D-0007 完全不变，三项都不进入 v0.3.0。

建议 A：尽量清架构债务，同时不贸然重开尚未资格化的 lifting 物理，也不重做已接受的
label 合同。

### Q-07：发布序列如何安排？

- **A（建议）**：S-03 至 S-10 完成后冻结 v0.3.0rc1，独立验收通过后发布 v0.3.0。
- B：内部资格完成后直接发布 v0.3.0。
- C：无论是否发现问题都预设多个 RC。

建议 A：既保留清晰的第二方证据边界，也避免无必要的多轮候选版本。

如果你同意全部建议，可以直接回复：`Q-01 至 Q-07 全部选择 A`。确认后 Codex 会把
草稿冻结为 S-02 合同，并创建 `R-0007` 交 Claude 做一致性检查。

## 范围声明

本草稿只新增计划和绑定测试，不修改 runtime、打包、问题登记表、路线图顺序、阈值、
skip/xfail、PSSolver-Control、nematics3d、tag 或 Release，也不运行 HPCC/GPU/H100。
