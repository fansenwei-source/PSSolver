---
project: pssolver
author: codex-pssolver
date: 2026-11-09
plan_step: S-24
status: candidate
---

# S-24：退役与历史证据边界

依据 D-0042、D-0047 和 R-0073 前向裁定实施。A 组为 7 个，B 组为
28 个；35 个过程模块逐字节迁到 qualification/stages，只有 A 组进入
测试侧加载器。B 组和其他退役 wiring 的原始测试在固定 v0.4.0 Git
检出中执行，不加载进当前 Provider。历史记录和冻结 benchmark 原样保留。

旧 Channel 顶层连接、separated-canary 连接及没有当前消费者的专用 helper
退役；canonical Stokes/PCG、完整应力方程和后续批次的执行/状态/规划基础设施
保持不变。三类退役 checkpoint 在读取任何张量之前失败关闭，提示使用
v0.4.x / maint/v0.4。Canary 和私有 lifting 夹具由指定 v0.4.0 wheel
生成，并在删除代码前单独提交。

保留组合的当前测试继续验证原来的数值、身份和错误契约；混合测试按变体拆分，
不放宽断言或容差。旧版本的来源/布局审计指向固定历史检出，新的来源哈希在
本 successor 前向绑定。此记录是候选说明，不代表 CPU、GPU 或发布资格通过。

不推送、不访问 HPCC、不使用 GPU、不实施 S-25。

同时导入 R-0071、R-0072 的 PASS 和 R-0073 的前向裁定；三份结果都没有
登记表更新。结果文件的 SHA-256 固定在 JSON 的 review_imports，不改写结果。
新增的删除/迁移表与冻结源码表逐项列出原路径、行数、理由及前后哈希。
RC6 两份仍在当前树中的记录测试接入既有 successor 解析器，原来的精确前驱
核对保留；它们继续拒绝未登记的哈希，不改历史记录里的任何值。
