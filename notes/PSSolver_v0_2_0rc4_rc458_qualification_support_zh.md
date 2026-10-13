# PSSolver v0.2.0rc4 RC4.5.8 qualification support

状态：`READY_RC4_5_8_H100_QUALIFICATION_SUPPORT`。

本地 qualification support 已实现统一的八 family profiler、96 项矩阵 runner、三组 caller state 的 H100 TF32 scope gate，以及 fail-closed cumulative analyzer。它们只属于资格验证层，没有修改 PSSolver runtime、数值方程、checkpoint schema、公开 API、生产默认值、PSSolver-Control 或 nematics3d。

实现前的真实构造审计发现并明确记录了两项外部合同错误。第一，`channel_legacy` 与 `channel_compiled_v2` 只能沿用已资格化的 float32 P7.3 构造合同；其余六条 family 使用 float64。第二，complete-stress/functional 路径不能继承 Plane 或 legacy Channel 的 transform-call 常数。A `51f47e9` 与 B `a195bd0` 的独立 tiny 单步 replay 得到完全相同的真实拓扑：Plane 7/32，legacy/compiled Channel 11/36，Periodic complete-stress 与 functional 7/46，Channel complete-stress 15/42，Channel functional 15/57。

新 profiler 同时区分 60 个正式计时步与包含 10 个 warmup 的 70 个 runtime 总步数，保存每步样本、显存、Dynamo graph break、compile fallback、runtime fallback、初末状态 SHA、transform count、installed import 路径和 caller TF32 状态。runner 为每个 profile 分配独立且初始为空的 Triton/TorchInductor cache，并拒绝包含 `pssolver/` source tree 的 support root。analyzer 要求 96/96 文件精确齐全、16 个 family/grid group 全部门限通过，且 auxiliary 中 CUDA-only、TF32 scope 和 Consumer H100 suite 全部通过。

Consumer 远端证据已就绪：qualified source 仍是 `b8539e0ec4c137a6c13746f6832e26e74be24515`；`0e3895d664d1b81925127848b2dfde638f39763d` 仅登记证据，不是新的 qualified source。

当前没有 push，也没有提交或运行 H100 Job。正式继续仍需用户分别授权 Provider branch push 和一次 RC4.5.8 H100 submission；不得自动重试。
