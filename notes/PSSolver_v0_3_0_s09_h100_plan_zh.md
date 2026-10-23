# PSSolver v0.3.0 S‑09 GPU/H100 资格计划（授权前草案）

日期：2026-10-23
路线图步骤：S‑09
状态：四项决定已确认，本地实现中；未授权 GPU/H100

## 先行诊断

S‑09 的 NEW‑14 不是单纯缺少一次 GPU 运行。当前 Periodic 与 Channel
functional runtime 只在 `step`、`observe` 和 `step_and_observe` 的同步调用期间进入
TF32 scope；这些调用返回后恢复调用者状态，而 autograd backward 随后才运行。

使用真实的两条 functional 图，把调用者设置成 `high / TF32 on` 后，Periodic 和
Channel 的 backward hook 都观察到调用者状态，而不是 runtime identity 声明的
`TF32 off`。因此在解决公共 backward 所有权前直接提交 H100，只会把已知的合同缺口
带入正式作业。

## 已确认决定

用户已确认四项全部选 A：

1. `S09-Q01`：给 stable functional runtime 增加公开 `execution_scope()`；Control 在同一
   scope 内完成 forward、checkpointed replay 和 backward。成功、异常与嵌套退出都精确
   恢复调用者状态。只在作业脚本全局关 TF32 不能修复普通消费者的执行合同。
2. `S09-Q02`：以最后一次 H100-qualified 的稳定 Consumer source
   `208edd03e6b26a0e788c07d295fddd7a67237189` 为基线，做最小 execution-scope 与 identity
   迁移；不把仍在变化的 `research/3d-warmup` 研究代码混入发布资格。
3. `S09-Q03`：复用最后接受的八 runtime family、两尺度、每格六 trial、交替顺序的
   192-profile 合同及 1.03/1.05 性能和 1.03 显存门限，不删样本、不替换 trial。
4. `S09-Q04`：CUDA 上运行 loop3d half/full 与 R1 half 三条轨迹，全部到 `t=200`。

`execution_scope()` 是新的公开协议方法，因此 functional API 从 `1.0`
升到 `1.1`，Provider runtime identity 也随顶层 `api_version` 改变。它不改变
checkpoint 的 forward-compatibility identity；当前格式的 API-1.0 checkpoint
由显式只读兼容 reader 恢复，不声称为精确 API-1.1 身份。只有在
`execution_scope()` 中完成的 forward、replay 和 backward 受 runtime 声明的
TF32 策略约束；离开该作用域后再执行 backward，仍使用调用者的 TF32 设置。

Control 最小迁移必须从 `208edd03e6b26a0e788c07d295fddd7a67237189`
建立独立分支和 worktree；不得切换或修改正在使用的
`/home/fansenwei/Desktop/Develop/PSSolver-Control` 与 `research/3d-warmup`，Control
的 push 仍需单独授权。

## 唯一 H100 作业

用户确认决定、Provider 实现、Consumer 迁移、installed-wheel CPU 门禁和 helper 的
synthetic negative gates 全部通过后，才可另行申请一次 H100 submission。作业顺序冻结为：

1. node-local 临时目录、环境、host compiler 与 installed-wheel 身份；
2. CUDA-only tests；
3. 三条 `t=200` Hermitian 轨迹；
4. 六种调用者状态下 Periodic/Channel forward+backward TF32 scope；
5. checkpoint/restart 与 released fixture；
6. 数值、梯度、Taylor、确定性 replay；
7. 独立 Control consumer；
8. 192 个 paired performance/memory profiles；
9. 累计 analyzer、checksums 与 `COMPLETE`。

在申请 H100 前，还要先用 CPU 对 Channel legacy、compiled-v2、
complete-stress 和 functional 路径的 tolerance/fixed-work 模式做单步耗时和
峰值内存对比，特别测量 S-06 新增的真实残差重算与状态保存。若预计
会超过已冻结的 `1.03/1.05` 耗时或 `1.03` 内存门限，必须在 H100
授权前停止报告，不得放宽门限。

任何阶段失败都立即停止。不得自动重试；不得在同一授权下提交第二个作业。

## 边界

本草案只新增 qualification 计划、生成器和测试。它不修改 runtime、Control、
`nematics3d`、生产默认值、阈值、tag 或 Release，也没有运行 GPU/H100 或执行 push。
机器可读计划由 `benchmarks/plan_v0_3_s09_h100.py` 生成。
