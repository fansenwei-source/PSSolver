# PSSolver v0.2.0rc4 RC4.5.0 跨 runtime 鲁棒性规划

## 结论

RC4.4 已正式关闭：B25、B28、B29 在 installed-wheel、Consumer、CUDA correctness、restart、负向门禁以及 12 项 H100 profile 中全部通过。权威记录是 `PSSolver_v0_2_0rc4_rc447_h100_closure.json`，对应 Job 10878767，归档 manifest 为 1534/1534 PASS。

RC4.5.0 现在完成的是 **planning-only**。它把 rc2 复核留下的最后九个 release-local residual 收拢为四个互不混淆的 cohort，并规定先做 RC4.5.1 analysis-only 重放，再冻结合同、修改源码。当前没有授权实现、Control 修改、H100、默认值晋升、自动 merge 或发布。

下一步是 RC4.5.1，而不是直接修代码。

## 为什么先分析

这些问题最早来自 rc2 的复核，但此后 checkpoint identity、lifting、pressure、public `Simulation` 和 functional API 都发生过有资格记录的演进。旧复现仍然是重要线索，却不能直接当作当前 HEAD 上的事实。

RC4.5.1 必须对每项给出下列四种裁决之一：

- `REPRODUCED_REPAIR_REQUIRED`：当前候选仍可复现，进入实现；
- `ALREADY_CLOSED_WITH_NEW_EVIDENCE`：后续改动已实质关闭，补回归后关闭；
- `PUBLIC_BOUND_REQUIRED`：算法只在明确有限范围内可靠，需要公开、验证并 fail closed；
- `ACCEPTED_LIMITATION_WITH_EXPLICIT_CONTRACT`：行为有意保留，必须改成可检查的公共合同。

不能只靠读代码关闭问题，也不能从 CPU 推断 CUDA 行为。

## 四个 cohort

### 1. Checkpoint transactionality：B7、B24、B27

需要统一验证：

- v2 元数据不能通过删除校验信息并伪装成 v1 来静默降级；
- tensor 相对路径必须规范化并被限制在 checkpoint 根目录内；
- 需要文件名身份的生产格式必须显式验证 canonical filename；
- extra-key 策略必须明确，不能不同 reader 各自猜测；
- 所有 restore 验证在 tensor load、target mutation 和 timestep 之前完成；
- 所有 writer 都先写 staging，再原子发布；
- 发布权限服从创建模式和调用者 umask，不硬编码绕过 umask 的 `0755`；
- 失败不留下目标目录、半成品或 staging 残留；
- not-found、exists、integrity、compatibility、I/O 进入稳定的公共异常分类。

这一步不授权 R2 的“大一统 checkpoint I/O”架构重构。允许共享最小 primitive，但不能借机改 tensor layout 或 checkpoint 数值身份。

### 2. Boundary semantics：B9、B21

B9 要回答一个数学问题：对 Q anchoring，`n` 与 `-n` 产生相同的二阶取向张量时，公共 API 是否应接受两种符号。无论最终选择如何，metadata 和错误信息都必须给出 axis、side、期望向量以及允许的等价关系。

B21 要对对称强 Robin impedance 做尺度扫描，并使用独立高精度参考。只有两种诚实结论：

- 算法能稳定覆盖预注册范围，并保持 root 有限、严格递增、残差受控；或
- 发布一个有限支持上界，并在 operator materialization 前明确拒绝。

不能把“某些大数大概不常用”当作隐含范围。

### 3. Public validation and observability：B13、B15、B17

这里不要求把四种 application 的科学 diagnostics 强行做成同一列结构；正确目标是：保留 family-specific payload，在外面增加统一、版本化的 envelope，至少记录 family、runtime、enabled、count、steps、artifact paths、payload schema 和 units。

同时建立一张 capability/catalog/compiler/lowering/exception truth table：

- 非法 device 与 pressure option 必须在创建输出目录前拒绝；
- catalog 声称可用的组合必须确实能编译；
- 不支持的组合必须显式拒绝，而不是退化成另一物理边界；
- 公共 compile 入口不能泄漏内部 `SimulationLoweringError`、裸 `ValueError` 或 `TypeError`；
- 已有 diagnostics 文件名和 family payload 不在 RC4.5 中删除。

### 4. Accelerator global state：B11

需要同时处理三个 caller-owned PyTorch 全局状态：

- `torch.get_float32_matmul_precision()`；
- `torch.backends.cuda.matmul.allow_tf32`；
- `torch.backends.cudnn.allow_tf32`。

PSSolver 可以在自己拥有的执行区间内应用请求策略，但 normal return、exception 和 nested call 都必须按栈恢复调用者状态。metadata 要区分 requested、effective、before 和 restored，不得声称拥有整个进程的全局状态。

这项需要真正的 H100 证据；登录节点上的 `CUDA unavailable` 不能作为证明。

## 分阶段顺序

1. **RC4.5.1**：九项 residual analysis-only 重放；
2. **RC4.5.2**：依据重放结果冻结公共 robustness 合同；
3. **RC4.5.3**：B7/B24/B27 checkpoint transactionality 与异常实现；
4. **RC4.5.4**：B9/B21 boundary semantics 与 Robin 支持范围实现；
5. **RC4.5.5**：B13/B15/B17 validation、capability、diagnostics 实现；
6. **RC4.5.6**：B11 scoped TF32 state 实现；
7. **RC4.5.7**：installed-wheel CPU 累计关闭，并按实际 public/functional delta 裁决是否需要 Consumer 重验；
8. **RC4.5.8**：一次单独授权的 H100 TF32 与累计非回退关闭。

这种顺序把 CPU 可裁决问题与必须在 GPU 上观察的全局状态问题分开，也避免在正式 H100 结果出现后再修改阈值。

## Consumer 边界

RC4.5.0 不授权修改 PSSolver-Control。只有出现以下任一变化，才需要单独冻结 Consumer commit 并重验：

- functional checkpoint schema、异常分类或 restore 语义变化；
- functional diagnostics 公共 schema 变化；
- functional runtime identity 或 capability 声明变化；
- TF32 的执行边界改变 Consumer 可观察行为。

纯 production 权限修复、只改错误信息而不改异常类型、或只文档化未改变的 Robin 范围，不自动触发 Consumer H100。

## 不在 RC4.5 做的事

- 不改 Beris–Edwards 方程、时间积分器、谱变换或 pressure 算法；
- 不添加模型、几何或边界类型；
- 不执行 R1–R9 的 post-0.2 架构合并；
- 不修改 nematics3d；
- 不修改生产默认 runtime；
- 不授权自动 merge、默认值晋升或发布；
- 不提交 H100 Job。

RC4.5.0 完成后，唯一合理的下一步是按计划生成 RC4.5.1 的只读 reproducer 与 evidence schema，再执行 analysis-only 裁决。
