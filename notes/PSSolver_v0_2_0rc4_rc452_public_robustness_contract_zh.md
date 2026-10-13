# PSSolver v0.2.0rc4 RC4.5.2 public robustness 合同

## 状态

`PASS_RC4_5_2_PUBLIC_CROSS_RUNTIME_ROBUSTNESS_CONTRACT`

RC4.5.2 已把 RC4.5.1 黑盒重现的九项 residual 冻结为可实现、可测试的公共合同。本阶段只写合同和合同测试，没有修改 checkpoint、边界、编译器、diagnostics 或 TF32 的运行时实现，也没有提交 H100。

## 1. Checkpoint transactionality：B7、B24、B27

### Schema 与完整性

- current-format checkpoint 在顶层、identity、bridge 和 tensor record 中都拒绝未登记键；
- legacy 输入只能通过显式登记的 source-release + format-version reader；
- current functional checkpoint 必须携带 functional-origin marker；
- metadata SHA-256 只声明检测意外损坏、未完成编辑和未登记 downgrade，不宣称在没有外部 trust root 时阻止恶意重封印。

### 路径与恢复事务

- payload 只允许单个 relative basename，不允许绝对路径、`..`、路径分隔符或 symlink；
- resolved payload 必须保持在 checkpoint root 内；
- metadata、schema、digest、origin、路径、identity、checksum、shape、dtype 和 finite 全部通过后，才能一次性修改 target；
- 任何失败都不得修改 target 的 tensor、workspace 或 completed-step 状态。

### 写入事务与权限

- 所有 writer 必须在 target parent 下建立同文件系统的唯一 sibling staging；
- 所有内容成功后只用一次 atomic directory replace 发布；
- 目录权限为 `0o777 & ~caller_umask`，普通文件为 `0o666 & ~caller_umask`；
- 禁止用后置 `chmod` 扩大 caller umask 已移除的权限；
- `umask 002/022/077` 都必须测试；
- 任一 payload、metadata、origin marker 或 publish failure 后，不得留下 target 或 staging residue。

### 公共异常

统一语义类别为 not-found、exists、integrity、compatibility 和 I/O。现有 functional exception 名称继续兼容；实现阶段增加可区分的 generic checkpoint category，并确保父路径形状、权限、磁盘和其他 OS 错误进入 I/O 类，保留原异常为 `__cause__`，不再靠 message text 分类。

## 2. Boundary semantics：B9、B21

### Finite-Q anchoring

Q 对 director sign 是偶函数，因此 `n` 与 `-n` 都接受，并 canonicalize 为 oriented outward face normal。sign-equivalent 输入必须产生相同 canonical identity。非平行、zero 或 non-finite normal 拒绝，错误必须报告 axis、side、expected outward normal、observed normal 和 `parallel_or_antiparallel` 等价规则。

### Robin

公开支持域冻结为：

`0 <= lower_impedance, upper_impedance <= 1e14`

`>1e14`、negative 或 non-finite 输入必须在 operator/output materialization 前拒绝。支持域内要求 roots finite、positive、strictly increasing，每个预期区间恰有一个 root，normalized characteristic residual `<=1e-12`，并与 80-decimal-digit 独立参考相差 `<=5e-12`。既有支持范围必须保持逐字节不变。

## 3. Public validation、capability 与 diagnostics：B13、B15、B17

### Device 与 pressure

- 设备只接受 `auto`、`cpu`、`cuda` 和 canonical `cuda:N`；
- 拒绝 `cpu:N`、unknown、whitespace 和非字符串 token；
- validation 必须发生在 output creation、runtime construction 和 CUDA allocation 前；
- requested、resolved 和 allocated identity 分开记录；
- iterative pressure solver 的 tolerance 必须 finite 且正，iteration 必须是正的非 bool integer，fixed 不得超过 max，unknown key 拒绝。

### Capability 与异常

catalogued supported combination 必须能 compile，未登记组合必须拒绝。Periodic no-slip 明确登记为“周期几何无 boundary face，因此与 free-slip 等价”的 qualified equivalent，并记录 requested/effective policy，不能静默 fallback。

公共 compile 边界把已知 binding/lowering/option-validation 的 `SimulationBindingError`、`SimulationLoweringError`、`ValueError` 和 `TypeError` 转成带 machine rejection code 的 `PublicSimulationCompilationError`，同时保留原异常为 cause。

### Diagnostics envelope

四个 application family 都在 `metadata.json::diagnostics` 写 schema version 1 envelope，至少包含：

- `schema_version`
- `application_family`
- `runtime_path`
- `enabled`
- `count`
- `steps`
- `artifact_paths`
- `payload_schema`
- `payload_units`

现有 `diagnostics.npy`、`diagnostics.csv`、字段名、dtype 和相同运行的 payload bytes 均保留；0.2 系列继续保留 `npy`/`csv` metadata alias。新 envelope 只统一发现和解释方式，不删除 family-specific 科学量。

## 4. Scoped TF32：B11

PSSolver 只在自己拥有的 execution scope 内临时应用 TF32 policy：先无副作用验证，再保存 matmul precision、CUDA matmul TF32 和 cuDNN TF32，应用 effective policy，最后在 `finally` 中精确恢复。成功、异常和 nested scope 都遵守 LIFO 恢复；construction-only 与 invalid policy 不得改变全局状态。

CPU 行为在实现阶段验证，CUDA 和所有 runtime family 的累计验证留给 RC4.5.8 单次 H100。

## 5. 后续顺序

1. RC4.5.3：B7/B24/B27 checkpoint implementation；
2. RC4.5.4：B9/B21 boundary implementation；
3. RC4.5.5：B13/B15/B17 validation/diagnostics implementation；
4. RC4.5.6：B11 scoped TF32 implementation；
5. RC4.5.7：installed-wheel CPU 累计关闭与 Consumer impact 裁决；
6. RC4.5.8：另行授权的单次 H100 累计关闭。

当前只具备进入 RC4.5.3 的资格；实现、PSSolver-Control 修改、H100、自动 merge、default promotion 和发布均未授权。
