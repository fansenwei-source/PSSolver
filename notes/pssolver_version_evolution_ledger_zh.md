# PSSolver 版本、性能与架构演进台账

本文是一份从 Shendruk benchmark 开发期延伸到正式发布版本的可持续追加台账。
它同时回答三个问题：

1. 每个版本相对 benchmark 时期究竟改变了什么；
2. 哪些性能提升有严格的同机配对证据；
3. 哪些改动是架构能力提升，而不是速度提升。

本文的当前冻结点是 `v0.1.2`，即提交
`4fa614616e9d67c98be52c150eaf303a0c80b5c1`。后续版本应追加新章节，
不得改写既有测量值或把跨任务数据重新解释为配对结果。

## 1. 口径和限制

### 1.1 benchmark 时期的定义

本台账把提交
`d32008949a0024dbe5db9ce64ed5881665ecbb64`
（`Add auditable stationarity time windows`）视作科学 benchmark 工作阶段的冻结参照。
当时的主要目标是验证 Plane Beris--Edwards--Stokes 模型、长期轨迹、稳态、
defect core 和 Shendruk-inspired observable，而不是形成稳定的软件发布接口。

严格说，当前档案中没有一次在完全相同 H100 作业内直接比较 `d320089` 和
`v0.1.2` 的测量。因此本文不声称存在这样的严格 A/B。首轮 real-first H100
任务中测得的 `legacy` 路径是最早具有完整统一测量合同的操作基线；它已经包含
`d320089` 之后、real-first 之前的少量诊断和复用改动。本文将它称作
“benchmark-era operational baseline”，而不是 `d320089` 的精确计时。

### 1.2 三类数字

- **严格配对结果**：同一作业、同一 GPU、同一网格和配置，采用平衡顺序运行
  parent/candidate；可以用于判断某项优化的因果增益。
- **版本代表值**：来自该版本或其直接冻结父线的合格 H100 测量；可以描述该阶段
  的实际速度，但不等于跨版本严格 A/B。
- **跨阶段参考包络**：将不同 H100 资格任务的代表值首尾比较；可回答总体量级，
  但不能分解成单项因果贡献。

所有生产数值比较均以 float64、TF32 关闭为默认理解。若没有特别说明，速度是完整
Beris--Edwards--Stokes timestep 的平均耗时，而不是单个微内核耗时。

## 2. 总结

### 2.1 性能总结

| 指标 | benchmark-era operational baseline | `v0.1.2` | 跨阶段参考变化 |
| --- | ---: | ---: | ---: |
| R128 平均 timestep | 14.457492 ms | 5.750187 ms | 2.514x throughput |
| R320 平均 timestep | 204.772225 ms | 45.162243 ms | 4.534x throughput |
| R320 peak allocated | 9.791970 GiB | 5.721501 GiB | 减少 41.57% |
| R320 peak reserved | 15.328125 GiB | 8.533203 GiB | 减少 44.33% |

换算成 R320 吞吐，代表值从约 `4.883` steps/s 增加到约 `22.142`
steps/s。这个 `4.534x` 是有用的总体工程量级，但不是一项严格配对实验；每项可归因
的增益见第 4 节。

### 2.2 框架总结

benchmark 时期的代码已经能够完成 Plane 科学模拟和严谨验证，但其权威入口主要是
生产脚本，几何、模型、执行策略和 provenance 仍紧密围绕当前应用组织。到
`v0.1.2`，PSSolver 已具备：

- 明确且冻结的 Plane Beris--Edwards--Stokes 科学支持边界；
- package-owned immutable configuration、application API、CLI 和历史兼容入口；
- 模型、几何、谱规划、执行策略、运行时和 workflow 的显式契约；
- 可审计的 metadata、诊断、checkpoint/restart 和原子完成标记；
- 通用周期 FFT 快路径和独立的 bounded-axis execution seam；
- 对关键性能策略的显式 rollback selector；
- 每次默认值晋升前的 CPU、CUDA、数值、显存和生产轨迹门禁。

它仍然是谱方法求解器。所有已晋升优化都保持原有基函数、归一化、模态顺序、边界
语义、投影、dealiasing 规则和离散方程。`v0.1.2` 的 bounded-axis executor 仍是
稠密 DCT/DST 矩阵法，并不声称已经实现 pruned 或 `O(N log N)` 的 DCT/DST。

### 2.3 框架对照表

| 维度 | benchmark 冻结参照 | `v0.1.0` | `v0.1.1` | `v0.1.2` |
| --- | --- | --- | --- | --- |
| 生产组装权威 | Plane 脚本为主 | package-owned configuration、runtime、application API 和 CLI | 与 `v0.1.0` 相同 | 与 `v0.1.0` 相同 |
| 模型/几何职责 | 围绕当前应用紧密组织 | 建立 model DAG、geometry specification、Stokes capability 和 workflow 边界 | 不扩大科学支持范围 | 不扩大科学支持范围 |
| 周期轴执行 | 历史逐轴/advanced 路径 | 已有 Plane 专用优化 | contiguous view + multidimensional FFT 成为通用默认 | 延续 `v0.1.1` |
| 有界轴执行 | planner 直接持有 dense DCT/DST 行为 | dense DCT/DST 经过生产验证 | 算法不变 | 独立 device-bound execution plan；dense 仍是唯一 executor |
| 数据流 | 存在重复 materialization 和 multiplier 构造 | 已完成主要 Plane storage/pointwise/stress 优化 | generic periodic 路径进一步去拷贝 | bounded gradients、projection、stress views 和 field sync 进一步去 materialization |
| 回退能力 | 以历史脚本和显式选项为主 | `legacy_production` 成为发布 rollback oracle | `advanced`/`axiswise` 可显式选择 | bounded-axis 算法未改变；既有 rollback 继续保留 |
| 发布与审计 | benchmark provenance | 版本、wheel、API、CLI、checkpoint/restart 和 release gate | 冻结 tag + H100 manifest | 冻结 tag + H100 manifest + 可替换 bounded executor seam |
| 正式科学范围 | Plane benchmark | Plane Beris--Edwards--Stokes | 不变 | 不变 |

## 3. 版本与框架演进

### 3.1 benchmark 冻结参照：`d320089`（2026-09-04）

**主要能力**

- Plane 周期--周期--有界几何；
- free-slip 速度边界与 Q 的 Neumann 边界；
- Beris--Edwards Q 动力学和 quasistatic Stokes--Brinkman 耦合；
- complete one-constant nematic force；
- Shendruk-inspired 长时模拟、稳态、defect 和方程级验证工具。

**相对后续版本的局限**

- 生产脚本仍是主要组装权威；
- 配置、模型、几何和运行期职责尚未形成正式发布契约；
- transform 执行和数据布局仍有较多重复搬运及物理/谱空间往返；
- 尚无稳定包版本、正式 application API 或发布冻结标签。

这一阶段的价值是建立可信的科学与数值参照，而不是提供最终软件结构。

### 3.2 benchmark 后、`v0.1.0` 前的性能主线

这一阶段在保持 Plane 模型和谱离散不变的前提下，逐项优化真实 timestep：

| 默认值晋升 | 提交 | 核心变化 | 直接机制 |
| --- | --- | --- | --- |
| real-first transforms | `57cf8b7` | 有界轴先做实数矩阵运算，再进入复数 FFT | 减少昂贵的复数 GEMM 和显存 |
| spectral molecular field | `7763ba1` | 线性 `L1 laplacian(Q)` 留在谱空间 | 每步 inverse transform `42 -> 37` |
| spectral stress divergence | `b410f96` | 相容导数在谱空间先求和；布尔 projector mask | 每步 inverse transform `37 -> 32` |
| compiled pointwise kernels | `9a67155` | 四个固定形状、fullgraph 的点态核使用 TorchInductor | 融合点态代数，减少 kernel/临时量 |
| truncated projected transforms | `fe7bf86` | 只计算 projector 保留的 DCT/DST 模态 | 减少有界变换和 FFT 批次工作量 |
| Plane Hermitian half spectrum | `2e54483` / `afbc95e` | 利用实输入共轭对称性只存半谱 | 减少谱存储和后续变换工作量 |

每一种新默认都保留显式历史路径作为 rollback 或诊断对照。这里的“谱空间先求和”
和“截断已知将被 projector 丢弃的模态”是代数等价重排，不是修改方程或降低精度。

### 3.3 `v0.1.0`：第一版正式受限发布（2026-09-16）

Tag：`v0.1.0`；发布提交：`8cdc0ac`；冻结生产候选：`38cd933`。

这一版本的主要跃迁是**从一个 benchmark 应用演变为有边界的科研软件包**：

- `PlaneBerisEdwardsRunSpec` 成为不可变配置权威；
- `pssolver.applications.run_plane_beris_edwards` 成为可调用 API；
- `pssolver-plane-beris-edwards` 成为安装后的 CLI；
- 历史脚本保留为兼容入口；
- Plane production assembly 被迁入 package-owned runtime；
- 配置、几何、模型 DAG、Stokes capability、谱计划、执行策略和 workflow 拥有显式
  边界；
- checkpoint/restart、provenance 和 completion contract 成为发布能力。

`v0.1.0` 的最终 Stage S parent/candidate H100 资格是性能中性的
（aggregate candidate/parent elapsed ratio `1.000991904034`），并以三组 R320
byte-identical Q/u/p 证明架构迁移没有改变生产数值。也就是说，它的主要收益是可维护
性、可审计性和后续可扩展性，而不是在发布点再额外加速。

### 3.4 `v0.1.1`：通用周期快路径（2026-09-17）

Tag：`v0.1.1`；发布提交：`07bbf80`；候选提交：`525ba23`。

新增两个 generic default：

- 连续 transform field group 使用零拷贝 slice/view，代替 Python list advanced
  indexing；
- 两个及以上周期轴使用单次 multidimensional `fftn`/`ifftn`，代替逐轴 FFT。

历史 `advanced` 和 `axiswise` 路径保留为 rollback。通用周期 microbenchmark 在
H100 上得到约 `1.565x`（float64 512x512）和 `1.621x`（float32
1024x1024）加速，peak allocated memory 约降至 rollback 的 `77.45%`。

对当前 Plane R320 应用，这一版本是**性能中性**而不是显著加速：candidate/parent
elapsed ratio 约 `1.0124`，且 Q/u/p byte-identical。原因是 Plane 的生产路径已经
采用 Hermitian-half mixed transform；通用 periodic fast path 的主要价值在于扩展
通用谱内核和未来几何，而不是重复加速已经专门优化过的 Plane 路径。

### 3.5 `v0.1.2`：有界轴执行层和数据流（2026-09-18）

Tag：`v0.1.2`；发布提交：`4fa6146`；性能候选：`f121428`。

框架变化：

- 把 DCT/DST axis execution 从 tensor-product planning 中分离出来；
- 通过 device-bound bounded-axis execution plan 建立可替换 executor seam；
- 当前唯一生产 executor 仍是经过验证的 dense matrix 方法；
- 为未来 pruned DCT/DST、库后端或其他有界轴算法建立了不侵入上层 planner 的接口。

数据流优化：

- 按 basis、axis、device、dtype 缓存周期梯度 multiplier；
- 非 autograd 所有权路径直接写 bounded-gradient output；
- 原地更新 owned spectral projection buffer；
- 连续 stress batch 使用 natural tensor view；
- dynamic field refresh 使用统一 grouped field-access contract。

bounded-axis 抽象本身在 H100 上性能与显存中性，并产生 byte-identical Q/u/p；它的
收益是职责分离。随后的 dataflow candidate 在同一个 H100 作业内得到：

| 网格 | parent | `v0.1.2` candidate | 严格配对 speedup | 数值结果 |
| --- | ---: | ---: | ---: | --- |
| R128 | 7.120957 ms | 5.750187 ms | 1.238387x | Q/u/p byte-identical |
| R320 | 48.960003 ms | 45.162243 ms | 1.084091x | Q/u/p byte-identical |

两种网格均为三次配对全部 candidate 更快；forward/inverse transforms 仍为
`7/32` 次每步，peak reserved memory 不变，peak allocated 增加低于 `0.04%`。

## 4. 严格配对的 H100 性能增益

下表只列同一个资格作业内的 parent/candidate 对照。不同表行不应机械相乘，因为
warmup、profiler 版本和完整配置可能不同。

| 优化 | H100 Job | 网格 | control -> candidate (ms/step) | speedup | 显存和调用变化 | 数值门禁 |
| --- | ---: | --- | ---: | ---: | --- | --- |
| real-first execution | `10797581` | R128 | 14.457492 -> 13.438717 | 1.0758x | allocated ratio 0.905 | 生产差异为 float64 roundoff |
| real-first execution | `10797581` | R320 | 204.772225 -> 176.129441 | 1.1626x | allocated -10.12%，reserved -7.21% | 生产差异为 float64 roundoff |
| spectral molecular field | `10799108` | R320 | 175.854459 -> 171.130451 | 1.0276x | inverse `42 -> 37`，显存不变 | relative L2 < 3.5e-16 |
| spectral stress + Boolean mask | `10803768` | R320 | 171.175 -> 159.071 | 1.0761x | inverse `37 -> 32`；allocated +0.79% | relative L2 < 2.9e-16 |
| compiled pointwise | `10809199` | R128 | 12.266629（约） -> 10.015152（约） | 1.2249x | allocated 不变 | relative L2 < 4.5e-16 |
| compiled pointwise | `10809199` | R320 | 158.995313（约） -> 113.636758（约） | 1.3992x | allocated 不变，reserved 略降 | relative L2 < 4.5e-16 |
| truncated projected transforms | `10817259` | R320 | 113.872 -> 85.598 | 1.3303x | allocated ratio 0.9741；reserved ratio 0.9084 | Q/u/p relative L2 <= 2.34e-15 |
| Hermitian half spectrum on truncated path | `10823992` | R320 | 85.520（由 speedup 反算，约） -> 49.364779 | 1.732440x | candidate 5.720723/8.533203 GiB；calls `7/32` | max relative L2 约 4.44e-15 |
| `v0.1.2` bounded dataflow | `10835210` | R128 | 7.120957 -> 5.750187 | 1.238387x | reserved 不变 | Q/u/p byte-identical |
| `v0.1.2` bounded dataflow | `10835210` | R320 | 48.960003 -> 45.162243 | 1.084091x | reserved 不变 | Q/u/p byte-identical |

注：Hermitian-half control 的精确均值没有在冻结 promotion report 中单独抄录；表中
control 值是由 `49.364779 * 1.732440` 反算的近似值，因此只保留三位小数。正式证据
应引用 candidate 时间和 speedup，而不是该反算值。

## 5. 代表性默认路径快照

这些数字来自不同 H100 任务，适合展示演进轨迹，不适合当作严格配对：

| 阶段 | R320 mean timestep | 说明 |
| --- | ---: | --- |
| benchmark-era operational `legacy` | 204.772225 ms | 首轮统一 real-first A/B 中的历史执行顺序 |
| real-first default | 176.129441 ms | 同一首轮 A/B 的 candidate |
| spectral molecular-field default | 171.130451 ms | 五次 inverse transform 被移除 |
| spectral stress default | 159.071016 ms | inverse calls 降至 32 |
| compiled pointwise default | 约 113.64 ms | fixed-shape TorchInductor；一次性编译成本约 21.4 s |
| truncated projected transform default | 约 85.60 ms | projector 保留模态的部分有界变换 |
| Hermitian-half Plane path | 49.364779 ms | truncated + half-spectrum |
| `v0.1.1` / `v0.1.2` parent representative | 48.960003 ms | bounded-axis abstraction之后、dataflow 之前 |
| `v0.1.2` | 45.162243 ms | 当前冻结代表值 |

因此，当前最稳妥的总体表述是：**在相同 H100 型号、R320 和 float64 的代表性
记录中，PSSolver 从约 204.8 ms/step 进步到约 45.2 ms/step，吞吐提升约 4.53
倍，同时 peak allocated/reserved 的代表值分别下降约 41.6%/44.3%。这是跨任务的
工程参考包络；严格因果结论应使用第 4 节的逐项配对数据。**

## 6. 精度和科学语义是否改变

截至 `v0.1.2`，没有任何被晋升的性能改动改变以下内容：

- Beris--Edwards 或 Stokes--Brinkman 方程；
- Q convention；
- Fourier/DCT/DST 基函数与归一化；
- 模态顺序和导数 parity；
- free-slip/Neumann 边界语义；
- incompressibility projector；
- `cubic_half` dealiasing 截断规则；
- float64 生产精度、TF32 关闭策略；
- 保存数据的 public shape 和格式。

出现 roundoff-level 差异的路线改变了数学等价运算的执行顺序，例如谱空间先求和、
TorchInductor fusion 或 Hermitian-half/full-complex 之间的等价表示。需要保持逐位一致
的改动（例如 `v0.1.2` dataflow）已通过 Q/u/p byte-for-byte 门禁。不同执行顺序
产生 `1e-15` 左右差异不意味着降低了离散精度；长期混沌轨迹仍可能因微小舍入差异
逐渐分离，因此科学比较必须使用统计 observable，而不能要求无限时间的逐位轨迹一致。

## 7. 未晋升和性能中性的路线

记录失败或中性实验与记录成功同样重要：

- **FFT 模拟 DCT/DST 的早期方案**：在当时的布局和长度下没有优于 dense 路径，
  被记录并拒绝；没有改变默认值。
- **R2R-C bounded transform policy**：H100 Job `10833253` 中没有单元满足晋升
  条件；R128 `auto` 相对 dense 回退约 8.54%，R320 基本中性，因此 dense 保持
  production default。
- **static nematic algebraic fusion**：第一版 packed-output 本地回退；最小 exact
  ablation 在 R128 仅约 `0.34%`，不足以消费 H100 资格资源或进入发布。
- **bounded-axis abstraction**：R128/R320 speed ratio 分别约 `1.0021` 和
  `1.0002`，属于性能中性；它因架构价值和 byte-identical 结果进入 `v0.1.2`。
- **`v0.1.1` 对 Plane**：通用 periodic fast path 对通用周期问题明显有效，但对
  已采用专用 half-spectrum mixed transform 的 Plane R320 中性。

这些结论意味着后续不应简单重跑同一实现；若重访，必须先说明新的算法、库后端、
数据布局或硬件条件为何改变原结论。

## 8. 后续版本如何追加

每个新版本在本文末尾追加一节，并至少填写：

```text
## vX.Y.Z — YYYY-MM-DD

Tag / release commit:
Frozen parent:
Scientific scope change: yes/no（若 yes，单独说明验证）
Architecture changes:
Default-policy changes:
Rollback selectors:

CPU gate:
GPU model / Job ID:
Grid, dtype, TF32, warmup, measured steps, trial ordering:
Parent mean/median/std:
Candidate mean/median/std:
Paired speedup and faster-pair count:
Peak allocated/reserved:
Transform/kernel-call changes:
Numerical comparison (bitwise or norms):
Trajectory/restart validation:
Manifest path and SHA-256:

Classification:
Known limitations:
Rejected alternatives:
```

追加时遵守以下规则：

1. 不用本地 RTX 数字替代 H100 production qualification；
2. 不把不同 PyTorch/CUDA/GPU/配置的数字称为严格 speedup；
3. 同时报告绝对时间和倍率；
4. 同时报告 peak allocated 与 peak reserved；
5. 明确 bitwise identity 与 tolerance-equivalence 的区别；
6. 明确一次性编译成本是否计入或如何摊销；
7. 保存 Job ID、提交、控制/候选身份和 checksum manifest；
8. 对中性或失败候选保留结论，防止未来重复试错；
9. 发布 tag 不回写，新工作从新分支和新版本开始；
10. 性能优化若改变科学离散、精度、边界或模型，必须作为独立科学变更验证，不能
    只通过性能门禁。

## 9. 证据索引

仓库内的主要证据入口：

- `benchmarks/initial_performance_report.md`：real-first、spectral molecular
  field、spectral stress、compiled pointwise、truncated transform 的详细演进；
- `benchmarks/plane_half_spectrum_default_report.md`：Plane Hermitian-half H100
  晋升；
- `benchmarks/hermitian_half_spectrum_report.md` 和
  `benchmarks/truncated_half_spectrum_integration_report.md`：本地候选与组合验证；
- `benchmarks/periodic_fast_path_candidate_report.md`：`v0.1.1` 通用周期快路径；
- `notes/pssolver_v0_1_scope.md`：`v0.1.x` 科学和架构支持边界；
- `notes/pssolver_v0_1_1.md`：`v0.1.1` 发布证据；
- `notes/pssolver_v0_1_2.md`：`v0.1.2` 发布证据；
- `CHANGELOG.md`：发布级 Job ID、commit 和 manifest hash；
- `notes/bounded_axis_execution_architecture.md`：有界轴执行层设计。

关键 H100 manifest SHA-256：

| 范围 | Job | Manifest SHA-256 |
| --- | ---: | --- |
| real-first transforms | `10797581` | `4a0c96b7b67d746e8ef01d49be92d2c5529f194f5dc7973d5da83d9b5e9bd3d2` |
| spectral molecular field | `10799108` | `05427a685264ed5bd42405effb306a016347362d8753ddd764a3a3df15ac68a0` |
| compiled pointwise | `10809199` | `a910441b354872023b014ce19a0cc97e6fc8ae6b67fbc8fc91123f1ad2576059` |
| truncated transform A/B/C | `10817259` | `71d3bd5d0bab54f6a8fa6ccc5762aaaa9879442040c463b1d7e20ce3c98a1df2` |
| `v0.1.0` release qualification | `10832784` | `b04d570fbe6968c372703f7ec9a1fa4f043147ca6a80698ea29c21a6cbf9a7a5` |
| `v0.1.1` candidate | `10833968` | `15ffa359a15eb7a8fd907053b7981dbf494b0f6f31aea95b4a7dfe7f116c5a3a` |
| `v0.1.1` implicit default | `10834013` | `f9a6f14b217a3f49d8abf065756f0082b124dd74b143f1a8499f1ded2fb76c7e` |
| Plane/Channel continuation | `10834996` | `8cf0f0d35f8035290973ac405c7946385824072c6ddff821a4c2e59b2b89ea37` |
| `v0.1.2` bounded-axis abstraction | `10835044` | `6ff57b61e27985f842340fa6f3edc7cc5e9cf3944f39a49caad777a994a07efe` |
| `v0.1.2` bounded dataflow | `10835210` | `65a72a34a42a526e8ada36bd436ff1cbbaed0d2213fb88804eed1db6248b7876` |

如果仓库内摘要与归档的 raw JSON/manifest 发生冲突，应停止更新并回到对应 commit、
Job 输出和 checksum manifest 核实，不应凭记忆修正数字。
