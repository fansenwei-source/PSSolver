# PSSolver v0.2.0rc4 发布后独立核验（Codex 草稿）

## 结论

当前草稿分类为 `PASS_WITH_DOCUMENTED_LIMITATIONS`。

在 `v0.2.0rc4` 声明的四个 model–geometry 组合和既定 functional API 1.0
范围内，没有发现新的求解器、数值、checkpoint、Hermitian、压力伴随或 TF32
回退。Claude 的历史矩阵中 38 项 correctness/API 问题都有关闭记录，相关早期修复提交
均属于发布 tag 的祖先；定向源码回归和已发布 wheel 的独立 CPU smoke 也通过。

这不等于“历史审计所有问题清零”。三项限制仍被明确接受，R1–R9 九项结构去重仍推迟到
0.2 之后；RC4.3 对不收敛的 lifted wall molecular-field 路径采取的是公开 fail-closed
限制，而不是重新发明并资格化一套正确的数学表示。

本文件是 Codex 的独立草稿，尚待 Claude 盲审。它没有修改源码、tag、Release、
PSSolver-Control 或 `nematics3d`，也没有重新提交 H100 Job。

## 1. 发布身份

- annotated tag：`v0.2.0rc4`
- tag object：`8822ab370e252d76549c6365dc5f5023ec3231c4`
- 解引用后的 qualified source：`658869456d762c917ca4409b091539c3733df2fb`
- GitHub prerelease：
  `https://github.com/fansenwei-source/PSSolver/releases/tag/v0.2.0rc4`
- Release：非 draft、prerelease

GitHub 下载后的两个资产与冻结记录一致：

| 资产 | 大小 | SHA-256 |
| --- | ---: | --- |
| `pssolver-0.2.0rc4.tar.gz` | 872255 | `c7f5f2be2881cd3377cd936f8596ce2faf1b2a35fd96a955e71d92b3a292964c` |
| `pssolver-0.2.0rc4-py3-none-any.whl` | 1094307 | `1bf31f324180dac58777a23785fb4e1424d3cc589099cf8c6328e2f6c74d8cc5` |

tag、sdist 和 wheel 中的 248 个 `pssolver/**/*.py` 文件逐文件 SHA-256 一致。

## 2. clean installed-wheel CPU 核验

建立了不继承 system site-packages 的临时 Python 3.12 venv，并安装 CPU PyTorch 与
wheel 声明的依赖：

- Python 3.12.13；
- PyTorch 2.6.0+cpu；
- NumPy 2.5.4；
- SciPy 1.18.1；
- pytest 9.1.1；
- `pip check`：PASS；
- `pssolver` 版本：0.2.0rc4；
- import 来自临时 venv 的 `site-packages`；
- source shadow import：false。

使用公开 API 构造 complete-stress Beris–Edwards + `PeriodicBox`，完成一个 CPU
timestep：

- requested/effective runtime 均为 `periodic_spectral`；
- `fallback_used = false`；
- `completed_steps = 1`；
- `COMPLETE` 存在；
- Q、u、p 都是 float64、形状正确且全部有限。

同一个算例在本机 CUDA-enabled overlay 环境和 clean CPU 环境中生成的 Q/u/p 文件
SHA-256 完全相同。

## 3. 定向源码回归

测试源码来自 `git archive v0.2.0rc4`，不是当前证据分支 tip。clean CPU 环境结果：

```text
95 passed, 1 skipped in 137.82s
```

唯一 skip 是精确绑定 H100 设备名的 CUDA qualification node。覆盖范围包括：

- Periodic 长 horizon Hermitian 修复；
- Plane Nyquist storage-equivalence support；
- Channel pressure transpose；
- Periodic/Channel checkpoint identity；
- lifting capability fail-closed restriction；
- scoped TF32 execution；
- public validation/observability；
- rc3 known-issues matrix。

补充在本机 RTX 3060 Ti 上运行 Plane Nyquist diagnostic，五个 case 全部有限，diagnostic
分类为 `PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE`；最大 velocity relative L2
为 `4.1324e-16`，最大 pressure relative L2 为 `3.4611e-16`。对应 pytest node 仅因
设备名不是 H100 而失败，不能解释为求解器回退。

## 4. 历史问题矩阵

原矩阵共 50 项：

| 分类 | 数量 | 当前判断 |
| --- | ---: | --- |
| `closed_verified` | 38 | 保持关闭；34 个早期修复提交均为 tag 祖先 |
| `partial_residual` | 0 | 无 |
| `accepted_limitation` | 3 | 仍存在且已记录 |
| `deferred_architecture` | 9 | 仍未实施 |

关闭分组如下：

| 阶段 | 问题 |
| --- | --- |
| rc2 direct fixes | B1–B4、B6、B12、B14、B16、B18、B19、B23、B26、A1、A3、A4 |
| rc3 | N1、N2 |
| RC4.1 | B5 Plane Nyquist |
| RC4.2 | B10、B22、N3、N5、N6 checkpoint/dynamics identity |
| RC4.3 | B8、B20、N4 lifting，按公开 fail-closed 合同关闭 |
| RC4.4 | B25、B28、B29 pressure contract |
| RC4.5 | B7、B9、B11、B13、B15、B17、B21、B24、B27 |

仍接受的限制：

- B30：Robin `boundary_residual` 不是独立 correctness gate；
- A2：SBDF2 可以声明，但没有注册的组合在编译期拒绝；
- A5：Shendruk activity 转换保留 1 ULP round-trip 差异。

仍推迟的架构工作为 R1–R9：组合注册、checkpoint I/O、functional facade、step
primitives、PCG、实验 Robin/finite-Q 隔离、canonical helper、qualification
infrastructure 和 import-boundary ratchet。

## 5. 公共能力的真实边界

声明层的 model、geometry 和 field boundary ownership 已分离，但执行层不是任意笛卡尔积。
live capability catalog 只资格化四个组合。一次未注册的 model–geometry 组合在输出目录创建
前稳定抛出 `PublicSimulationCompilationError`，不存在 runtime fallback。

以下能力仍不属于公开执行范围：

- 任意 model–geometry 组合；
- public nonhomogeneous Neumann；
- general public Robin；
- dynamic/trainable boundary data；
- functional batch > 1；
- Plane optimal control 和 production optimizer campaign。

## 6. 新发现的非数值问题

### DOC1：scope 文档与 live catalog 矛盾

`notes/pssolver_v0_2_scope.md` 声称 Plane complete-stress 已资格化 static prescribed-Q
和 strong homeotropic/planar Q。实际 live catalog、RC4.3.3 记录和测试都明确把
`prescribed_q`、`strong_homeotropic_q`、`strong_planar_q` 标为 declarable but
non-executable，并在 runtime construction 前拒绝。

运行时行为是安全的；错误在文档。建议下一次 evidence-only 或 rc5 文档提交修正 scope，
不得借修文档重新开放未经资格验证的 lifting runtime。

### DOC2：缺少完整 public `Simulation` 示例

README 说明了 capability catalog 和历史 Plane API，但没有给出一份完整可运行的新
`Simulation` 示例，也没有列出 Periodic execution options 的精确全集。独立 smoke 首次
省略 `disable_q_gradient_reuse=False` 时，编译器正确 fail-closed：

```text
periodic execution options must declare every qualified option explicitly;
missing=('disable_q_gradient_reuse',), extra=()
```

这不是 runtime bug，但会增加新用户正确使用公开 API 的难度。建议补充一份从 imports 到
`run_simulation()` 的完整示例。

### ENV1：本机 `nematics3d`–NumPy metadata 冲突

本机现有 Nematics3D 环境报告：`nematics3d 0.9.0b1` 要求 `numpy==2.3.2`，实际为
NumPy 2.4.3。该环境没有被用作 clean-wheel 资格证据；本轮没有修改 `nematics3d`。
完全隔离、不含 `nematics3d` 的 venv 中 `pip check` 通过，因此这不是 PSSolver
release dependency failure。

## 7. 暂定使用建议

可以继续使用 v0.2.0rc4 运行 capability catalog 明确资格化的四个组合，并遵守各 runtime
的身份、checkpoint、boundary 和 execution 合同。不要把 rc4 解释为任意几何、任意模型、
任意边界或优化控制均已资格化。

当前不建议移动 tag、改写 GitHub Release 或上传 PyPI。先让 Claude 对本草稿做独立复核，
尤其检查 DOC1、DOC2、B8/N4 的 fail-closed 解释，以及 R1–R9 是否存在遗漏。
