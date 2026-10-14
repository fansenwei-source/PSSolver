# PSSolver v0.2.0rc4 发布后核验：Claude 对 Codex 草稿的盲审

- 日期：2026-10-14
- 被审草稿：
  - `notes/PSSolver_v0_2_0rc4_post_release_verification_codex_draft_zh.md`
  - `notes/PSSolver_v0_2_0rc4_post_release_verification_codex_draft.json`
- 草稿结论：`PASS_WITH_DOCUMENTED_LIMITATIONS`（本审计把它当作待验证的主张）
- 机器可读记录：`notes/PSSolver_v0_2_0rc4_post_release_verification_claude_review.json`
- 本人的独立证据：
  - 在读到 Codex 草稿之前完成的发布后验证：`/home/fansenwei/Desktop/Develop/PSSolver/notes/PSSolver_v0_2_0rc4_post_release_verification.json`（sha `24f7e1a1…`）及配套的 `_zh.md` / `_zh.pdf`；
  - 证据目录 `/home/fansenwei/pssolver_rc4_post_release_verification_20261013_v1`；
  - 累计追踪 `PSSolver/notes/PSSolver_v0_2_issue_tracking_zh.pdf`。

## 结论字段

```text
classification = FAIL
tag_identity_verified = true
artifact_identity_verified = true
historical_issue_matrix_verified = false
closed_verified_count = 28
accepted_limitation_count = 3
deferred_architecture_count = 9
doc1_confirmed = true
doc2_confirmed = true
runtime_regression_found = true
scientific_or_numerical_failure_found = false
nematics3d_modified = false
release_mutation_recommended = false
```

**判 FAIL 的原因**：B10 在矩阵里记为 `closed_verified`，却在 rc4 开发区间内回归了。另有 9 个"已关闭"条目仍有可复现的残留，所以草稿里"38 关闭 / 0 部分"不成立。

把独立结论映射回矩阵的状态，计数是：closed_verified 28、partial_residual 9、关闭后回归 1、accepted_limitation 3、deferred_architecture 9。
- partial_residual 的 9 项：B7、B11、B13、B17、B20、B23、B24、N3、N6；
- 关闭后回归的 1 项：B10。

**不判数值失败的原因**：在已认证的 fresh-run 路径上没有发现数值回归。影响科研使用的健壮性问题单独列在第 3 节。

## 1. 已独立验证的主张

1. **发布身份**：
   - annotated tag object `8822ab370e252d76549c6365dc5f5023ec3231c4` 解引用到 `658869456d762c917ca4409b091539c3733df2fb`，本地与远端一致。
   - GitHub Release 是 non-draft prerelease。
   - 两个资产的大小（872255 / 1094307）和 SHA-256 与登记值一致。
   - tag 里有 248 个 `pssolver/**/*.py`；wheel 和 sdist 的文件与 git tree 逐字节相同。
   - `156589b`、`212518e` 只改了 notes 和 tests，不是 qualified source；分支 tip 是 `212518e`。
2. **声明范围内没有新的 solver / 数值 / Hermitian / 压力伴随回归**：
   - B1：functional 跑 7500 步（t=150），production 跑 6000 步，反 Hermitian 范数都是 0；同一脚本在 rc1 上会在 t=95–100 出现 NaN。
   - B5：用独立 collocation oracle 核对，残差 ≤1.2e-14；rc3 负对照的残差是 0.32。
   - N1：稠密算子 ‖T−Aᴴ‖/‖A‖ ≤ 3.65e-16。
   - B2、B25、B29 的复现都成立。
3. **修复 commit 的祖先关系**：所有引用的修复 commit 都通过 `git merge-base --is-ancestor <c> 6588694`。
4. **公开能力边界**（见第 4 节）：只有四个已认证组合；声明层已经解耦；执行层不是笛卡尔积；草稿列出的七项能力都不在公开执行范围内。
5. **clean installed-wheel**：
   - 独立 venv 里 `pip check` 通过，import 全部来自 site-packages；
   - Periodic 两种存储和 Channel 各跑 4 步，2+2 restart 与连续运行逐字节一致；
   - PSSolver 测试集对 installed wheel 跑出 3146 passed，260 个失败全部是 harness（读源码路径或调用 git）；
   - PSSolver-Control `b8539e0` 对 rc4 wheel：310 passed / 1 skipped。
6. **H100 节点失败不是求解器回退**：本机 RTX 3060 Ti 上，`tests/test_rc4_plane_nyquist_cuda.py` 的存储等价分类已经通过，只是挂在硬编码的 `"H100"` 设备名断言上。这一点与草稿一致。
7. **R1–R9 仍然延期**：R2 已部分集中到 `io/checkpoint.py`，但仍有 10 套实现；R3 从 5,153 行增长到 7,369 行；R6 从 2,626 行增长到 2,819 行，另有 1,443 行公开不可达的 lifting 基础设施。
8. **ENV1**：确认只是本机依赖元数据冲突（见第 6 节）。

## 2. 不成立或证据不足的主张

| 草稿主张 | 判定 | 依据 |
|---|---|---|
| 没有新的 checkpoint 回归 | **不成立** | 已安装的 rc4 wheel 读 rc1/rc2/rc3 正式版本写出的真实 functional checkpoint：Periodic 和 Channel 的直接 import、显式 `migrate_legacy_checkpoint` 全部被拒（6/6）。同一脚本在 `1b15bf8` 上能导入 rc3 Periodic（`qualified_periodic_functional_bridge_v2_reader`），迁移后状态逐位相等，rc3 Channel 也能迁移。回归 commit 是 `dc8c87a`（Channel）和 `a195bd0`（Periodic），与 rc424 记录的承诺相反 |
| 矩阵 38 / 0 / 3 / 9 | **不成立** | 草稿只核对了"有关闭记录、commit 是祖先"，没有用真实发布产物重新测试关闭语义。独立结果：28 / 9 / 1 / 3 / 9 |
| 没有 TF32 回退 | 相对 rc3 成立，但 B11 不能算关闭 | `configuration/execution_device.py:86-93` 先恢复精度、后设 `allow_tf32`，调用方的 `'medium'` 会被恢复成 `'high'`。`tests/test_rc456_tf32_execution_scope.py:32-35` 和 `benchmarks/qualify_rc458_tf32_scope.py:34-37` 都没有构造出真实的 `'medium'` 状态，所以测试和 H100 都没发现 |
| 关闭分组 | 37/38 一致 | A3 在矩阵里的 target 是 `rc3`（`closed_breaking_migration`），不是 rc2 direct fix |
| B8/B20/N4 靠 fail-closed 限制关闭 | B8、N4 成立；B20 不对 | B8 的物理问题未修：`models/active_nematics/stokes.py` 自 v0.2.0rc2 起没改过，壁面 H 误差停在约 1.85e-5；内部 lifted 路径仍可执行。B20 不是靠限制关闭的，它是显式的 fresh-conditioning 策略，而 rc432 合同要求的 strict 拒绝没有实现（`operators/lifting.py:397-398`） |
| B30 / A2 / A5 为接受限制 | 行为成立 | 但 B30 的处置是"补文档"，一直没做：`operators/robin.py:306` 的 docstring 没有说明，p852/p853/p855 测试仍把它当门禁 |
| 95 passed / 1 skipped 足以支撑结论 | **证据不足** | 见第 5 节 |

## 3. 新发现的问题（草稿未覆盖）

| ID | 严重度 | 问题 | 证据 |
|---|---|---|---|
| B10 回归 | 中 | functional checkpoint 无法跨版本读取（见上表） | 主会话在 installed wheel 上复现，并在 `1b15bf8` 上做了对照 |
| NEW-1 | 中 | Periodic 不开 checkpoint 时，发散后 Q/u/p 全为非有限值，仍返回 `SimulationRunStatus.COMPLETE` 并写下 COMPLETE 标记 | `workflows/periodic_beris_edwards.py:97-230` 没有有限性检查；installed wheel 复现 |
| N3 残留 | 中 | `legacy_channel`（Channel 的生产默认 runtime）的 identity 在 rc1 和 rc4 下相同（`e6b4806f…`），但偶数网格动力学已经不同；rc4 会静默续跑 rc1 偶数网格 checkpoint（Q_4 的 sha：rc1 `a7631b8c…`，rc4 `568d5c7b…`，混合续跑 `ff0f71fd…`） | installed wheel 复现；rc423 记录 `p7_channel_checkpoint_modified=false` |
| NEW-2 | 中 | wheel 往 site-packages 顶层安装了 88 个模块的 `benchmarks` 包 | `setup.py:24` 的 `find_packages()` 没有 include 过滤 |
| NEW-3 | 低–中 | Plane 的实现溯源对 `parents[2]/Plane_beris_edwards_stokes.py` 计算哈希；重新打包时漏掉这个 shim，Plane 就会失败 | `applications/plane_beris_edwards.py:111,334-342`，`setup.py:25` |
| B11 残留 | 低 | 调用方的 TF32 `'medium'` 被恢复成 `'high'` | installed wheel 复现 |
| NEW-4 | 低 | README 的 "Historical Plane application" 示例照抄会抛 ValueError（save_start_step 默认 5000，steps 只有 10） | installed wheel 复现 |
| NEW-5 | 低 | CHANGELOG 的 0.2.0rc4 段保留准备期说法（"不打 tag、H100 smoke 仍待做"），并写"Restrict public lifting capabilities to qualified static Dirichlet data"，实际没有任何 lifting 策略可执行 | CHANGELOG |
| NEW-6 | 低 | tag commit 不在 `origin/main` 上，但 GitHub Release 的 `targetCommitish` 是 `main` | git 与 release 元数据 |
| NEW-7–15 | 低/信息 | Robin 溢出抛普通 ValueError；压力 tolerance 用递推残差判定达标；首次压力求解失败时异常类型不对；rc1 奇数 Nx 的 Channel BE checkpoint 被当作 exact 升级；legacy Channel 写 checkpoint 扰动正在跑的 run；`run_simulation` 信任任意 `CompiledSimulation`；normalization 不写入输出目录；functional 拒绝 `auto`；diff-check 遗留 | 详见我的验证记录 |

## 4. 公开能力边界（逐问回答）

1. **是否只有四个已认证组合**：是。installed catalog 的 `qualified_combinations` 是 Plane complete-stress（legacy_production、compiled_v2）、legacy active-force Channel（legacy_channel、compiled_channel_v2）、Periodic complete-stress（periodic_spectral）、Channel complete-stress（channel_complete_stress）。
2. **声明层是否解耦**：是。geometry 类不含模型逻辑，模型构造器不接收 geometry，边界策略是通用的。残留耦合有三处：模型声明带有特定几何风味的初值族；legacy 模型写死了 `channel_stokes`；各族的 execution-option schema 不同。
3. **执行层是否仍不是笛卡尔积**：是。通过 registry 加 if-chain 显式分派（`configuration/public_simulation_runner.py:774-800`）。
4. **未注册组合是否在建目录或执行 timestep 前稳定拒绝**：编译期合同违规是。6/6 探测都以 `PublicSimulationCompilationError` 拒绝，且不建目录；40 次 lifting 尝试都以 `unsupported_lifted_wall_molecular_field` 拒绝，0 个张量、0 个文件。**例外**：在没有 GPU 的主机上请求 `cuda`，编译会通过，运行时才失败，而 Periodic 和 Channel BE 会留下空的输出目录（B13 残留）。另外，PeriodicBox 配 `no_slip_velocity` 会被接受，这是有记录的等价设计，元数据记了 requested、effective 和原因。
5. **以下能力是否都不在公开执行范围内**：都不在。
   - 任意 model–geometry 组合；
   - 公开的非齐次 Neumann；
   - 一般公开 Robin（catalog 中 `robin` 的 `executable=false`）；
   - 动态或可训练的边界数据；
   - functional batch > 1（runtime kind 只有 `*_batch_one`）；
   - Plane optimal control；
   - 生产规模的优化器 campaign。

## 5. 测试证据复核

草稿那 95 个定向测试 + 1 步 smoke，覆盖的是"已登记的修复在当前代码里仍然成立"，**不足以支撑"没有 checkpoint 回归"和"38 项关闭"**。缺少的内容如下：

- 用 rc1/rc2/rc3 正式 wheel 写出的真实 checkpoint 做跨版本读取。仓库里的测试是用当前 identity 构造"旧版" bridge（`tests/test_rc4_functional_checkpoint_identity.py:194-221`），所以 B10 回归被掩盖了；
- legacy Channel 的跨版本 restart（N3）；
- TF32 调用方状态为 `'medium'` 的情形（B11）；
- 不开 checkpoint 时发散 run 的结束状态（NEW-1）；
- wheel 的顶层内容检查（NEW-2、NEW-3）；
- PSSolver-Control 对已发布 wheel 的兼容性运行；
- 全量源码测试，以及 PSSolver 测试集对 installed wheel 的运行。

正式的 H100 发布证据只做了审计、没有重跑：
- Job 10881426（`156589b` 的记录）是 31 秒的 installed-wheel smoke：7 个只在 CUDA 上跑的测试，加 Plane 走 1 步。
- 累计 TF32 闭合（rc4.5.8）适用于发布包，因为 `pssolver/` 相对 `a195bd0` 只差 `_version.py`。
- CPU↔CUDA checkpoint 矩阵（Job 10874642）早于 RC4.3–4.5，没有在 6588694 上重跑。

## 6. nematics3d 边界

- **现象**：conda 环境 Nematics3D 里，`pip check` 报 `nematics3d 0.9.0b1 has requirement numpy==2.3.2, but you have numpy 2.4.3`。这只是本机的依赖元数据冲突。
- **与发布无关**：PSSolver wheel 的 `Requires-Dist` 只有 `numpy>=1.22`、`scipy>=1.8`、`torch>=2.5`、`tqdm>=4.64`；nematics3d adapter 是可选的。我建的 venv 不装 nematics3d，`pip check` 通过。
- **没有证据**表明这是 v0.2.0rc4 的发布依赖失败。
- 本轮没有修改 nematics3d。

## 7. 必须修正的文档 / 是否改源码 / 下一步

### DOC1：成立，错误在文档

- **矛盾所在**：`notes/pssolver_v0_2_scope.md:33-34` 写着 Plane complete-stress "additionally qualifies static prescribed Q data and strong homeotropic/planar Q policies through the static lifting route"。这一段从 rc1 准备提交 `1c52371` 起就没改过。而 installed catalog 里这三个策略都是 `declarable=true, executable=false, qualified_applications=[]`。
- **处理方式**：只改文档。
- **绝对不能借修文档重新开放 lifting runtime**：壁面分子场的缺陷还在，strict fresh-conditioning 合同也没实现。
- 同类问题还出现在两处：CHANGELOG 的 lifting 那句话；README 的限制清单没写明 lifting 不可执行。

### DOC2：成立，属于文档和易用性缺口

这不是 runtime bug，也不是 API bug：fail-closed 正是设计中的合同。

我独立复现时只传了 `tf32`，报错列出了全部缺失项：`missing=('disable_q_gradient_reuse', 'molecular_field_linear_space', 'pointwise_execution', 'stress_divergence_sum_space')`。

建议：
- README 增加完整、可运行的示例：至少一个组合，并指向 `capability_catalog()`。最好由 catalog 生成或用测试校验，以免和代码漂移。
- 明确写出"只有 catalog 里的组合可以执行"。
- 按 runtime path 列出精确的 option 全集。
- 写清 `InvocationSpec` 从哪里导入。
- 示例里只用已认证组合，不要给出任何会让人以为别的组合也可用的变体。
- 顺带修掉现有 Plane 示例的 ValueError（NEW-4）。

### 其他需要修正的文档

- `operators/robin.py:306` 的 docstring（B30）；
- known-issues matrix：登记 B10 回归和 9 个部分残留。矩阵需要扩展出一个"关闭后回归"状态，用 evidence-only commit 提交，并同步更新测试和 successor 登记。

### 源码

需要改，但不在本轮做。rc5 必须做的：
- B10：reader 重新设计，按版本冻结 identity，用真实发布 wheel 生成 fixture；
- NEW-1：有限性门禁；
- N3：legacy Channel 加动力学版本，或拒绝 rc1 偶数网格 checkpoint；
- NEW-2、NEW-3：打包修正；
- B11：调整恢复顺序，并修正测试和 H100 harness 的 helper。

其余排在后面：B17、NEW-7、B13、B23、B24、B7、N6、B20、NEW-8 到 NEW-11。

### 下一步建议

1. Codex 依据本审计重新裁定草稿：由 `PASS_WITH_DOCUMENTED_LIMITATIONS` 改为 `FAIL`（B10 回归）。
2. 不移动 tag，不改 GitHub Release，不发 PyPI（`release_mutation_recommended = false`）。
3. 在 `fix/v0.2.0rc4-audit` 上做一个 evidence-only commit：登记回归和残留，更新矩阵，不碰 runtime。DOC1、DOC2 可以放进同一个文档提交，前提是不开放任何 lifting runtime；否则留到 rc5。
4. 按上面的源码项规划 rc5；rc5 冻结后做一次授权的单卡 H100 累计运行，覆盖 CPU↔CUDA checkpoint 矩阵、TF32 `'medium'` 调用方状态和标准门禁。

## 约束遵守

- 没有修改 PSSolver 源码、PSSolver-Control、nematics3d；
- 没有改动 tag 或 Release；
- 没有 commit、merge、push；
- 没有提交 H100；
- 没有放宽测试、阈值或能力合同。

本文件和配套 JSON 都是 `PSSolver_v0_2_0_rc4_audit/notes/` 下的未提交文件。按要求就此停止，等待 Codex 或用户决定是否实施 DOC1、DOC2 的修正。
