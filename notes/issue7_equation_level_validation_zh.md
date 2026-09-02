# PSSolver 问题 7：独立方程级验证报告

日期：2026-09-01
基线：`develop` 分支，修改前 `HEAD=53e7f1a`
验证对象：`Plane_beris_edwards_stokes.py` 使用的完整 Beris--Edwards--Stokes 生产路径

## 1. 状态结论

问题 7 的独立方程级 CPU 验证已经完成。对于正式 benchmark 使用的

- `float64 / complex128`；
- `cubic_half` dealiasing；
- (x,y) 周期、(z) 方向 Q/切向速度/压力为 Neumann-DCT；
- (z) 方向法向速度为 Dirichlet-DST；
- `zero_mean` 或正摩擦 `friction` 两种切向零模策略；

当前测试已经独立覆盖完整 (partial_jPi_{ij})、DCT/DST 奇偶求导、Schur 压力解、
压力 gauge、非零平均切向力、墙面功率及半离散自由能--功率预算。上述正式路径可以将
问题 7 标记为 **complete**。

这个结论不是“严格 Shendruk benchmark 已复现”，也不替代长时间、多 seed 或 activity
scan。它证明的是：当前 quasistatic Beris--Edwards--Stokes 方程在 PSSolver 中的离散实现
通过了独立、低模、可解析的方程级门禁。

`dealias=none` 不在正式通过范围内。它保留 cell-centered DST 的 terminal mode；该模的
连续导数需要不存在的 DCT (m=N_z) 模，因此在当前离散导数中落入 nullspace。正式结果
继续使用 `cubic_half`，它会显式删除该模。

## 2. 验证的方程

Q 方程为

\[
(\partial_t+\mathbf u\cdot\nabla)Q-\mathcal S(W,Q)=H/\gamma,
\]

其中

\[
H=-AQ-B\left(Q^2-\frac{I}{3}\operatorname{tr}Q^2\right)
  -C\operatorname{tr}(Q^2)Q+L_1\nabla^2Q.
\]

准静态流动方程为

\[
0=-\nabla p+\eta\nabla^2\mathbf u-\mathrm{fric}\,\mathbf u
  +\partial_j\Pi_{ij},
\qquad \nabla\cdot\mathbf u=0.
\]

忽略可吸收到压力中的各向同性应力后，测试中的完整 nematic stress 为

\[
\Pi=2\lambda M(Q:H)-\lambda(HM+MH)+QH-HQ
     -L_1\partial_iQ:\partial_jQ+\beta\alpha Q,
\qquad M=Q+I/3.
\]

准静态模型没有流体动能项。正确的全局功率预算是

\[
\frac{dF}{dt}
=P_{\rm wall}+P_{\rm active}+P_{\rm constraint}
-D_Q-D_\eta-D_{\rm fric},
\]

\[
D_Q=\frac{1}{\gamma}\int H_{\rm resolved}:H_{\rm resolved}\,dV,
\quad
D_\eta=\int 2\eta E:E\,dV,
\quad
D_{\rm fric}=\int \mathrm{fric}\,|\mathbf u|^2\,dV.
\]

主动功率为

\[
P_{\rm active}=-\int \Pi^{\rm active}:\nabla\mathbf u\,dV
=-\beta\alpha\int Q:E\,dV.
\]

当 (\beta=-1)、(alpha=\zeta) 时，
(P_{\rm active}=\zeta\int Q:E\,dV)。当前边界条件只是 kinematic free-slip，
不是 total-traction-free，因此 (P_{\rm wall}) 一般不能省略。

## 3. 为了使验证真正独立而做的架构调整

此前 projector 和完整 mixed-basis Stokes 类定义在可执行脚本中；导入该脚本会立即解析
命令行并开始模拟，单元测试无法安全调用正式实现。现在：

1. `BasisAwareSpectralProjector`、row-wise stress divergence 和
   `FreeSlipModalStokesSolver` 位于可导入的 `pssolver/transforms.py`；
2. 完整生产 adapter 位于
   `pssolver/models/active_nematics/stokes.py::BerisEdwardsFreeSlipStokes`；
3. `Plane_beris_edwards_stokes.py` 直接导入并使用该 adapter，不再保存测试之外的副本；
4. 新 adapter 已加入正式 run 和 validation runner 的 implementation SHA-256 provenance；
5. 保留了原实现对大型 GPU run 很重要的中间 tensor 及时释放，避免提高 R512 峰值显存。

因此，manufactured tests 验证的就是正式脚本所执行的共享实现，而不是测试目录中的复制品。

## 4. 独立验证矩阵

### 4.1 完整应力散度

`tests/test_beris_edwards_force_manufactured.py` 共 5 项：

1. 构造九个不同、非对称的 row-major 低模应力，逐行验证
   (f_i=\partial_j\Pi_{ij})，并证明转置散度 oracle 与正确答案显著不同；
2. 分别制造 even DCT 与 odd DST distortion stress，覆盖九个分量和三个方向导数；
3. 使用真实 active stress helper 和
   (Q_{xz}=A\cos(\pi z/L_z))，验证解析切向力及其非零离散平均；
4. 使用真实 `SpectralSolver`、`Fields` 和生产
   `BerisEdwardsFreeSlipStokes.compute_nematic_force` 重复同一检验；
5. 使用两组不对易的 3×3 对称无迹矩阵构造 (Q,H)，以独立 full-matrix +
   autograd oracle 分别验证 reactive+active、distortion 和 total force。

解析/自动微分比较采用 CPU float64，主要阈值为 (2\times10^{-11}) 或更严。

### 4.2 DCT/DST 与 Schur 压力解

`tests/test_free_slip_stokes_manufactured.py` 共 9 项：

- Neumann cosine (\rightarrow) Dirichlet sine 及其反向求导；
- 最低模和最高可配对模的符号、波数及 boundary-condition swap；
- (D_{N\to D}=-D_{D\to N}^{T}) 的离散反伴随关系；
- terminal DST mode 的 derivative-null 行为，以及
  `cubic_half`/`two_thirds` 删除、`none` 保留该模；
- 解析 divergence-free (mathbf u) 和已知 (p) 的 manufactured force，恢复
  (mathbf u) 与零平均有效压力；
- pressure gauge、不可压缩性、动量残差和 Schur residual；
- 独立 (4\times4) dense KKT 单模 oracle；
- 常数切向力下 `zero_mean` 与 `friction` 的不同响应。

测试明确区分：压力常数 gauge 和切向 plug-flow 零模是两件不同的事。

### 4.3 非零平均切向力

对于

\[
Q_{xz}=A\cos(\pi z/L_z),\qquad
\Pi^{\rm act}_{xz}=\beta\alpha Q_{xz},
\]

有

\[
f_x=-\beta\alpha A\frac{\pi}{L_z}\sin(\pi z/L_z),
\]

其 cell-centered 离散平均非零。测试证明：

- `zero_mean`, `fric=0`：uniform tangential force 被 Moore--Penrose inverse 删除，
  (langle u_x\rangle=0)；
- `friction`, `fric>0`：uniform mode 被保留并满足
  (mathrm{fric}\langle u_x\rangle=\langle f_x\rangle)；
- 压力零模始终为零，压力不能吸收 uniform tangential force。

这验证了先前注释中的建模含义：删除平均切向模是明确的 reference-frame/model choice，
不是 pressure gauge。

### 4.4 自由能、耗散与墙面功率

`tests/test_beris_edwards_energy_budget.py` 共 6 项：

1. 直接对 Landau--de Gennes 自由能做 autograd 方向导数，验证
   (delta F=-\int H:\delta Q\,dV)；
2. 使用非恒等 `cubic_half` 投影，验证 resolved passive relaxation
   (dF/dt=-\int H_{\rm resolved}:H_{\rm resolved}/\gamma\,dV)；
3. manufactured Stokes 解验证 pressure work 为零，且
   force power (=D_\eta+D_{\rm fric})；
4. 以独立组装的 DCT/DST 导数矩阵验证离散边界双线性型

   \[
   B_h=\langle u,D_h\Pi\rangle_h+\langle\Pi,D_hu\rangle_h;
   \]

5. 验证 (B_h) 向解析 wall traction power 二阶收敛；
6. 在一个 (z)-extruded、(Q_{xz}=Q_{yz}=0)、严格零墙功的被动构型中，
   把生产 complete force、Schur flow、Q advection/alignment/relaxation 合在同一个测试里，
   验证

   \[
   \dot F+D_Q+D_\eta+D_{\rm fric}=0.
   \]

同时另行验证 (\beta=-1) 时主动注入功率的正负号，以及 `zero_mean` uniform
constraint reaction 的功率为零。

普通 cell-center midpoint 体积分与端点 wall traction 不应在有限 (N_z) 下被错误地
要求机器精度相等。当前 refinement 的绝对误差为

| (N_z) | wall-power error |
|---:|---:|
| 16 | (1.492795083951\times10^{-1}) |
| 32 | (3.715399410078\times10^{-2}) |
| 64 | (9.278212755412\times10^{-3}) |

相邻观测阶数为 2.00643 和 2.00160，符合 midpoint 离散的二阶收敛；
solver-consistent 离散边界双线性型本身则达到机器精度闭合。

## 5. 测试发现并修复的真实问题

第一次 float64 manufactured Stokes 测试没有通过。Schur residual 已达到约
(10^{-16})，但恢复的 (mathbf u,p) 只有约 (2\times10^{-8}) 的相对精度。

原因是 periodic wavenumber 使用 `torch.fft.fftfreq` 时没有在创建时指定 dtype：数组先按
默认 float32 生成，再转换成 float64。转换不能恢复已经丢失的波数精度。

修复后，periodic mode 和 projector mode number 都在创建时显式使用
`dtype=self.real_dtype` 和目标 device。测试阈值没有放宽；同一 manufactured case 随即
达到 (5\times10^{-12}) 以内的 (mathbf u,p) 恢复精度。

这说明问题 7 的测试产生了实际价值：它不仅补充覆盖率，还发现了原有内部自洽测试无法
暴露的 float64 精度路径漏洞。

## 6. 最终验证结果

- 独立 Issue 7 tests：20 passed；
- 直接受影响的 Beris--Edwards、CLI、validation-plan 回归：全部通过；
- 完整 PSSolver CPU suite：311 passed，8 subtests passed；
- `git diff --check`：通过；
- 正式 `Plane_beris_edwards_stokes.py`：小网格、CPU、float64、`cubic_half`、1-step
  端到端 smoke 通过，生成 `metadata.json`、`Q_1.npy` 和 `COMPLETE`。

## 7. 保留限制与使用建议

1. 正式 benchmark 继续使用 `cubic_half`；不要把 `dealias=none` 当作验证通过的生产设置。
2. kinematic free-slip 不等于 total-traction-free。含 wall traction 的构型必须把
   (P_{\rm wall}) 纳入能量解释，不能笼统声称 passive (F) 单调下降。
3. `zero_mean` 是无摩擦 quasistatic 模型的参考系/约束选择；若研究平均流，应使用正摩擦、
   指定平均流量、平均压降或显式 mean-momentum dynamics。
4. 本报告验证的是半离散方程级恒等式和 Stokes solve；它不声称 IMEX Euler 对任意大时间步
   无条件能量稳定。
5. 惯性项、strict Shendruk momentum model、长期统计、初始化敏感性和 activity scan 仍是
   另外的物理复现任务，不属于问题 7 的软件方程门禁。

## 8. CPU 方程级结论

问题 7 对正式 `cubic_half` Beris--Edwards--Stokes 路径可以关闭。今后若修改以下任一部分：

- compact Q convention 或 full-matrix stress algebra；
- DCT/DST mode mapping；
- pressure Schur complement；
- tangential zero-mode policy；
- stress projection/dealias staging；
- free-energy coefficients或墙面条件；

都应把这 20 项独立测试作为强制 CPU regression gate，而不能只依赖长模拟“看起来正常”。


## 9. HPCC R512 GPU integration preflight

2026-09-01 至 2026-09-02，HPCC 在 detached worktree
`ff084fb8a861d384e6cc2e6aca6ef68c51eaa098` 上完成了问题 7 的生产集成预检。
这一附录记录部署证据，不改写前述 CPU 方程级结论。

### 9.1 CPU 门禁

- 完整 HPCC CPU suite：310 passed，1 skipped，8 subtests passed in 74.21 s；
- 唯一 skip 是既有可选依赖 `nematics3d` 缺失；
- 无其他 failure 或 skip。

### 9.2 唯一一次 R512 H100 模型执行

实际 run 为 `A18_R512_dt0p005_cubic_half_seed24_ec03fd61d53c`，配置为：

- shape = `(512, 512, 128)`，domain = `(100, 100, 20)`；
- A18，seed 24，Δt = 0.005，1 step；
- float64 / complex128，TF32 off；
- `cubic_half`，spectral refresh disabled；
- `zero_mean`；
- NVIDIA H100 PCIe。

模型和 validation runner 本身完成，runner return code = 0，且其
`_validate_completed_run` 验收通过。`COMPLETE`、`metadata.json`、`Q_1.npy`、
`u_1.npy`、`p_1.npy`、`diagnostics.npy` 和 `diagnostics.csv` 均存在。Q、u、p
全部为 float64，shape 正确，且逐块检查无 NaN/Inf。13 个 implementation
provenance 文件的 SHA-256 与目标 worktree 一致。

最终 step 1 的主要诊断为：

- `div_max = 3.3631443170731995e-18`；
- `div_rms = 6.215103487946210e-19`；
- `div_rel = 2.304254122349574e-16`；
- `schur_abs_residual = 1.0729904031905977e-12`；
- `schur_rel_residual = 1.580187876179205e-16`；
- `wall_normal_momentum_max = 2.117582368135751e-22`；
- `wall_normal_momentum_rms = 2.9883367192027164e-23`。

相对 divergence 和 Schur 残差都远低于 `1e-10` 验收门限。这里的
`wall_normal_momentum` 不是墙面功率，不能替代第 4.4 节的 CPU 能量预算验证。

峰值 GPU 已用显存为 64,405 MiB（约 62.90 GiB），约为 PyTorch 可见显存的
79.5%，余量约 16--17 GiB。模型 metadata elapsed time 为 28.5301 s，Slurm elapsed
为 68 s；这个单步数字不应直接线性外推为长模拟总时间。

### 9.3 Slurm 状态的基础设施 caveat

Slurm job 10767815 的官方 accounting 为 `FAILED / ExitCode 1:0`，因此预定的
`COMPLETED/0` 外层门禁没有通过，不能把原作业追溯性地称为 Slurm PASS。

失败发生在模型和 runner 成功结束之后：HPCC 控制目录的附加 post-validator 将
`diagnostics.csv` 中的 `1.000000000000000000e+00` 直接传给 `int()`，从而触发
`ValueError`。该解析器不属于 PSSolver 仓库中的模型或 runner，也未改变已经完成的模拟
输出。修正后的只读 postcheck 已对同一组文件完全通过，且没有重跑模型。因此，这是模型
结束后的 workflow false negative，不是 CUDA、PDE、Schur solve、数值残差或数据完整性
失败；但修正后的 postcheck 也不能改变原始 Slurm accounting。

### 9.4 关键 provenance

- plan file SHA-256：`b5b496976863d93da8084002dfcd6bfff951dcc0e9ae193469b67b28658596b4`；
- canonical plan SHA-256：`8f1f701925b87d15bde10dbdda86359cdd5dc141c0c825b499fceb2ccdcd6ff5`；
- validation config SHA-256：`d221a62520b467ebd03a82f104e47f9249cd411d165ec943632c26d3f2c80eb5`；
- metadata SHA-256：`bb2944eb920e6512dac75c4b57004cc68440899e8b411021f14e8229923e0438`；
- Q/u/p SHA-256：`26161b1e...a7314af`、`80cdda72...cd9003`、`f08f6086...e46e89`；
- diagnostics NPY/CSV SHA-256：`11b2b83b...61ddd`、`3d5e5649...54538`。

输出根为 `/home/fansenwei/data_beris_issue7_ff084fb_preflight_20260901`，控制证据目录为
`/home/fansenwei/beris_issue7_ff084fb_preflight_20260901_control`。后者保留原始 stdout、
stderr、runner log、命令、GPU trace 和修正后只读 postcheck 报告。

Plan 顶层的 `beris_edwards_stokes_issue6` 是历史 schema 标签，不表示本次执行了问题 6 的
其他 stage。

## 10. 最终关闭状态

问题 7 的状态必须按层次记录：

- CPU 独立方程级门禁：**PASS / CLOSED**；
- 指定配置下的 R512 H100 单步模型执行与 runner 验证：**PASS**；
- 数据、provenance 和数值残差：**PASS**；
- 端到端 Slurm wrapper 门禁：**FAIL / NOT CLOSED**，原因为模型结束后的 CSV 整数解析错误；
- 问题 7 的科学与软件方程级结论：**closed with an infrastructure caveat**。

因此，这次包装脚本错误不重新打开问题 7 的方程实现结论，但也不能声称完整调度工作流
已经通过。若未来需要严格的 `COMPLETED/0` 审计记录，应先把修正后的 post-validator
版本化，再经用户明确授权进行 workflow-only 复核；不得把它误写成第二次物理模拟。

上述 GPU preflight 只是一个时间步的生产集成、精度路径和显存 smoke test。它不验证
float32、GPU friction、`dealias=none`、其他边界条件、时间/空间收敛、长期稳定性、多 seed
统计、defect core、activity scan 或严格 Shendruk benchmark agreement。有限数组和很小的
残差证明离散约束求解正常，不证明物理解已经收敛。
