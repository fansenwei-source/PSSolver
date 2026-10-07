# PSSolver RC4.1.1 Plane Nyquist 修复

RC4.1.1 修复了 RC4.1.0 在 `v0.2.0rc3` 上确定性复现的 B5：Plane free-slip Stokes 在偶数周期网格上没有统一采用实网格一阶导数的 Nyquist 约定，导致 `full_complex` 与 `hermitian_half` 给出不同的物理解。

## 实现

修改范围只有 `pssolver/linear_solvers/stokes/plane_free_slip.py` 以及直接相关的诊断、测试和历史源码 successor 登记。

求解器现在对 `x`、`y` 两个周期轴分别执行以下规则：

1. 如果该轴长度为偶数，将其 Nyquist 一阶导数乘子设为零；
2. 建立两个周期轴 Nyquist 平面的并集 mask；
3. 在 tangential 和 normal Helmholtz 路径中投影这些平面；
4. 在 pressure gauge、Schur solve 和 pressure-gradient 路径中采用同一投影；
5. mask 注册为 `persistent=False`，所以 `state_dict` 的 key、shape 和 dtype schema 不变。

这与已经资格化的 Periodic/Channel Stokes 约定一致。修复处理两个周期轴，而不是只处理 RC4.1.0 中暴露 full/half 差异的非压缩轴。

## 数值结果

同一组固定随机实空间力得到：

| case | 修复前速度差 | 修复后速度差 | 修复前压力差 | 修复后压力差 |
|---|---:|---:|---:|---:|
| 16×16×8，half-y | 3.0402e-2 | 3.5974e-16 | 1.0465e-1 | 3.4841e-16 |
| 16×15×8，half-y | 2.7553e-2 | 3.7349e-16 | 1.0867e-1 | 3.5864e-16 |
| 16×16×8，half-x | 4.3109e-2 | 2.8471e-16 | 1.1498e-1 | 2.6983e-16 |

默认偶数 case 中，两种存储的 native divergence、独立 collocation divergence 和 pressure residual 也都通过，collocation divergence relative L2 约为 `2.1e-15`。

## 非回退合同

- 两个偶数周期轴的纯 Nyquist force 在 `full_complex` 与 `hermitian_half` 下都产生精确零解；
- 奇数周期网格在两种存储下都与 rc3 原公式逐字节相同；
- 既有 manufactured Stokes、pressure diagnostics、energy budget、lifting、公共 facade 和 Plane runtime 测试通过；
- P7.7.0 的历史哈希没有改写，只把 `plane_free_slip.py` 加入已有的 `subsequently_connected` 集合；
- `state_dict` schema、公开 API 和生产默认值均未改变。

## 当前边界

RC4.1.1 的 CPU 修复已经完成，但 B5 尚未正式关闭。还需要以本提交为候选做一次 installed/source identity 检查和单次 H100 非回退资格验证，确认 R128/R320 Plane timestep、显存、transform calls、有限性和生产数值结果没有不可接受的回退。

本步骤没有修改 PSSolver-Control、`nematics3d`、checkpoint identity、lifting 或 Channel pressure policy。它也不授权默认提升或发布。

机器可读记录见 [PSSolver_v0_2_0rc4_rc411_plane_nyquist_repair.json](PSSolver_v0_2_0rc4_rc411_plane_nyquist_repair.json)。
