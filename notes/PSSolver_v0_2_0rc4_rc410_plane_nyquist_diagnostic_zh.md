# PSSolver RC4.1.0 Plane Nyquist/storage 诊断

这一步只复现并定位 `v0.2.0rc3` 已知问题 B5，不修改求解器、生产默认值或公开 API，也不使用 H100。基线是不可变的 `v0.2.0rc3` 提交 `5071d73a00e0d19be62ebd39edf9918818d1267a`。

## 复现方法

诊断入口为：

```bash
python -m benchmarks.diagnose_plane_nyquist_storage --summary-only
```

它使用 CPU `float64`、固定随机种子 `24680`、`execution_order="real_first"`、`dealias_rule="none"`、friction `0.23` 和 viscosity `0.73`，把同一组三个实空间随机力分别交给 `full_complex` 与 `hermitian_half` Plane free-slip Stokes 求解器。

## 结果

| case | shape | half-spectrum 轴 | 非压缩周期轴 | 速度最大 relative L2 | 压力 relative L2 |
|---|---|---:|---:|---:|---:|
| even x/y | 16×16×8 | y | x | 3.0402e-2 | 1.0465e-1 |
| odd x/y control | 15×15×8 | y | x | 3.3696e-16 | 3.8704e-16 |
| only x even | 16×15×8 | y | x | 2.7553e-2 | 1.0867e-1 |
| only y even | 15×16×8 | y | x | 3.6849e-16 | 3.5977e-16 |
| rotated storage | 16×16×8 | x | y | 4.3109e-2 | 1.1498e-1 |

默认 `hermitian_axis=1` 时，`y` 是压缩轴、`x` 是非压缩周期轴。只有 `x` 为偶数且输入含有 `x` Nyquist 分量时，两种存储的物理结果出现明显差异；只有 `y` 为偶数不会引起存储差异。把 half-spectrum 轴改为 `x` 后，触发方向也从 `x` 旋转到 `y`。

只从输入力中删除触发的非压缩周期轴 Nyquist 平面，不删除其他模式，所有偶数触发 case 的速度和压力差异都恢复到 `4.3e-16` 以下。这构成了因果对照，而不只是相关性观察。

## 为什么现有残差测试没有发现它

在出现 3% 速度差和 10% 压力差时，两个存储路径各自的 native modal divergence 与 pressure residual 仍约为 `1e-16`。原因是求解器使用同一套未归零的 Nyquist 一阶导数乘子构造 saddle operator 和残差；因此它能精确满足自己的离散方程，却不一定满足偶数实网格共同采用的 Nyquist 约定。B5 需要跨存储物理结果 oracle，不能只依赖 native residual。

## 本步骤的结论与边界

B5 已被确定性复现并定位为：**偶数、非压缩周期轴上的 Nyquist 分量使 Plane full-complex 与 Hermitian-half 路径采用不同的实网格语义。** 奇数网格是干净对照，交换 half-spectrum 轴会交换触发轴，删除单一触发平面会恢复存储等价。

这里的输入过滤只用于证明因果关系，不是生产修复方案。Plane 的合格修复应像已经资格化的 Periodic/Channel Stokes 一样，对两个偶数周期轴统一处理实网格一阶导数 Nyquist 约定，并一致投影 force、pressure/velocity solution 和 diagnostics。

因此当前状态是：

- RC4.1.0 诊断与 oracle 冻结完成；
- B5 仍未关闭；
- 下一步允许进入 RC4.1.1 实现；
- 实现后仍需 manufactured Stokes、奇数网格逐字节不变、偶数网格两种存储等价、完整 CPU suite 和单次 H100 非回退资格验证；
- PSSolver-Control 与 `nematics3d` 均不需要在本步骤修改。

机器可读记录见 [PSSolver_v0_2_0rc4_rc410_plane_nyquist_diagnostic.json](PSSolver_v0_2_0rc4_rc410_plane_nyquist_diagnostic.json)。
