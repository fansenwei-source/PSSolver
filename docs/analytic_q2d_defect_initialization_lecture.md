# 从解析的 ±1/2 缺陷构造二维 nematic Q 场

## 0. 这份讲义要解决什么问题？

我们的目标是在一个二维周期区域中构造

\[
Q_{\mathrm{2D}}(x,y),
\]

使它包含若干个 \(+1/2\) 和 \(-1/2\) nematic 缺陷。这个二维场随后可以沿
\(z\) 方向复制，成为贯穿通道的竖直缺陷线，再叠加一个很小的三维 twist
扰动。

需要先区分两个概念：

1. **解析缺陷初态**可以用公式直接构造；
2. **统计稳态二维活性湍流**是混沌动力学状态，不能仅靠一个闭式公式得到，
   必须让解析初态经过二维动力学预热。

下面的算法是对多篇论文中不同结果的工程化组合，不是某一篇论文逐行给出的
现成算法。

## 1. 文献依据

### 1.1 单个 ±1/2 缺陷的角度场

Vromans 和 Giomi 在一常数 Frank 模型中写出单缺陷的最低能角度场

\[
\theta=k\phi+\theta_0,
\]

其中 \(k=\pm1/2\)，\(\phi\) 是围绕缺陷核的极角。他们也讨论了
\(+1/2\) 与 \(-1/2\) 缺陷的取向以及如何从离散 \(Q\) 数据中识别取向。

- A. J. Vromans and L. Giomi, *Orientational properties of nematic
  disclinations*, Soft Matter **12**, 6490–6495 (2016),
  [DOI](https://doi.org/10.1039/C6SM01146B),
  [arXiv](https://arxiv.org/abs/1507.05588).

### 1.2 两个缺陷不能总靠简单角度叠加精确描述

Tang 和 Selinger 同样从 \(\theta=k\phi+\theta_0\) 出发，并用 conformal
mapping 求解具有指定相对取向的双缺陷场。他们特别指出：简单相加两个
`atan2` 项可以放对缺陷的电荷和位置，却一般不能独立控制两个缺陷的取向。

- X. Tang and J. V. Selinger, *Orientation of topological defects in 2D
  nematic liquid crystals*, Soft Matter **13**, 5481–5490 (2017),
  [DOI](https://doi.org/10.1039/C7SM01195D),
  [arXiv](https://arxiv.org/abs/1706.05065).

### 1.3 用复序参量和 Q 张量处理缺陷核

Pismen 把二维无迹对称 \(Q\) 写成复序参量

\[
\chi=p+iq=\rho e^{i2\theta}.
\]

这样 nematic 的 \(\pm1/2\) 缺陷变成复场 \(\chi\) 的整数 \(\pm1\) 涡旋。
他还给出无量纲缺陷核振幅方程

\[
\rho''+\frac{1}{r}\rho'-\frac{1}{r^2}\rho
 +(1-\rho^2)\rho=0,
\]

其中 \(\rho(0)=0\)，远离缺陷时 \(\rho\to1\)。这为“缺陷核处降低
序参量”的做法提供了直接依据。

- L. M. Pismen, *Dynamics of defects in an active nematic layer*,
  Physical Review E **88**, 050502(R) (2013),
  [DOI](https://doi.org/10.1103/PhysRevE.88.050502),
  [arXiv](https://arxiv.org/abs/1308.3364).

### 1.4 为什么缺陷核要用 Q，而不能只用 director？

Schopohl 和 Sluckin 的经典工作研究了 Landau–de Gennes 理论中的 nematic
缺陷核结构。核心思想是：director 在缺陷核处没有定义，而 \(Q\) 张量仍可以
通过降低有序度或产生更一般的张量结构保持有限。

- N. Schopohl and T. J. Sluckin, *Defect Core Structure in Nematic Liquid
  Crystals*, Physical Review Letters **59**, 2582–2584 (1987),
  [DOI](https://doi.org/10.1103/PhysRevLett.59.2582).

### 1.5 解析缺陷初态为什么还要做活性预热？

Giomi、Bowick、Ma 和 Marchetti 从一个周期盒中的 \(\pm1/2\) 缺陷对出发，
用二维 active-nematic 方程研究其运动、湮灭和缺陷增殖。他们展示了活性足够
强时新缺陷对不断产生，动力学进入频繁产生和湮灭缺陷的混沌状态。因此，
“有一对解析缺陷”与“已经达到二维活性湍流稳态”不是一回事。

- L. Giomi, M. J. Bowick, X. Ma, and M. C. Marchetti, *Defect Annihilation
  and Proliferation in Active Nematics*, Physical Review Letters **110**,
  228101 (2013), [DOI](https://doi.org/10.1103/PhysRevLett.110.228101),
  [arXiv](https://arxiv.org/abs/1303.4720).

## 2. 最少的物理背景

### 2.1 Nematic 为什么没有箭头的头和尾？

普通向量满足 \(\mathbf n\ne-\mathbf n\)，但 nematic director 满足

\[
\mathbf n\equiv-\mathbf n.
\]

在二维中可以写成

\[
\mathbf n=(\cos\theta,\sin\theta,0),
\]

但 \(\theta\) 和 \(\theta+\pi\) 描述同一个物理状态。

### 2.2 什么是 +1/2 和 -1/2 缺陷？

沿闭合曲线绕缺陷一周：

- \(+1/2\) 缺陷使 director 角度总共增加 \(\pi\)；
- \(-1/2\) 缺陷使 director 角度总共减少 \(\pi\)。

一般写成

\[
\oint d\theta=2\pi k,
\qquad k=\pm\frac12.
\]

### 2.3 为什么数值计算更适合使用 Q？

director 在缺陷中心无法定义。\(Q\) 同时包含方向和有序度，并允许有序度在
缺陷中心下降，所以不会出现无限大的数值量。

不同论文对 \(Q\) 的归一化不同。例如二维理论常用

\[
Q_{ij}=S\left(n_i n_j-\frac{\delta_{ij}}2\right),
\]

而当前三维 solver 统一使用 de Gennes 标量有序度约定

\[
Q_{ij}=\frac{3S}{2}\left(n_i n_j-\frac{\delta_{ij}}3\right).
\]

因此单轴 \(Q\) 的本征值为 \((S,-S/2,-S/2)\)，也就是
\(S=\lambda_{\max}(Q)\)。不同归一化的系数不能直接混用。

## 3. 单个解析缺陷

在位置 \((x_a,y_a)\) 放置电荷 \(k_a\) 的缺陷。先定义相对坐标

\[
\Delta x=x-x_a,\qquad \Delta y=y-y_a,
\]

以及极角

\[
\phi_a=\operatorname{atan2}(\Delta y,\Delta x).
\]

director 角度取为

\[
\theta(x,y)=k_a\phi_a+\theta_{0a}.
\]

\(\theta_{0a}\) 控制缺陷图案的朝向，但不改变拓扑电荷。

## 4. 多个缺陷的第一近似

给定缺陷列表

\[
\{(x_a,y_a,k_a)\}_{a=1}^{N},
\]

最简单的远场近似是

\[
\theta(x,y)=\theta_{\rm bg}+\sum_{a=1}^{N}k_a\phi_a(x,y).
\]

在周期区域中必须满足

\[
\sum_a k_a=0,
\]

所以 \(+1/2\) 和 \(-1/2\) 的数量应相等。否则周期盒中无法容纳这个净拓扑荷，
除非再引入边界、电荷背景或其他补偿结构。

这个简单叠加适合生成初态，但不是任意多缺陷问题的精确最低能解。尤其是：

- 它不能独立指定每个缺陷的取向；
- 周期接缝可能不光滑；
- 缺陷之间的真实弹性形变会偏离简单叠加。

因此后面还需要周期投影和短时间被动松弛。

## 5. 给缺陷加一个有限大小的核

director 公式在 \(r=0\) 奇异。定义到第 \(a\) 个缺陷的距离 \(d_a\)，并使用
一个从 0 平滑上升到 1 的核函数，例如

\[
g(d_a)=\tanh\left(\frac{d_a}{\xi_Q}\right).
\]

\(\xi_Q\) 是缺陷核宽度。多个缺陷可用

\[
S(x,y)=S_{\rm eq}\prod_a g(d_a)
\]

作为简单标量有序度模型。这样在任意缺陷中心 \(S\to0\)，远离所有缺陷时
\(S\to S_{\rm eq}\)。

`tanh` 是方便的近似，不是 Pismen 核方程的精确解。更精确的办法是预先数值
求解径向方程得到 \(\rho(r/\xi_Q)\)，再用插值表代替 `tanh`。

## 6. 从角度和振幅生成 solver 所需的 Q

先计算

\[
n_x=\cos\theta,\qquad n_y=\sin\theta,\qquad n_z=0.
\]

按照三维 solver 的约定：

\[
Q_{ij}=\frac{3S(x,y)}{2}
\left(n_i n_j-\frac{\delta_{ij}}3\right).
\]

五个独立分量为

\[
\begin{aligned}
Q_{xx}&=S\left(\frac14+\frac34\cos2\theta\right),\\
Q_{xy}&=\frac{3S}{4}\sin2\theta,\\
Q_{xz}&=0,\\
Q_{yy}&=S\left(\frac14-\frac34\cos2\theta\right),\\
Q_{yz}&=0.
\end{aligned}
\]

第六个对角分量由无迹条件确定：

\[
Q_{zz}=-(Q_{xx}+Q_{yy}).
\]

## 7. 周期边界是算法中最容易踩坑的地方

### 7.1 周期最短距离

对于长度为 \(L_x\) 的周期方向，应使用

\[
\Delta x_{\rm p}
=\Delta x-L_x\operatorname{round}(\Delta x/L_x),
\]

\(y\) 方向同理。距离为

\[
d=\sqrt{\Delta x_{\rm p}^2+\Delta y_{\rm p}^2}.
\]

### 7.2 仅使用 minimum image 还不保证全局光滑

`atan2` 的分支切线和 minimum-image 坐标的跳变可能在周期接缝产生高频误差。
建议采用下面的生产流程：

1. 用解析公式生成带正确缺陷位置和电荷的近似场；
2. 对 \(Q\) 或复序参量 \(\chi\) 做一次周期 Fourier 投影/温和低通；
3. 在周期边界下进行短时间被动 Landau–de Gennes 松弛；
4. 松弛初期固定缺陷核附近的小区域，防止正负缺陷立即移动或湮灭；
5. 接缝平滑后解除固定。

如果需要数学上严格的 torus 解析解，应使用周期 Green 函数、Ewald 求和或
Jacobi theta functions；这通常不是初次实现所必需的。

## 8. 如何验证构造正确？

### 8.1 拓扑电荷检查

使用复序参量

\[
\chi\propto(Q_{xx}-Q_{yy})+2iQ_{xy}=|\chi|e^{i2\theta}.
\]

沿一个网格 plaquette 绕一圈，对 \(\arg\chi\) 的相邻差做 \((-\pi,\pi]\)
wrap，然后求和：

\[
m=\frac{1}{2\pi}\sum_{\square}\Delta\arg\chi.
\]

- \(m=+1\) 对应 nematic \(+1/2\)；
- \(m=-1\) 对应 nematic \(-1/2\)。

### 8.2 缺陷核检查

缺陷位置附近应看到：

- \(|Q|\) 明显下降；
- director 绕核旋转 \(\pm\pi\)；
- 周期边界附近没有额外的假缺陷链。

### 8.3 全局检查

应满足：

- 正负缺陷总数相等；
- 总拓扑荷为零；
- \(Q\) 对称且无迹；
- 被动松弛后能量下降；
- 未固定时缺陷可以自然移动，而不是被网格接缝卡住。

## 9. 从“解析缺陷场”到“二维活性湍流快照”

推荐分三阶段：

### 阶段 A：解析播种

随机放置等量的 \(\pm1/2\) 缺陷，并设置最小间距，避免缺陷核重叠。

可以用活性长度

\[
\ell_a=\sqrt{K/\zeta}
\]

作为初始缺陷间距的量级参考，但前面的无量纲系数需要通过二维模拟校准，
不能由解析理论唯一确定。

### 阶段 B：被动清理

暂时令活性为零，只保留 Landau–de Gennes 松弛，消除解析拼接、周期接缝和
过强的局部梯度。这一步不是为了达到最终统计态，而是为了得到数值上平滑的
初态。

### 阶段 C：二维活性预热

恢复目标 \(K,\zeta,\eta,\gamma,\lambda\)，在二维中运行到下列统计量不再有
系统漂移：

- 缺陷数密度；
- 平均动能或 RMS 速度；
- 涡量和速度相关长度；
- 平均标量序参量。

最终保存的这个场才适合作为

\[
Q_{\mathrm{2D}}(x,y)
\]

输入三维 Fig. 4 benchmark。

## 10. 可直接实现的伪代码

```text
input:
    periodic box Lx, Ly
    grid Nx, Ny
    equilibrium principal order S_eq
    core radius xi_Q
    neutral defect list [(x_a, y_a, k_a), ...]

assert sum(k_a) == 0

for every grid point (x, y):
    theta = theta_background
    S = S_eq

    for each defect a:
        dx = periodic_minimum_image(x - x_a, Lx)
        dy = periodic_minimum_image(y - y_a, Ly)
        phi = atan2(dy, dx)
        distance = sqrt(dx*dx + dy*dy)

        theta += k_a * phi
        S *= tanh(distance / xi_Q)

    n = (cos(theta), sin(theta), 0)
    Q = uniaxial_Q(n, S)

periodically smooth/project Q
passively relax Q while temporarily pinning cores
release pins
run 2D active dynamics to statistical steady state
verify charges and save Q_2D
```

## 11. 对 Fig. 4 初始化最重要的结论

1. 解析公式可以替代外部二维输入文件，生成一个包含竖直缺陷线所需的二维
   缺陷场；
2. 解析公式不能替代二维活性预热，因为湍流的缺陷密度和相关结构是动力学
   选择的；
3. 周期区域必须电荷中性；
4. 缺陷核必须通过 \(Q\) 振幅下降或完整 Landau–de Gennes 松弛正则化；
5. 简单角度叠加是好用的初始化近似，但不是任意缺陷取向的精确解；
6. 三维扩展前，应先确认二维场已经具有稳定的缺陷统计。

## 12. PSSolver 中的解析周期缺陷气体生成器

PSSolver 现在注册了 `analytic_periodic_defect_gas_2d`。它使用矩形 torus
上的 Jacobi-\(\vartheta_1\) 涡旋相位，而不是直接拼接 minimum-image
`atan2`，因此复 nematic 序参量

\[
\chi=(Q_{xx}-Q_{yy})+2iQ_{xy}=\frac{3S}{2}e^{i2\theta}
\]

在两个周期方向都没有相位接缝。

```python
from pssolver.models.active_nematics import (
    create_initial_condition,
    sample_periodic_neutral_defects_2d,
)

parameters = dict(
    lengths=(100.0, 100.0),
    num_defect_pairs=6,
    min_separation=10.0,
    core_radius=1.5,
    seed=24,
    S_initial=1.0 / 3.0,
)

Q_2d = create_initial_condition(
    "analytic_periodic_defect_gas_2d",
    (256, 256),
    **parameters,
)

# Optional: recover the exact positions and charges generated by the same seed.
positions, charges = sample_periodic_neutral_defects_2d(
    lengths=parameters["lengths"],
    num_defect_pairs=parameters["num_defect_pairs"],
    min_separation=parameters["min_separation"],
    seed=parameters["seed"],
)
```

返回值是五个形状为 `(Nx, Ny)` 的 float32 CPU tensors：

```text
Qxx, Qxy, Qxz, Qyy, Qyz
```

这里 `min_separation` 和 `core_radius` 使用与 `lengths` 相同的物理长度单位。
这里显式设置的 `S_initial=1/3` 控制初始 Q 的远场幅值。当前
benchmark 中它恰好等于由 \(A=0,B=-0.3,C=0.3\) 决定的
`S_bulk=1/3`，但两者语义不同：前者属于初始条件，后者属于
热力学模型。该初始化器已经直接生成
(Q=(3S/2)(nn-I/3))，送入 `extruded_2d_twist` 后不应再次缩放。
