# Successor 87：S-17 R3 / R6 本地实现候选

本 successor 在不改写任何历史记录的前提下，向前登记 D-0017 的 R3/R6
本地实现提交 `667735ed79154ecdaae25a5861eb2fd33391c90e`。

- Functional API 1.1 与两个 Control 固定的 constitutive blob 不变；
- 宽泛 Functional facade 和四个 pilot package facade 进入 v0.4 弃用窗口，
  但继续可用；
- 新增 `pssolver.experimental.boundary_pilots` 显式入口；
- Robin 公开执行与 lifting 仍 fail-closed；
- checkpoint reader、schema、数值算法、GPU 路径、门限和默认值均未改变。

本记录只授权本地候选与 R-0050 复核，不授权 push、发布或 H100。
