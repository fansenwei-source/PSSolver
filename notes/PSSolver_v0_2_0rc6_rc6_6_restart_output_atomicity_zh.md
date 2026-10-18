# PSSolver v0.2.0rc6 RC6-6：restart 输出目录原子性

## 结论

`A2-18` 已完成本地 CPU 关闭：当 restart checkpoint 的完整性、身份或
runtime 兼容性验证失败时，如果目标输出目录原先不存在，它在失败后仍然不存在。

## 修正

- Periodic complete-stress、Channel complete-stress、Plane 和 legacy Channel
  四条公共 workflow 都把缺失目标目录的创建延后到 restart preflight 成功之后。
- fresh run 的输出行为不变；既有非空目录的拒绝合同不变。
- checkpoint schema、科学方程、积分器、transform、梯度算法、runtime 默认值和
  production 默认值均未改变。

## 验证

- 四类拒绝路径：`4 passed`；均断言失败后目标目录不存在。
- 受影响的非记录测试：`62 passed, 7 deselected`。
- 历史记录审计子集：`1044 passed`。
- 完整 CPU suite：`3926 passed, 1 skipped, 7 deselected, 8 subtests passed`。
- 唯一 skip 是未安装可选依赖 `nematics3d`；七个 deselect 均为既有 CUDA-only
  node ID。

## 边界

本阶段未运行 H100、未修改 PSSolver-Control 或 nematics3d、未 push，且不授予
automatic merge 或 default promotion。
