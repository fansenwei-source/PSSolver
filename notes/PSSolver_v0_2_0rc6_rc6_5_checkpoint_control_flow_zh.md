# PSSolver v0.2.0rc6 RC6-5 functional checkpoint 控制流合同收尾

## 结论

RC6-5 / A5-12 已完成本地 CPU 关闭，分类为：

`PASS_RC6_5_FUNCTIONAL_CHECKPOINT_CONTROL_FLOW_CPU_CLOSURE`

本阶段没有 push，没有提交 H100 作业，没有修改 PSSolver-Control、nematics3d、生产默认值、科学方程、积分器、变换算法、梯度算法或 checkpoint schema。

## 修复内容

PyTorch 的 non-reentrant checkpoint 在默认 early-stop 重计算完成后，会抛出内部 `_StopRecomputationError` 作为控制流哨兵，并在 checkpoint 自己的调用帧中捕获。此前 stable functional facade 把该哨兵翻译成 `FunctionalError`，改变了异常类型，导致 Periodic 和 Channel 的默认 checkpoint 反向传播都失败。

RC6-5 在公共错误翻译之前，仅识别精确的 owning module 与 qualified class name，并原样重新抛出这个控制流哨兵。实现没有导入 PyTorch 私有异常类；若 PyTorch 将来改变内部类型，识别会关闭而不会扩大透传范围。普通 PyTorch 异常、用户异常以及空消息异常仍按原有公共合同翻译。

## 验证结果

- RC6-5 直接合同测试：3 passed。
- Periodic 默认 non-reentrant checkpoint 梯度与非 checkpoint 参考逐位一致且有限。
- Channel 默认 non-reentrant checkpoint 梯度与非 checkpoint 参考逐位一致且有限。
- 普通空消息 `Exception()` 仍精确翻译为带 operation/details 的 `FunctionalError`。
- 受影响的 functional、checkpoint、Phase 9 与架构边界测试：78 passed。
- 登记前完整 CPU suite：3914 passed，1 skipped，7 deselected，8 subtests passed；唯一 failure 是预期的历史 successor 尚未登记。
- 最终完整 CPU suite：3920 passed，1 skipped，7 deselected，8 subtests passed，0 failed，0 xfailed。
- 唯一 skip：可选 `nematics3d` 依赖不可用；没有新增 skip 或 xfail。
- `git diff --check`：PASS。

## 历史证据处理

旧资格 JSON 与旧发布记录均未改写。RC6-5 通过 `historical_successor_sources` 登记 `pssolver/functional/api.py`、changelog 和 RC4.2.4 审计测试的前驱哈希、当前哈希及原因；旧审计测试只增加读取 RC6-5 forward-only successor 的能力。

机器可读记录：`notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json`。

## 后续边界

RC6-5 完成后可进入 RC6-6 的合同核对与本地实现；本记录不授权自动合并、默认值提升、push 或 H100 submission。
