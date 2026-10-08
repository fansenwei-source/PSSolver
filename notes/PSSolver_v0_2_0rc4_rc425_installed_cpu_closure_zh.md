# PSSolver RC4.2.5 installed-wheel 跨版本与 Consumer CPU 关闭

## 结论

RC4.2.5 已完成。由精确 Git archive 构建并安装的 PSSolver 与 PSSolver-Control wheel 在隔离环境中完成 95 项关闭测试：Provider 35 项、Consumer 60 项，全部通过，且没有 skip、xfail、collection error 或源码 shadow import。

该结果关闭了 M01–M16 跨版本矩阵、当前 checkpoint round trip、已登记 legacy reader、未知或不兼容身份的 pre-mutation 拒绝、迁移 source 不变性，以及 Periodic/Channel Consumer 的 checkpoint、身份 oracle、checkpointed/full-history gradient、finite-difference、Taylor 和负向门禁。Functional API 仍为 `1.0`，兼容策略和两个 functional bridge 均为 v3。

## 环境恢复

第一次环境构造继承了已有 `nematics3d 0.9.0b1` 的严格 NumPy metadata，与该环境实际 NumPy 版本冲突，因此在 `pip check` 后、95 项测试启动前停止。没有修改、卸载或重装 `nematics3d`。

正式 v2 使用不含 `nematics3d` 的干净 Eva 基础环境，并从本机离线包物化缺少的 SciPy 与 pytest。`pip check` 通过，`nematics3d` 在资格 venv 中明确不存在。该隔离只用于资格测试，不改变任何生产环境或 dependency metadata。

## 完整 CPU 回归

增加 qualification runner 后的第一次完整 suite 为 `2879 passed, 7 deselected, 8 subtests passed`。冻结本记录和记录测试后，最终 suite 为 `2884 passed, 7 deselected, 8 subtests passed`；七个 deselect 精确对应既定 CUDA-only node IDs，其他 failure、skip 或 xfail 均为零。

## 范围与下一步

本阶段没有修改 runtime 热路径、checkpoint 接受规则、stable functional API、PSSolver-Control 源码、`nematics3d` 或生产默认值，也没有运行 H100。PSSolver-Control 的既有 source commit `03dbc39` 已通过新 Provider installed-wheel 累计再资格。

RC4.2.5 完成后可以规划 RC4.2.6；若要声明 CPU checkpoint 与 CUDA restore 的设备可移植性，应在该独立阶段运行冻结的单次 H100 portability 门禁。当前没有授权 H100、push、merge、默认值提升或发布。
