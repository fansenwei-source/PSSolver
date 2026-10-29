# S-16 / R8 执行式依赖盘点更正

本记录向前更正 `notes/PSSolver_v0_4_0_s16_r8_inventory.json` 的计数，不改写旧记录，
也不实施 R8。

在仅含 `dev/0.4` 可达对象的新克隆中，移除 `origin`，并通过 Python audit hook
把克隆、Python 基线和审计插件之外的 `/home/fansenwei` 路径全部模拟为不存在。
`7e9871c` 的完整 CPU suite 因外部依赖产生 **35 个失败，分布于 27 个测试文件**；
该集合与 R-0043 的独立结果逐项相同。其余结果为 4869 passed、7 个 CUDA-only
deselected 和 8 个 subtests passed。

第一版盘点遗漏 9 个测试文件和 6 个归档根，因为那些路径来自记录 JSON 字段，
而不是测试源码中的直接字面量。v2 记录列出全部 35 个测试、13 个外部根及逐根大小。

本次只完成 V-51 的盘点更正。历史记录保持逐字节不变，不新增 skip/xfail，
不改运行时、接口、门限、checkpoint schema 或生产默认值。R8 的 pytest 自包含迁移和
本机只读核对工具必须等待用户另行通知后实施。
