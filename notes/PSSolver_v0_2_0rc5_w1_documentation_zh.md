# PSSolver v0.2.0rc5 W1 文档对齐记录

W1 已完成，分类为 `PASS_RC5_W1_DOCUMENTATION_ALIGNMENT`。本批只对齐公开文档、示例、注释和历史 successor 记录，没有改变运行时行为、数值实现、科学阈值或生产默认值。

完成内容：

- README 新增一份可逐字执行的、使用正交模型/几何/边界声明的完整 Plane `Simulation` 示例，并修复历史 Plane 示例的 `save_start_step`；
- README 明确各 runtime 的 execution-option 所有权和 `InvocationSpec` 的真实导入位置；
- capability 文档明确 `prescribed_q`、`strong_homeotropic_q`、`strong_planar_q` 仅可声明、不可执行，继续在 runtime 构建前 fail closed；
- CHANGELOG 改为陈述已经发布且不可变的 rc4 状态，不再暗示公开 static-Q lifting 已获得资格；
- Robin `boundary_residual` 被明确说明为构造诊断，而不是内部 Helmholtz 算子的独立正确性门禁；
- fixed-work pressure gradient 被明确说明为精确有限迭代算法的导数；需要收敛控制的优化应使用 tolerance mode；
- NEW-4、NEW-5 已关闭，B30 的公开解释已完成。当前下一批为 `rc5_must`。

本批没有 push，没有 H100 提交，没有修改 PSSolver-Control 或 `nematics3d`，没有重新开放 lifting，也没有修改 v0.2.0rc4 tag、Release 或 PyPI 产物。

验证结果：定向 CPU 门禁 `152 passed`；完整隐藏 CUDA 的 CPU suite 为 `3422 passed, 7 skipped, 0 failed, 0 xfailed`。七个 skip 与 W0 基线相同，由六个 CUDA-only 测试和一个可选 `nematics3d` 测试组成。
