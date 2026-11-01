# PSSolver v0.4 S-18 / R4 本地候选

分类：`R4_LOCAL_CPU_CANDIDATE_AWAITING_INDEPENDENT_REVIEW`

本批按照 D-0018，在 `9ffe2fe9d148e10bc5ec6c23cdac373e5ebdb280`
之上共享半隐式 Euler 的精确顺序执行器、谱刷新时钟、state-backed
逆变换/刷新实现，以及 runtime adapter 的进度读取和恢复逻辑。运行时候选为
`9cf54553b9f1bb2b94c2807c2e47adedeb415846`。

共享逻辑只位于原有依赖方向允许的内部层。首次实现曾让顶层 integrator 和
workflow 校验依赖 execution；`9cf5455` 已将这些位置恢复为原控制流，并由架构
棘轮确认没有增加依赖边。这个更正不改变已经共享的 state-backed、compiled-v2
和 runtime adapter 路径。

硬合同保持不变：Python 和张量操作顺序、原地更新与别名所有权、real-first
变换顺序与次数、谱刷新决定和计数、generation、callback 时机、异常边界、
restart 边界以及所有已发布 checkpoint 的只读兼容性。Functional API 仍为
1.1；Control 使用的两个 Functional 路径以及 Beris--Edwards、Q-tensor blob
没有修改。没有容差、门限、skip 或 xfail 的变化。

CPU 验收需要两套全量测试、屏蔽外部路径测试、八种 family 每步二十步逐位
对比、`None/1/2/7` 四种刷新周期、刷新边界 restart、已发布 checkpoint 读取、
Functional Periodic/Channel writer 字节树、错误进度负例、干净 wheel 和 Control
main 安装版全量测试。CUDA 图、CUDA transform/restart 和 H100 性能内存仍归
S-19；本批不运行 GPU/H100。

本记录不授权 push、tag、Release、PyPI、HPCC、Slurm 或 H100。
