# S-25 / R-0085 前向修正

R-0086 §2 明确要求在本步这份 successor 144 中补登记：MANIFEST 按字母序加入
memberwise_transforms.py；四个新增测试以 R1 和分类器的原始完整 chains 登记。
此补充通过新提交完成，不 amend ca9919d；successor 143 及更早记录保持不变。
T19 harness 排除 options 中重复的 device，只传显式 device，并在冻结前对两侧
运行 CPU 构造加一步检查。该脚本位于新的证据目录，不改包内实现。

按设计 r1.10 §24，仅在 Engine 的 batch > 1 时安装私有逐成员变换适配器。
单分量四维输入使用 narrow 视图，参数原样传入，逐成员结果沿 batch 维 cat；
batch 维不符时拒绝。batch = 1 原 backend 对象与调用路径不变，五维输入不拆分。
新增 T14b（batch = 3，各精度 8 步），T14 的逐位标准不变。

successor 143 和所有旧记录不改写。共享 FFT、两处冻结调用、数学实现、参考、
容差和 §9 两项比较例外不变。此记录只冻结返工内容，不预先声明资格通过。
新候选一次重跑五项试跑与全部资格；T19 依据 D-0052 使用本机 GPU，
先加锁，估计 20 分钟、硬上限 40 分钟，只记录不作门禁。无 HPCC 或推送。
