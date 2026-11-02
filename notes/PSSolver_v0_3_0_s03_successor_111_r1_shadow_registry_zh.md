# S-18 / R1：影子应用注册表，尚未切换分派

本步依据 D-0018，在 R5 已收口的 dev/0.4 上新建独立分支。
新增 `pssolver/configuration/application_registry.py`：只含标准库、不可变
描述符和只读索引；四个应用、六条运行路径。没有新的公开扩展接口。

现有编译器、lowering、binding、capability 和构造分派均原样保留，
没有任何现有运行时模块导入新表。编译器顺序与 lowering 错误中的
支持组合顺序本来就不同，本表分别保存；binding 的历史插入顺序也保留。
对照测试核对键、字段、顺序、错误上下文、编译器调用参数、工厂身份、
包级 builder 选择、只读性和未知键拒绝；它们不是数值替身验收。

原型提交 2475cefd314d82296fa98d04c232b7116cc06df2 的 4 个 topology
负向测试在 SimulationSpec 的更早边界组合校验处失败，不是运行时回归。
本提交改为复制后故障注入，以单独检查 lowering 的保护条件，保留失败
日志及 JUnit；没有 amend。更早两个未提交的测试原型曾错误地对派生属性
required_capabilities 和包装类 TorchSpectralExecution 使用 dataclasses.replace，
已在提交原型前修正，没有修改生产代码。

本步不宣称 R1 或 S-18 完成；不改变 R1-R9 登记项，也不提交最终验收请求。
下一步在此影子基线上核对完整声明的旧/新分派行为并切换唯一内部入口，
仍保留旧路径为负对照。R1 最终候选还须具备全部已发布 checkpoint 的
只读兼容、writer 字节树（包括两种 functional 导出）、故障矩阵、干净
wheel、Control main 已安装候选 CPU 全量及其余共同门禁，再创建 R-0059。

本阶段验证计划：只含可达对象的克隆中跑 conda、a2 基线硬链接克隆及
屏蔽外部路径的 CPU 全量；复用冻结 R4 helper 做 8 family × 4 种刷新
间隔 × 20 步，对父提交及 v0.3.0 比较状态、进度和身份。运行结果写入
外部独立归档，不倒填本记录为未执行的成功。pycache 如由 python -I 子进程
写入克隆内，会如实盘点，并随临时克隆修剪，不宣称全部重定向成功。

没有修改 Functional API、beris_edwards.py 或 q_tensor.py 的 blob、数值、
checkpoint 格式、发布路径、GPU 专用代码、阈值或 skip/xfail。
没有 push、tag、Release、PyPI、HPCC、Slurm、GPU 或 H100 操作。
