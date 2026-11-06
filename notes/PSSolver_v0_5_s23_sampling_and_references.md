# S-23：采样边界、可重放基准与独立参考

本批是 D-0044 / D-0045 的本地预做，等待 S-22（R-0071）先验收，
之后再单独验收 S-23。不是 0.5 运行时、GPU 或发布资格，不启动 S-25。
S-22 分支保持 64fc24d；本分支不修改 pssolver、打包配置、版本和门限。

## 1. 连续与重启比较的边界

旧草稿在连续轨迹的 23/40 步执行 checkpoint，但在从 23 步恢复的轨迹中
漏了 40 步 checkpoint 的同步。由此比较了不同事件阶段的约束场。
新 `sampling.sample` 对两条轨迹执行同一顺序：观测前、观测、checkpoint、
静态场准备后的 pre-update callback、一步更新后。没有重写任何运行时标志。

`_static_fields_are_current` 是恢复/复用一次的 token，不是物理时间戳。
Periodic/Plane 在恢复边界为 true，连续采样可能是 false；准备静态场时
该 token 被消费，所以 callback 中也不能以它为 true 断言场是否同步。
字段字节、step_count、refresh_count、refresh_interval 必须精确相同。
Channel tolerance 模式恢复后的构造器 A·p 缓存与 pressure_guess 的对象身份
不匹配，不能命中；它和 last_pressure_iterations 可能只在恢复那一步不同。
下一步必须恢复相同。原始值全部保留，跨版本/两次基线仍对整个 JSON 精确比较。
任何新的时间点、类型或范围的元数据差异都使分析器失败。

观测开/关（P5）在静态场准备后、显式 RHS 前比较真实输入场，64 步逐位相同。
不能把未同步的水动力缓存叫作同一物理时刻的观测。

## 2. 矩阵与导数

123 个场景：104 接受、19 拒绝。网格、dtype、64/32 步、23/40 重启及
Plane 刷新 20 均按 §6.1；两种 Plane runtime_path 均覆盖。
包含 CPU eager/compile（真实 Inductor 探针成功），相同工作绝对路径顺序复用，
每次独立新编译缓存。基线两次和候选的 123 个原始记录逐字节相同。
所有被接受的函数式场景追加整个 32 步终态和观测的固定余切 VJP，所有梯度有限。

v0.4.0 实际边界如实保留：functional Channel 只接受 float64、warm_start=True；
functional 的 pointwise compile 被拒绝；none 去混叠搭配 full 投影才被接受。
float32 Channel 的 1e-10 tolerance 声明被拒绝，另跑其既有默认 1e-6 的接受声明；
未替换或删除原来的拒绝用例，更未调整求解器门限。

`tests/baselines/v0_4` 保存逐步 hash 表，约 1.3 MB。分类新增
`baseline_fingerprint`：当前 D-0045 数值依赖和四线程指纹必须匹配；不匹配
直接运行该类即明确失败。不同环境只能显式按类别排除此类，并重新 A/B 测量，
不能凭已有哈希宣称数值通过。可移植表只去掉输出路径相关的 identity 对象；
外部三次原始记录仍完整比较 configuration/runtime identity，不删字段。

## 3. checkpoint

由指定 v0.4.0 wheel 实际写出六组小网格夹具，约 512 KB，含 manifest 的
完整文件大小和 SHA；所有 .npy 必须被 Git 跟踪。每组加载后续算 16 步，
v0.4.0 和候选逐位相同且夹具树不变。另对已有 v0.2.0 的五个保留家族做
16 步精确续算；历史夹具、历史 oracle 和旧版本信息均不改写。

格式名以实际元数据为准：生产 Periodic/Channel 和 Plane 当前写 3；
functional Channel 写 3；functional Periodic 的外层仍是 2（桥接 API 1.1），
不能把它人为标成设计简表所写的 3。已有 rc1/rc2/rc3 格式与迁移检查继续保留。
S-23 增加未知格式在张量读取前拒绝的六条检查。
未来引擎对 legacy active-force Channel、canary、lifting 的拒绝和映射属于
S-24/S-25 之后，当前不修改旧 reader 或伪造旧 checkpoint。

## 4. MMS 与结构

`tests/mms/derive_primitives.py` 独立于 PSSolver，在 SymPy 专用环境推导，
两次表达式清单 SHA 均为
`c3b7a649b4a8819398e1de976fa555c75f470936ccac18e413de00b90ab77238`。
14 项：FFT/DCT/DST 节点/单元中心的往返、导数、Laplacian；扩散的隐式
Euler 离散模态解与一阶收敛；周期/自由滑移 Stokes；独立 NumPy 基函数
组装的 no-slip Schur 算子、PCG 和转置点积。固定门限来自已有同类测试，
不是跨版本非零容差。最初探针的零常量 dtype 和 batch 维遗漏在参考脚手架
中修正，原失败日志保留；未修改解析解或放宽误差界。

涉及 ARCHITECTURE §6（qualification）和 §7（测试、独立参考）；层边界未变。
入口 trajectory_hashes → sampling/RuntimeCase → 已安装 runtime；
VJP 独立走 Functional API。复用 benchmarks/_artifact_io 的原子结果写入。
MMS 的符号解/独立矩阵是有意保留的 oracle，不合并进生产内核、无退役计划；
RuntimeCase 只作版本化资格适配器，后续引擎提供新适配后由独立验收决定退役。

SymPy skill 使推导与数值实现保持独立，记录版本和表达式哈希；四项 Python
工程 skill 用于职责拆分、分类和失败关闭测试。方法来源：Kassis, T.,
Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026),
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)
（2026-11-06 核对最新为 2026-09-02 修订版）。工具方法不是资格证据。

S-19 作业 10945436 / R-0065 的 H100 吞吐证据仅按已有 successor 128/129
引用为历史参考；未访问 HPCC，未重测、未外推 CPU 到 GPU，E3 正式 A/B 在 S-31。
