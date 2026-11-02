# S-18 / R1：私有注册表分派切换

先完成 successor 111 的独立影子阶段，再把 compiler、lowering、binding、capability 与两层构造选择接到同一份不可变私有注册表。各组合的具体实现、数值调用、公开名称与版本、校验和错误顺序不变。旧分派源码只在测试中从可达父提交加载，作为负对照；生产没有第二入口或回退。

`configuration/simulation.py` 有意不改：它定义通用声明与组合验证，并无重复的应用构造分派。`configuration/package_construction.py` 的实现名称表只是规范化后的 provenance 元数据，不是运行时选择；保持原样。32 个具体配置、编译器、规划、运行时与 workflow 文件冻结。Plane legacy 的 request 类型检查仍由下层构造器完成，不提前插入 guard。

本地候选门禁：两套新克隆全量、屏蔽外部路径、8 family × 20 步（四种刷新间隔）对父提交及 v0.3.0 的状态/进度/身份逐位比较、已发布 checkpoint 只读兼容、故障矩阵、含 functional Channel/Periodic 的 writer 字节树、干净 wheel 与已安装 Provider 的 Control main 全量。最终实测结果在外部归档与后续记录中绑定，本记录不预称门禁或独立验收通过。

API 与两个 Control constitutive blob 未变，不触发迁移；仍要做 Control main CPU 资格化。发布机制、所有 writer 及其调用图不变，不触发真实 NFS。没有修改 GPU 专用实现；正式 GPU 资格留到 S-19。不得把 check_sync 对此前 R5 的 PASS 自动推断当作 R1 已完成。

早期未提交的定向测试有两例因 TimeStepping 包装器不接受 dataclasses.replace 的字段而失败；改用基础 TimeIntegrationSpec，生产代码没有据此调整。修正后 147 项通过。不采用非零容差，不改阈值或 skip/xfail。没有 push、远端操作或 GPU 授权。
