# Successor 72：S-16 / R-0043 返工登记

本记录向前更正 successor 70 的“R7/R9 已完成、R8 已盘点”结论，不改写
successor 70 或 71。

- `V-50`：`80e96ba6cdcd4a16c26c79ac4eea4aa1d35cb992` 恢复精确的顶层模块
  导入棘轮、transforms 外观规则和两条精确 stokes 实现边。
- `V-52`：同一提交把 R-0043 列出的 56 个完全等价编码点中的 55 个迁入
  `core._canonical`。唯一保留的是 `io/checkpoint_identity.py` 的本地 encoder，
  因为该 schema-independent primitive 已有禁止导入任何 `pssolver` 模块的硬门禁。
  lifting/robin 的 tensor-header digest 使用不同 wire format，也保持本地。
- `V-51`：`835ee25c2a30067a8899e3d61dee0c7adc935f98` 用执行式屏蔽门禁更正盘点，
  得到 27 个测试文件、35 个失败和 13 个外部根，与 R-0043 逐项相同。

三项均登记为 `claimed_fixed`，等待 S-16 最终验收。R8 的 pytest 自包含迁移和本机
只读核对工具尚未实施；本记录不把任何同步仓库中的外部状态当作实施授权。

本次不改公开接口、数值算法、GPU 专用路径、checkpoint schema、门限、skip/xfail、
生产默认值或历史记录，也不授权 push、tag、Release、PyPI、HPCC、Slurm 或 H100。
