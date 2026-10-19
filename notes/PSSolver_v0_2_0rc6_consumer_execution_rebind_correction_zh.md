# PSSolver RC6 Consumer execution 设备重绑定修正

RC6 recovery v4 的 H100 Job `10901131` 已通过环境、CUDA-only、公共合同、TF32、
当前与历史 checkpoint、八个数值族以及 192/192 profile。Consumer CPU suite 也得到
`344 passed, 1 skipped`。唯一停止点位于 qualification helper 的 CUDA finite smoke：
helper 对 `TorchSpectralExecution` 调用 `dataclasses.replace()`，从冻结 dataclass 基类重放
了自定义公开构造器不接受的 `backend` 参数，因此在 runtime 构建和科学 timestep 之前触发
`TypeError`。

本批次只修正 `benchmarks/run_rc6_consumer.py`。新的 helper 要求输入确实是公开
`TorchSpectralExecution`、backend 为 `torch_spectral`、旧 device 合法且
`fallback_allowed` 精确为 false；随后移除由公开构造器拥有的 `device` 与
`fallback_allowed` options，通过公开构造器在目标 device 上重建，并验证旧对象完全未改、
新旧 metadata 除 `execution.options.device` 外完全相同。它不会访问私有 Provider 路径。

使用冻结 Consumer `208edd0...` 的真实 `periodic_runtime.py` 和 `channel_runtime.py` 做了
CPU construction/rebind smoke：两个 runtime 均保留 backend、runtime path、其余 options 和
禁止 fallback 的合同，只把 device 从 `cpu` 改为 `cuda`。

本地验证结果为：全部 RC6 定向测试 `156 passed`；完整 CPU suite `4038 passed`、
`7 deselected`、`8 subtests passed`，无 failure、skip 或 xfail。7 个 deselect 仍精确为既定
CUDA-only node ID。

旧 v4 失败归档和更早记录均保持只读；本记录只登记 forward-only source successor。
没有修改求解器、数值实现、公共 API、checkpoint schema、阈值、PSSolver-Control、
nematics3d 或生产默认值。本批次没有 push 或 H100 权限；正式 recovery 仍需单独授权。
