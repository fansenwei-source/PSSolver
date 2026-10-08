# PSSolver RC4.2.6 CUDA-test TF32 进程引导恢复

状态：`READY_RC4_2_6_TF32_PROCESS_BOOTSTRAP_RECOVERY_NOT_SUBMITTED`

## 问题

Recovery v3 的 H100 preflight P01–P08 全部通过，但随后独立启动的 CUDA-test Python 进程没有继承 preflight 进程对 PyTorch TF32 flag 的赋值。七个 CUDA-only tests 中六个通过；Plane Nyquist 测试观察到 `torch.backends.cudnn.allow_tf32=True`，而冻结合同要求它为 `False`。

这属于外部 qualification harness 的进程边界错误。Job `10861272` 没有启动 X01–X14 checkpoint 矩阵或正式科学 timestep，也没有运行 analyzer；它不支持 solver、checkpoint、数值或科学失败结论。

## 修正

`benchmarks/run_rc426_cuda_tests.py` 现在在同一个即将进入 `pytest.main()` 的 Python 进程中显式执行：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

runner 会在启动 pytest 之前将以下内容原子写入 `preflight/cuda_test_tf32_policy.json`：

- 两个 flag 的设置前值；
- 两条赋值是否成功及异常；
- 两个 flag 的设置后值；
- fail-closed enforcement 结果。

如果任一赋值失败或设置后值不是 `False`，pytest 不会启动；CUDA-test 汇总仍会持久化失败状态。正式 `cuda_only_tests.json` 使用 schema v2，并记录 TF32 policy 文件路径、SHA-256 和 enforcement 结果。

Slurm harness 只增加 `--policy-output` 参数，没有修改 scheduler、七个 node ID、X01–X14 matrix、analyzer 或门限。

## 本地验证

- 定向合同测试：32 passed。
- 真实 PyTorch runner smoke：进入 runner 时 `cudnn.allow_tf32=True`，设置后两个 flag 都为 `False`；policy 在 pytest 前写入，最终报告绑定的 SHA-256 正确。
- 完整 CPU suite：2916 passed，7 deselected，8 subtests passed，0 failed，0 skipped，0 xfailed，用时 201.91 秒。
- `git diff --check`：PASS。

## 边界

本次只修改 qualification support、其测试和恢复记录。没有修改 runtime、checkpoint reader/writer、数值实现、PSSolver-Control、`nematics3d` 或生产默认值；没有运行 H100、提交 Slurm、push、merge 或发布。

下一步需要单独明确授权一次新的 RC4.2.6 H100 recovery submission；不得自动重试。
