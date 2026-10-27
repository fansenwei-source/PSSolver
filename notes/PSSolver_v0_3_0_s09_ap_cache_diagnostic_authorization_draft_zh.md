# S-09 A·p 撤回代理 analysis-only H100 诊断授权草稿

> 此文件是 R-0030 复核材料，不是用户授权。R-0030 PASS 后必须依据 D-0009
> 重新生成带完整绑定哈希的一键授权文本，再由用户明确授权。

依据：R-0030（必须为 PASS）。

拟授权一次非资格的 single-H100 诊断，比较 v0.2.2 baseline、v0.3 candidate
cache-on 与同一 candidate wheel 中由 harness 精确编译的 `f94b9be` revert-proxy。
诊断固定为 72 个计时 profile，并另运行两次不计时的 R512 CUDA allocator
snapshot；不改变 runtime、wheel、冻结计划、门限或案例，不判定 S-09 通过。

正式授权文本必须完整列出并绑定：runtime、support commit、两个 Provider wheel、
Provider 与 support archive、生成后的 H100 plan、job script、profile、child、runner、
analyzer、revert-proxy、snapshot helper 及本草稿的 SHA-256。

正式授权还必须写明：

1. 新 control/output/venv 目录在上传前均不存在；
2. 上传后、创建 venv 前在 HPCC 核对全部文件 SHA-256；
3. Provider 与 support 目录只能从已核对的 tar 解出，support 中不得含
   `pssolver/` 或 `pssolver_control/`；
4. 基于批准的 HPCC 基线环境创建 baseline/candidate 两个 venv，使用
   `--system-site-packages` 和 `pip install --no-deps --no-cache-dir`；
5. 两个 venv 都用 `python -I` 核对 `pssolver.__file__` 位于自身 site-packages、
   版本、`pip check` 与 wheel RECORD 摘要；
6. 登录节点在 `sbatch` 前原样重放 job 开头的全部 `test` 与哈希检查；
7. 作业清除 `LD_LIBRARY_PATH`，固定 GCC/G++ 7.3.0，加载 `cuda/12.1`，使用带
   EXIT trap 的节点本地 `TMPDIR`；矩阵前持久化 `CC --version`、`module list`、
   `nvcc --version`，并复用主 S-09 的空 cache CUDA `torch.compile` 预检；
8. 只提交一个 H100 Job，不自动重试、不提交第二个 Job；保存原始 `sacct`、
   `scontrol`、完整授权原文和 action 记录；
9. 72 个计时 profile 使用三角色全部六种排列；单 trial 比值使用冻结
   `_profile_center`，不得用原始样本均值替代；
10. 两次 allocator snapshot 与计时矩阵隔离，不进入冻结统计；CPU profiler
    结果只作不可复现的参考，不能作为一压力场差异或 V-41 机制的裁决证据；
11. 无论成败都归档完整证据；取回并核对清单后才删除两个 venv；
12. 不包含 push、tag、Release、PyPI、Control push 或任何源码、阈值、计划修改。

计划中的控制目录为：

`/home/fansenwei/pssolver_v0_3_s09_ap_cache_diagnostic_h100_85ba3ae_20261026_v1`

本草稿中的最终绑定哈希须由 R-0030 candidate record 给出；R-0030 PASS 前不能
直接执行。
