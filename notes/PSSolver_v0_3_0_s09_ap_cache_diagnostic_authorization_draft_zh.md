# S-09 A·p 缓存 analysis-only H100 诊断授权草稿

> 此文件是 R-0028 复核材料，不是用户授权。R-0028 PASS 后必须依据 D-0009
> 重新生成带完整绑定哈希的一键授权文本，再由用户明确授权。

依据：R-0028（必须为 PASS）。

拟授权一次非资格的 single-H100 诊断，比较 v0.2.2 baseline、v0.3 candidate
cache-on 与同一 candidate wheel 的 harness-only cache-off。诊断固定为 72 个
profile，不改变 runtime、wheel、冻结计划、门限或案例，不判定 S-09 通过。

正式授权文本必须完整列出并绑定：runtime、support commit、两个 Provider wheel、
Provider 与 support archive、生成后的 H100 plan、job script、profile、child、runner、
analyzer 及本草稿的 SHA-256。

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
7. 只提交一个 H100 Job，不自动重试、不提交第二个 Job；保存原始 `sacct`、
   `scontrol`、完整授权原文和 action 记录；
8. 无论成败都归档完整证据；取回并核对清单后才删除两个 venv；
9. 不包含 push、tag、Release、PyPI、Control push 或任何源码、阈值、计划修改。

本草稿中的路径和哈希在 R-0028 PASS 前故意不冻结，不能直接执行。
