# PSSolver RC6 recovery v6：script/import root 分离

## 结论

recovery v5 的失败不是求解器、CUDA、数值或科学失败。Job 10901181 在任何预检和科学阶段开始前，由外部 sbatch 的单一 root 合同自相矛盾而停止：同一个目录既要提供 `benchmarks/` 脚本，又不能暴露完整归档中的 `pssolver/` 源码。

本提交只修正 qualification bootstrap，不修改历史 runner。v6 使用两个不同目录：

- source-free script root：只从 Git archive 提取 `benchmarks`、`jobs`、`notes`、`tests`；
- source-free import root：只含生成的 `benchmarks/__init__.py`，其 `__path__` 指向 script root 的 `benchmarks/`。

`PYTHONPATH` 只指向 import root。因此 `benchmarks.*` 可以从固定且哈希绑定的资格脚本导入，而 `pssolver` 必须继续从指定 installed-wheel venv 的 `purelib` 导入。

## Fail-closed 门禁

新 validator 在 CUDA module、node-local `/tmp` 预检和 11 阶段执行之前，逐项记录并验证 18 个 predicate，包括：解释器身份、两个 root 的普通目录身份、源码隔离、initializer 精确内容、stage-plan schema、11 个脚本绑定的路径与 SHA-256、正式输出尚不存在，以及 installed-wheel `pssolver` 和 source-free `benchmarks` 的实际 import identity。

所有 observation 会先原子写入并回读，再根据最终 `passed` 返回 0 或 1。目标已存在时拒绝覆盖。

## 当前边界

本地 `py_compile`、`bash -n` 和 5 个定向测试已经通过。本轮尚未 push，也没有提交或申请 H100。下一步是在 HPCC 上制作 source-free archive、两个独立 root 和冻结 stage plan，并完成无 GPU 提交前审计；完成后再单独请求 push 和唯一一次 recovery v6 H100 授权。

机器可读记录见 `PSSolver_v0_2_0rc6_recovery_bootstrap_root_separation.json`。
