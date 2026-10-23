# S-09 R-0023 最终本地复核候选

最终运行时资格源为 `85ba3aef75548d549d9db453ca4a5ab604b17a9d`。
V-35 与 V-36 的本地返工、双环境完整 CPU、installed-wheel Provider/Consumer
门禁、artifact 重建及 H100 合同修正均已完成。依照 R-0022 的明确要求，两项在登记表中
仍保持 `registered`，等待 R-0023 独立复核。

最终结果：conda 4696 passed；pip 4695 passed、1 个 optional nematics3d skip；Provider
installed 7 passed；Consumer installed 351 passed、6 个既有 CUDA/benchmark skip；全部零
failure，source shadow 为 false，pip check 通过。

H100 草稿已逐项列出分析器冻结的性能规则、HPCC 上传目录、必须初始不存在的输出目录、
所有 `S09_*` 变量和三套 Python 环境来源，并明确 API 1.1 checkpoint 不能由 v0.2.x
读取。当前没有 push、HPCC、Slurm、GPU 或 H100 操作，也没有 H100 授权。
