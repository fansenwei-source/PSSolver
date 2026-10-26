# S-09 R-0026 recovery 候选

本记录只绑定资格支持层返工。Provider 运行时仍是 `85ba3ae…`，三个 wheel、源码 archive、
192 项 profile matrix、性能门限、案例和阶段顺序都没有变化。

`V-39` 与 `V-40` 由 `72286694c43ee15e5859c0658c989431404fe505` 修复；
`eba9560f7c9ee73064f8b64a6448038984d22f47` 补齐 CPU 演练对子进程的参数传递。
最终 support 源是 `719ae76069ba97b88a893ccf7404ddf18239fcc6`，其非遮蔽 archive
SHA-256 为 `7500e889a01733d998ae08c121f12de00bf1d5a831f9eefdcc14fd763843b3cd`。

最终 installed-wheel CPU 演练为 `PASS_S09_INSTALLED_WHEEL_CPU_REHEARSAL`：三套安装环境身份、
短时程、TF32、checkpoint、数值对照、独立 Consumer 与 192 项正式 profile 规划均通过；
CUDA 身份、CUDA-only tests、跨设备 checkpoint 和正式 profile 执行按合同只在 H100 阶段运行。

本记录把 `V-39`、`V-40` 推进为 `claimed_fixed`，等待 `R-0026` 独立复核；`NEW-14`
保持 `open`。本记录不授权 push、HPCC 写入、Slurm 或 H100。
