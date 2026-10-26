# S-09 R-0024 最终本地 support 候选

本记录绑定 R-0023 要求的纯 fixture、测试和部署合同返工。运行时资格源仍为 `85ba3ae`，
Provider wheel 仍为 `40036450…`；没有修改 `pssolver/`。

- V-37 的两个真实 v0.2.2 迁移 payload 已按 manifest 的精确哈希提交，守卫测试要求
  manifest 列出的所有文件都由 Git 跟踪；
- Provider support archive 从 `c66fd9e` 重建，包含 fixture 修复、R-0023 导入和 D-2
  部署合同，但排除授权草稿自身以避免哈希自引用；
- HPCC 环境改为 `python -m venv --system-site-packages` 后用
  `pip install --no-deps --no-cache-dir` 安装，禁止跨挂载 hard-link clone；
- 草稿给出提交前 `logs/` 创建、完整 `S09_*`/`sbatch` 命令、全部新建目录及其保留或清理策略。

V-35、V-36 保持 `claimed_fixed`，V-37 保持 `registered`，等待 R-0024 独立复核；A3-18
与 NEW-14 不变。本记录不授权 push、HPCC、Slurm、GPU 或 H100。
