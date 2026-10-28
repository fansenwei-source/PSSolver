# R-0036 S-10 集成记录返工

`R-0036` 确认 S-10 的运行时和三方合并内容正确，但在只包含分支可达对象的新克隆中
发现 `V-46`：第一版集成候选仍引用变基前的 10 个 S-10 提交。那些对象只由本地 reflog
维持，不能成为远端候选的实时身份。

本次只修改记录和测试。`V-33`、`V-43` 保留 `R-0033` 验证结论，同时把 `fixed_in`
从原始复核提交 `61838af…` 前向更正为 patch-id 和 `pssolver` 树均相同的可达重放提交
`a540740…`。successor 58 提供完整 10 项原提交→重放提交映射；集成测试只使用可达
提交、`d1de8fb8…` 运行时树和从 `ff70b4c…` 开始的精确 10 提交序列。

原 S-10 头 `b0a2034…` 仅通过本地 `refs/archive/s10-pre-integration` 保存，不推送。
以后任何经过 replay、rebase 或 amend 的候选都必须在 `git clone --no-local
--single-branch` 且删除 `origin` 后的独立克隆中通过两套完整 CPU 门禁。

`V-46` 现为 `claimed_fixed`，等待 `R-0037` 复核。本记录不授权 push、HPCC、Slurm、
GPU、H100、tag 或 Release。
