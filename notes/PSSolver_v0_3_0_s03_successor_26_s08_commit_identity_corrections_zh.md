# PSSolver v0.3 S-08 提交身份前向勘误

本记录导入 R-0020 的 records-only 返工结论，不改写 successor 14、successor 25 或任何已发布历史记录。

- S-08 返工实现的真实提交为 `e2343d19301de146ca632a1a12f7c9659e598f73`；先前记录中的 `e2343d16a146cfc9fc362e3fb556d589480c404c` 不存在。
- successor 14 的真实父提交为 `1b6321a9d0e43a49077974f13a7a5989cae20ea0`；原记录中的长哈希不存在。
- 另有十个发布历史记录中的长哈希由七位短哈希错误手工补全。本记录逐项绑定错误值、真实 Git 提交和原始文件，原文件保持不动。
- 新的守卫测试要求当前登记表的每个 `fixed_in` 和 successor 的 PSSolver 提交身份都能由 Git 解析；跨仓库 `result_commit` 被明确标为 PSSolver_sync 身份。
- `V-33`（S-10）与 `V-34`（S-08）登记为 `registered`；A2-25、N5、V-31、V-32 保持 `claimed_fixed`，等待 R-0021。

本批只改记录与记录测试；未修改运行时、数值、checkpoint schema、阈值或默认值，未执行外部操作。
