# R-0017 S-07 独立复核导入

R-0017 对本地 S-07 候选给出 `REWORK_REQUIRED`。六项原有 S-07 问题继续保持
`claimed_fixed`，不提前声明为 `verified`。

新发现 `V-30` 登记为 high / registered / 0.3 / S-07：候选 wheel 排除了
`pssolver.control`，但保留的 `pssolver.channel` 仍导入其中的 active-force helper，导致
legacy 和 compiled_v2 两条 installed-wheel Channel runtime 都无法构建。

本记录只导入独立复核结论，不修改 runtime、打包配置或历史证据。S-07 必须完成依赖闭包
返工和新的 installed-wheel / real-path 门禁后，再以 R-0018 请求复核。
