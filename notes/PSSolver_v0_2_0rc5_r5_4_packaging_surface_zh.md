# PSSolver v0.2.0rc5 R5-4 打包边界与 installed Plane provenance 收尾

## 结论

R5-4 已完成，分类为：

`PASS_R5_4_PACKAGING_SURFACE_AND_INSTALLED_PLANE_CPU_CLOSURE`

本项关闭 consolidated report 中的 `NEW-2` 与 `NEW-3`。没有修改数值方程、runtime、checkpoint schema、公开 API 或生产默认值，也没有修改 PSSolver-Control 与 nematics3d；本项不需要 H100。

## 修复

- `setup.py` 只发现 `pssolver` 与 `pssolver.*`，仓库内部 `benchmarks` 不再安装为顶层包。
- 历史 `python -m Plane_beris_edwards_stokes` shim 继续保留，console script 也继续保留。
- shim 只负责转发 CLI，不再计入 Plane 数值实现 provenance。
- Plane provenance 的所有键均以 `pssolver/` 开头，因此公开 Plane 运行不再依赖 site-packages 顶层 shim。

## 验证

- 修复前 wheel 含 88 个 `benchmarks` 文件；去掉 shim 后两条 Plane runtime 均因 provenance 读取失败。
- 修复后 wheel 顶层只有 `Plane_beris_edwards_stokes.py`、`pssolver` 与 dist-info，`benchmarks` 文件数为 0。
- 从 sdist 再构建的 wheel 得到相同顶层合同。
- console script 与历史 shim 的 `--help` 均通过。
- 在复制的 installed package 中完全不提供 shim 时，`legacy_production` 与 `compiled_v2` 均完成 1 步。
- source 完整 CPU：`3470 passed, 7 skipped, 8 subtests passed, 0 failed`。
- installed-wheel 定向 CPU：`35 passed, 0 failed`。
- installed-wheel 适用完整 CPU：`3463 passed, 7 skipped, 8 subtests passed, 0 failed`。
- `pip check`：PASS；source shadow import：false。

证据目录：

`/home/fansenwei/pssolver_rc5_r54_installed_cpu_4678be3_20261014_v1`

## 下一步

下一项为 R5-5。R5-4 不授权 push、自动合并、默认提升或 rc5 发布。
