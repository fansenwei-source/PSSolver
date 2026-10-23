# PSSolver v0.3 S-09 single-H100 提交授权草稿（待 R-0023 复核）

本文件不是授权，也不表示已经提交 H100。只有 R-0023 返回 PASS 后，才能依据 D-0009 把下列内容转成给用户的一键授权文本。

## 绑定身份

- Provider 运行时资格源：`85ba3aef75548d549d9db453ca4a5ab604b17a9d`
- Provider qualification-support 源：`9dd83cf2bec1a01bde5fc4e7b5e3a4afbd1ad3ad`
- Consumer 稳定基线：`208edd03e6b26a0e788c07d295fddd7a67237189`
- Consumer migration：`48dd4d3a9a101f8b24d8311c57ed67fab46ceea0`
- Consumer evidence：`97d3c06450c4a7a6d72e0fe07a7a157ef095c133`
- v0.2.2 baseline wheel：`a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e`
- Provider candidate wheel：`4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8`
- Consumer candidate wheel：`f7ae4c1f5f3c73e0be540b5e4ec5e05a9eef662867a149325462835820ae94ef`
- Provider source archive：`bff21905eb7d4c5e8313b10c871f40d8b96b411975260fdb0edb75b16a99a3b9`
- non-shadowing Provider support archive：`2810f1d033bb012354712c3f351f1113ac36348db0a176e1a3ea488de43990c9`
- Consumer source archive：`02f3e8249bc02cbb0484636d9f8bf46cc8a6815d90e13f8286f7aaf8291157fd`
- Consumer support archive：`2243f14cbdad1a58cb8a1ff73e2bb550752d97f5edf2190302005d9d69ff66a1`
- 冻结 H100 plan SHA-256：`1ad29c4887a8e79610227a84e1e7e6b6846da246df8ce1f805aadb3ff336c620`
- Slurm script SHA-256：`4c0757bf13ee031b42407c105ffaf203b4571fe5fe51b60bd7de3834476f92f5`

Provider support archive 从上述 support 源的 `benchmarks/`、`jobs/`、`tests/` 和
`notes/` 构建，但明确排除本授权草稿自身，避免 archive 哈希与草稿内哈希形成自引用；
archive 不含 `pssolver/`，不能遮蔽 installed wheel。

## HPCC 路径与环境

全部上传输入、只读 archive、wheel、support tree、job script、日志和最终控制证据放在：

`/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1`

`S09_OUTPUT_ROOT` 固定为下列路径，而且在提交前必须不存在：

`/scratch1/vincent/PSSolver/data_pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1`

三个 installed-wheel Python 环境都必须从固定基线 `/home/fansenwei/miniconda3/envs/Nematics3D` 新建独立 hard-link clone，再分别使用 `--no-deps` 安装对应 wheel；不得复用含源代码 editable install 的环境。固定变量如下：

```text
S09_BASELINE_PYTHON=/scratch1/vincent/PSSolver/venvs/pssolver_v03_s09_85ba3ae_baseline/bin/python
S09_CANDIDATE_PYTHON=/scratch1/vincent/PSSolver/venvs/pssolver_v03_s09_85ba3ae_candidate/bin/python
S09_CONSUMER_PYTHON=/scratch1/vincent/PSSolver/venvs/pssolver_v03_s09_85ba3ae_consumer/bin/python
S09_BASELINE_WHEEL=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/artifacts/pssolver-0.2.2-py3-none-any.whl
S09_CANDIDATE_WHEEL=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/artifacts/pssolver-0.3.0.dev0-py3-none-any.whl
S09_CONSUMER_WHEEL=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/artifacts/pssolver_control-0.0.1.dev0-py3-none-any.whl
S09_PROVIDER_SUPPORT=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/provider_support
S09_CONSUMER_SUPPORT=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/consumer_support
S09_PROVIDER_REFERENCE=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/provider_85ba3ae
S09_CONTROL_ROOT=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1
S09_OUTPUT_ROOT=/scratch1/vincent/PSSolver/data_pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1
```

三个环境均须记录克隆来源、Python 3.10.18、PyTorch 2.5.1+cu121、CUDA runtime 12.1、NumPy 1.22.3、SciPy 1.8.1、`sys.path`、导入路径、wheel/RECORD 摘要、`pip check` 和 source-shadow 判定。

## 拟授权范围

授权上传上述只读 archive、wheel、资格 helper 和 job script 到 HPCC，并提交一次 PSSolver v0.3 S-09 single-H100 cumulative qualification Job。不得自动重试，不得提交第二个 Job。

作业必须使用：`hagan-gpu`、`hagan-lab`、`medium`、`gpu:H100:1`、8 CPUs、64G、3 小时、`--no-requeue`。在任何资格阶段前完成 node-local `/tmp`、`RENAME_NOREPLACE`、host compiler、installed-wheel import、CUDA device identity 和 TF32 初始状态门禁。任何门禁失败必须立即停止并完整归档。

阶段顺序固定为：环境/编译器 → 7 项 CUDA-only tests → 三个 CUDA t=200 Hermitian 长时程 → forward/backward TF32 scope → checkpoint/restart/released fixture → 数值/梯度/Taylor/replay → 独立 Control consumer → 192 项 performance/memory profiles → cumulative analyzer/checksum。独立 Control consumer 必须先于性能矩阵。

性能合同保持冻结：8 runtime families × 2 grids × 6 trials × baseline/candidate = 192 profiles；每个 profile 60 个正式样本；交替 A/B 顺序；全部样本保留。每个 family/grid 单元格必须同时满足：

- 6 个 trial ratio 中至少 4 个 `<= 1.05`；不要求 6/6 全部 `<= 1.05`；
- baseline→candidate 顺序层的 ratio 中位数 `<= 1.05`；
- candidate→baseline 顺序层的 ratio 中位数 `<= 1.05`；
- 两个顺序层中位数的较大值/较小值 `<= 1.05`；
- 全部样本均值比 `<= 1.03`；
- 六个 profile mean 的中位数之比 `<= 1.03`；
- 六个 trial ratio 的中位数 `<= 1.03`；
- peak allocated group ratio 与 peak reserved group ratio 均 `<= 1.03`。

不得删样本、替换 trial、放宽门限或在失败后自动重跑。

Functional API 1.1 checkpoint 是有意的一向格式边界：v0.3 仍可读当前 API 1.0 checkpoint；API 1.1 checkpoint 不能由 v0.2.x 读取，必须得到明确的 `unsupported` 格式错误，不能被误读为旧格式。`execution_scope()` 在作用域内拥有 forward/backward 的 TF32 策略；作用域之外执行 backward 时仍使用调用者当前的 TF32 设置。

作业必须持久化完整环境、输入/源码/运行时文件/wheel/RECORD 哈希、全部 JSON、pytest 日志与 JUnit XML、Slurm provenance、checksums.sha256；所有门禁通过并复核 checksum 后才能写 COMPLETE。A3-18 与 NEW-14 只有在该 H100 结果通过且再经 Claude 独立验收后才能改为 verified。

本授权不包含 push、tag、Release、PyPI、PSSolver-Control push、生产默认值修改、阈值修改或科学 Control 结果声明。
