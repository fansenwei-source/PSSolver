# S-22 工程底座与 R-0070 的前向处置

依据已确认 D-0042、D-0043 和 R-0070 的 R1–R4。架构地图对应第 2、3、7、9 节；
这里只接入当前已经成立的两条导入契约，不提前实施新数值引擎。

## 范围和历史处置

- 从 e32ac21 开 0.5.0.dev0；当前版本记录由
  notes/pssolver_active_version_record.json 显式指向开发记录，冻结时再推进到发布记录。
- 运行时仅改 _version.py 与 visualizer.py 的 pygame 延迟导入、错误提示；
  setup.py 仅增加 visualization extra。不改数值计算、门限、登记表、benchmark 或 jobs。
- MANIFEST.in 先清除 setuptools 的自动来源，再逐文件允许 wheel 实际发布的源码、
  顶层入口、README、CHANGELOG 和构建文件。setuptools 生成的 PKG-INFO / egg-info /
  setup.cfg 是构建元数据，不是额外源码；PKG-INFO 按 README 生成的 METADATA 同义审计。
- D-0043 RQ1：四个旧 release 测试读取 4c12228 的 MANIFEST.in，原有文件存在断言不变。
  rc6 的允许后继集合只新增 MANIFEST.in 及实际变化的旧测试路径，不修改任何旧哈希。
  新的 manifest 边界由独立测试核对，不用注释、先 include 再排除或读取替身绕过断言。
- D-0043 RQ2：CHANGELOG 两个历史行原字节保留；隐私扫描只允许指定文件整体哈希、
  指定行内容哈希各出现一次。全包其他命中失败；Consumer 名称每一处都列进构建报告。
- 新 successor 按历史哈希链向前绑定以上变化；旧 successor、发布记录及冻结测试门限不改。

## 工程检查与证据分类

新增 tools/check.sh：改动文件严格 Ruff；只格式检查新文件；导入契约；api/core 局部
mypy；CPU 快速回归。69 条已有 Ruff 诊断逐项分类，不批量修复。
api/core 原有 38 个类型错误和 1 条说明完整展示；只有源字节不变的既有诊断可作为
已知基线，新增诊断或改动了对应文件即失败。这不是“类型零错误”或完整类型资格；
不使用全局类型忽略去隐藏结果。Python 3.10 的工具 TOML 解析需要 tomli，3.11+ 用 tomllib。

tests/categories.toml 对每个测试文件显式分类，不依文件是否存在选择断言。
混合 CPU/GPU 文件里的原 7 个 GPU node ID 单独标记并要求设备；原 CPU 全量仍仅 deselect
这 7 个。没有增加跳过或 xfail。历史类要求完整可达 Git 历史；安装探针明确要求
PSSOLVER_INSTALLED_PYTHON，缺失均为带 PSSOLVER_TEST_PREREQUISITE 的失败。
分类钩子不修改测试函数、收集顺序或数量。基线 5457 项按 node ID 对照，新测试另外计数。

结构指标定义固定在脚本和结果中，复现 e32ac21：83564 行、experimental 25299 行、
58 个超过 150 行的函数、1 个内部导入环。它们是趋势，不是质量或数值门槛。

## 保持项和清理

未增加生产并行路径，数值入口和数学内核的阅读路径不变；复用 S-03 successor 机制。
AGENTS 和 ARCHITECTURE 来自已确认附件；CLAUDE.md 不变。本轮未全仓格式化。
仅删除确认不存在的三个 conda 前缀登记行，备份和逐项缺失检查在新证据归档中；
既有证据目录和工作树不删除。没有推送、HPCC、GPU 或发布操作。
