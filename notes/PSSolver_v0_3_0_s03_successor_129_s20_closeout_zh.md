# S-20 第一批：R-0065 导入与发布前收尾

只改记录和记录测试，不改运行时、benchmarks、jobs、打包或CHANGELOG。

## 提示分流

|来源|内容|处理|理由|涉及pssolver|
|---|---|---|---|---|
|R-0042/I-1|本机绝对路径归档测试（报告 I-1；结果 notes 同项）|已完成|D-0015 的仓库内合同和只读核对工具；R-0044 验证，V-53/V-55 后由 R-0045/R-0047 关闭。|False|
|R-0042/I-2|工作树 .pytest_cache|已完成|R-0049 第 1/7 节确认迁移工作树和 Provider 工作树包括 ignored 都为空；本轮新克隆单独修剪。|False|
|R-0042/I-3|只新建 dev/0.4，main/maint 保持（report I-3）；notes I-3 实为下项路线图|已完成|D-0014 与 A-0058 落实；保留结果 notes/report 编号差异。|False|
|R-0042/I-4|S-14 issue_ids 漏 V-47..V-49|已完成|R-0042 同次路线图更新修正；后续 R-0043 验收前已完成。|False|
|R-0043/I-1|canonical helper 成功路径构造异常对象|延后|开销可忽略且行为已资格化；任何优化改 pssolver，D-0024 冻结，留下一开发线。|True|
|R-0043/I-2|未用 hashlib/json import 及两处 import 位置|延后|R-0044 I-1、R-0047 I-3 仍在；仅外观但改 pssolver，D-0024 延后。|True|
|R-0043/I-3|p842/p843/rc432/rc430 superseded 豁免改为前向哈希|延后|可选加固，历史 successor 已绑定；未声称完成。下一测试维护批决定，不加 skip。|False|
|R-0043/I-4|successor 70 送审前原地更正历史|本批完成|如实保留历史及 successor 71 的前向绑定；本批只新增 successor，不 amend 或改写旧记录。|False|
|R-0043/I-5|models 到 linear_solvers/operators 整层放开|已完成|V-50 在 R-0044 verified；精确 stokes 两条边及不变宽棘轮。|False|
|R-0043/I-6|Control 仅 main 和每次 Slurm/H100 均需确认|已完成|D-0017/D-0020 明确；R-0049 隔离核查及 R-0065 作业授权审计通过。|False|
|R-0044/I-1|21 个未用 import、2 处 import 顺序|延后|同 R-0043 I-2，冻结 pssolver，下一开发线整理。|True|
|R-0044/I-2|diagnostics→core 收紧为 tensor_inventory→_canonical 精确边|延后|可选测试加固，当前边由既有棘轮冻结；本批不扩展架构规则范围。|False|
|R-0044/I-3|catalog 哈希与已有记录增加仓库内一致性测试|延后|既有本机 verifier 与记录绑定有效；可选额外测试另批，不声称已经实现。|False|
|R-0044/I-4|successor 73 送审前原地加绑定|本批完成|保留原文；本批新增 successor 129，所有哈希从文件和 git 取得。|False|
|R-0044/I-5|V-50..52 登记标题非逐字引用|本批完成|R-0044 已核内容一致且 verified；本批不改已验收发现标题，记录历史措辞差异。|False|
|R-0045/I-1|共享输出迁移后权限 0664→0600，内容不变|已完成|R-0046 核对 v2 记录；R-0047 保留权限勘误；不修改历史输出。|False|
|R-0045/I-2|R-0045 请求已提交后原地扩写|本批完成|保留已提交请求；本轮只创建一个新请求，不原地改写。|False|
|R-0045/I-3|3 个等价性临时目录未修剪|已完成|R-0046 核对 v2 PRUNED；R-0047 删除 pytest 中外部 exists 访问，只保留记录。|False|
|R-0045/I-4|successor 76 错误完整提交号|已完成|successor 77 前向更正，R-0045 已验证守卫失败关闭；完整号取 git。|False|
|R-0045/I-5|可选 import/精确边/catalog 测试及顶层定义空行|延后|import 涉 pssolver，其他属可选维护；同 R-0044 I-1..3。冻结 benchmarks 字节不动。|True|
|R-0046/I-1|e074ecd amend 为 5bdf1ac 未申明|已完成|R-0047 前向披露；历史不重写，本轮新提交，无 amend。|False|
|R-0046/I-2|v2 两处 binding_references 标注不准|已完成|R-0047 核对新 successor 更正，原始 v2 清单不改。|False|
|R-0046/I-3|工作树 pytest/pycache 缓存|延后|与资格无关；既有目录不在本轮删除范围。本轮仅修剪自己创建的新克隆并记录真实缓存计数。|False|
|R-0046/I-4|helper 棘轮未覆盖所有语法变体|延后|R-0046 扩大扫描确认当前无漏项；新增语法需要时再加固，不放宽现有棘轮。|False|
|R-0046/I-5|R-0044 可选整理未做|延后|同 R-0044 I-1..3；不混入记录收尾的运行时修改。|True|
|R-0047/I-1|v2 分类计数未叠加前向勘误|延后|两个文件仍按字节固定；下一次调整清单时测试勘误后分类，本批不改历史清单。|False|
|R-0047/I-2|ignored __pycache__|延后|旧工作树缓存不清理；本轮新克隆记录并修剪。|False|
|R-0047/I-3|R-0044 可选整理仍未做|延后|同 R-0044 I-1..3，涉及运行时的部分按 D-0024 延后。|True|
|R-0048/I-1|Control 历史测试依赖 git 检出，不支持 sdist 执行|已完成|R-0049 验证完整可达新克隆；R-0064/R-0065 把仅两项历史检查放在登录侧，节点不依赖 git。非承诺 sdist 支持。|False|
|R-0048/I-2|Control 迁移 worktree build/egg-info/pytest 缓存|已完成|R-0049 第 1/7 节确认清理，仅自己的迁移目录。|False|
|R-0048/I-3|Control 推送门禁允许研究分支独立推进|已完成|A-0062/A-0063 经 R-0050 审计，research 不在迁移范围；不改研究检出。|False|
|R-0049/I-1|Control rc4 历史映射固定三个祖先文件及可达性|延后|可选 Control 测试加固；本批不改 Control，后续 Consumer 同步时处理。|False|
|R-0049/I-2|保全决定 A 已选，历史记录仍 awaiting|已完成|A-0062 保全分支；R-0050 核查 A-0062/A-0063，原记录不改。|False|
|R-0049/I-3|Consumer-first 必须在推送或发布前，不仅 release 前|已完成|D-0017 生效；R-0050/R-0051 和 S-18 各批 Consumer 门禁通过。|False|
|R-0050/I-1|dir() 暴露 facade helpers/annotations|延后|V-59 修复了四包 helper 别名，R-0051 I-1 仍有 planning/annotations；余项改 pssolver，留后续。|True|
|R-0050/I-2|4 个冻结 benchmark 仍导入弃用 facade|延后|D-0016 冻结；D-0017 删除兼容层前另定四脚本迁移，不能现在改。|False|
|R-0050/I-3|7 个 CUDA 用例应沿用 deselect|已完成|R-0051 起的全量及 R-0065 资格记录沿用；本轮使用同七项 CPU deselect，不新增 skip。|False|
|R-0051/I-1|planning dir helpers、functional annotations|延后|不影响 __all__，冻结运行时；下一次相关文件修改时处理。|True|
|R-0051/I-2|4 个冻结 benchmark facade 导入|延后|同 R-0050 I-2，删除兼容层前先决定迁移。|False|
|R-0052/I-1|每批均需 Control main CPU 全量门禁|已完成|R-0054 范围设计通过，R-0055/56/58/59 各批落实。|False|
|R-0052/I-2|明确 R2 真实 NFS 复测触发条件|已完成|R-0054 范围清单和触发条件验收；R-0055 确认发布机制/调用图未变，不触发。|False|
|R-0053/I-1|合并报告 I-1：确认通知时间线及每批 Control CPU 门禁|已完成|R-0053 SUPERSEDED，无独立结论；R-0054验收门禁，successor94/R-0055更正为在Codex得知确认之前。|False|
|R-0053/I-2|合并报告 I-2：真实NFS发布路径触发条件|已完成|R-0054明确目标创建、staging/fsync/rename/清理/symlink及调用图；同 R-0052 I-2。|False|
|R-0054/I-1|D-0018 确认时间先于 R-0053，通知 Codex 在后；报告另引用每批Control门禁|已完成|successor 94 / R-0055 验证：在 Codex 得知确认之前，不是决定未确认；每批Control门禁已纳入设计。旧记录不改。|False|
|R-0054/I-2|合并报告引用的其他 I-2（非 R-0054 独立提示）|本批完成|报告复用带来的交叉编号保留于来源清单；R-0052 I-2 NFS 触发已由 R-0054 验证。|False|
|R-0055/I-1|writer 字节树增加 functional Channel/Periodic 导出|已完成|R-0056 后 2463 个文件含 196 个 functional 导出；R-0058/R-0059 复验。|False|
|R-0055/I-2|reference _read_tensor 检查顺序微调（外部不可达）|本批完成|保留 R-0055 独立结论：外部输入触发不到；不是额外运行时修复，不修改代码。|True|
|R-0056/I-1|R4 五个 batch_change 文件未改，需逐项说明|本批完成|见 r4_retained_files 五行；D-0024 (f) 明确延期，保留不是整体重构完成。|True|
|R-0056/I-2|dt 读取提前到 begin_generation 前|本批完成|dt 只读，R-0056 8×5×20 含计数逐位等价；如实记录，不回改资格化代码。|True|
|R-0057/I-1|S-09 proxy 测试依赖可达完整历史并临时替换 inspect|本批完成|85ba3ae/f94b9be 可达；非 shallow 的 --no-local --single-branch 克隆门禁保留；finally 恢复，不承诺浅克隆。|False|
|R-0057/I-2|冻结 revert proxy 不适用 R5 后，cache-off 需新设计|延后|S-19 不使用旧 proxy 做新代码资格；未来需要 cache-off 时先设计，不解冻旧脚本。|False|
|R-0057/I-3|旧报错 termination 和 implicit RuntimeError 不理想|延后|保持父提交行为为 R5 要求；改语义触及 pssolver，D-0024 后续开发线另议。|True|
|R-0058/I-1|python -I 忽略缓存环境变量，116 .pyc 与 PRUNED 措辞|本批完成|不宣称全部缓存重定向；本轮记录实际 ignored .pyc 数量并修剪本轮克隆。增加 -B 留下一测试维护批。|False|
|R-0058/I-2|R-0057 三项提示延续|本批完成|逐项沿用本表 R-0057 I-1/I-2/I-3 的分流，无重新关闭或运行时变更。|False|
|R-0058/I-3|R-0057 I-3 的交叉引用|延后|非独立新问题；旧异常语义按 R-0057 I-3 留下一开发线。|True|
|R-0059/I-1|simulation_construction docstring 仍称无 registry lookup|延后|D-0024 不允许改 psssolver；下次修改该文件再更正，不单独改运行时。|True|
|R-0059/I-2|8-family factory 比较不覆盖 6 族分派层|本批完成|不把该脚本当分派证据；分派依 R-0059 独立446例和公开路径/重启探针。|False|
|R-0059/I-3|非 __all__ 模块名增删，无仓库或 Consumer 使用|本批完成|R-0059 已确认公开 API 不变；记录边界，无兼容性修复承诺。|True|
|R-0059/I-4|中间批误写 PASS 导致步骤提前推导完成|本批完成|不改 R-0055/56/58 结果；本轮请求 INTERIM_PASS_S20_CLOSEOUT，sync 四状态演练防误导。|False|
|R-0060/I-1|949970e 缩进疏漏|已完成|4d7ec08 修正，R-0060 已核，纯测试外观。|False|
|R-0060/I-2|请求环境漏 S19_MANIFEST_SHA256|已完成|命令实际已导出；R-0061/R-0064 完整包装绑定，R-0065 实际身份通过。|False|
|R-0060/I-3|登录 module/pip 使用 /tmp 须记录与清单|已完成|A-0071/A-0075 及 R-0065 已审，按 D-0025/D-0027；本轮不访问 HPCC。|False|
|R-0060/I-4|INTERIM_PASS 后 NOW 仍 D-0009 授权旧说法|本批完成|sync 工具独立提交：D-0020 分类、D-0025 设计及 waiting；四状态脚本。|False|
|R-0060/I-5|旧 scratch 目录 mtime 与 Codex 无关|本批完成|仅记录 R-0060 观察；不推断为 S-19 写入、不清理。|False|
|R-0060/I-6|预计耗时补充 25–35 分钟，总70–80分钟|已完成|提交确认 A-0076 包含估算；R-0065 实际 1:18:21，无扩容或重试。|False|
|R-0061/I-1|跨 ssh 的提交包装须冻结完整13个export|已完成|D-0023 包装及 A-0072 审计；retry R-0064/A-0076 同样独立完整冻结。|False|
|R-0061/I-2|上传布局只读记录|已完成|A-0071/A-0075 和 R-0065 已核布局；旧预检不重放。|False|
|R-0061/I-3|deployment 替换使旧 INPUTS 失效，门禁一次性|已完成|A-0071 如实记录；R-0064 新根新清单，R-0065 11/11 验证；旧场景不改。|False|
|R-0061/I-4|scp SFTP 会 setstat artifacts/plans，前后记录|已完成|A-0071 前后记录经审计；旧归档不改。|False|
|R-0061/I-5|每份 /tmp 清单记录 hostname|已完成|后续冻结采集器/比较器具备，R-0065 已核；未来需取新基线。|False|
|R-0061/I-6|module load 的外部变化须停下报告|已完成|A-0071 正确停止，D-0023 决定后续；D-0025/D-0027 当前规则，不追溯扩权。|False|
|R-0061/I-7|test -z git remote 隐藏失败但 git_identity 失败关闭|本批完成|R-0061 已验证整体 fail-closed；冻结脚本不改，新工具无依赖该缺陷。|False|
|R-0062/I-1|CentOS7 C.UTF-8 警告，包装固定 LC_ALL=C|已完成|R-0063/R-0064 包装及 R-0065 实际运行核对。|False|
|R-0062/I-2|系统 scratch 删除触发停止|已完成|D-0026 仅一次接受；D-0027 重提删除规则；A-0079 更正数据/cache 分布，不回改旧证据。|False|
|R-0063/I-1|squeue 失败被 test -z 隐藏|已完成|R-0064 包装改 queue=…; test -z，负例 exit17 且不提交；R-0065 授权审计。|False|
|R-0063/I-2|证据表补 uname -p / fc-list 软调用|已完成|R-0064 登录/计算节点表及真实 replay；R-0065 节点资格通过。|False|
|A-0075/confirmation-baseline|提交前用冻结 PRECHECK_LOCAL 与 A-0075 outside_final，不用旧基线|已完成|A-0076 明确采用，R-0065 已核。|False|
|A-0075/gpu-cache-exception|三处 GPU 编译/字体缓存新增记录不停止|已完成|A-0076 用户明确确认；R-0065 收尾核对，未扩至其他路径/类型变化。|False|
|A-0075/queue-empty|包装要求用户队列为空|已完成|A-0076 唯一提交门禁和 R-0065 审计通过。|False|
|A-0075/inventory-count|757942 仅工具目录外条目，完整清单769273|本批完成|R-0065 已更正；不改 A-0075 报告，保留两种计数口径。|False|
|A-0079/system-deletions|1274 删除=1232 cache+28 restart+14 trajectories|本批完成|R-0065 原“全部cache”不准确；42项数据文件，系统清理与S-19无关，只记录不处理。|False|
|A-0079/temporary-registry-ref|S-19 临时登记表 ref 已按 C-1 删除|已完成|A-0079 独立核对合规；本批不重新加入或推进 dev/0.4。|False|
|A-0079/stale-now|NOW/NEXT 旧推导不能授权开始 S-20|本批完成|用户确认本设计 Q1=A；sync waiting 字段由Claude设置/清除。|False|
|A-0079/regex-warning|operations.py 两处非raw字符串 \. SyntaxWarning|延后|已封存证据不改，正则值正确；以后新脚本用raw字符串，不为此重跑。|False|
|R-0065/array-count|1993/5979只数.npy后缀；全量2134/6402含diagnostics.npy字段|本批完成|按R-0065独立统计导入；两组均有效，不把子集当全量。|False|
|R-0065/future-baseline|A-0077 外部清单已过时、HPCC现场保留|本批完成|未来HPCC步骤必须新取基线；本轮无HPCC或远端操作。|False|

## R4 五个保留文件

- `pssolver/integrator.py`：旧 v0.1 积分器保留独立 step_count/refresh_count；接入共享层会新增依赖边，D-0024 (f) 明确延期，不声称已消除重复。
- `pssolver/integrators/scalar_reference.py`：独立标量参考/对照实现有意保留；避免把 oracle 与共享被测步进逻辑耦合，进一步合并按 D-0024 (f) 延后。
- `pssolver/workflows/channel_beris_edwards.py`：工作流 wrapper 的构建/续跑协调保留，已共享的底层 runtime 步进完成 R4；wrapper 收拢按 D-0024 (f) 延后。
- `pssolver/workflows/periodic_checkpoint.py`：checkpoint/restart 进度恢复边界保留家族实现，非单纯每步刷新；避免改变已发布 restart 约定，按 D-0024 (e)(f) 延后。
- `pssolver/workflows/plane_checkpoint.py`：Plane 各格式 reader/恢复协调保留，不改变历史 checkpoint 状态；按 D-0024 (e)(f) 延后。

## 0.4.0 已知限制（供候选CHANGELOG/Release使用，本批不写发布正文）

- V-63（D-0024）：CPU-only deprecated unrolled oracle may yield nonfinite gradients at float32 fixed-work saturation. Accepted, not fixed; production/transpose/implicit are unaffected; remove alongside the facade after 0.4.
- V-41（D-0013）：Historical 0.3 R512 float32 reserved-memory exception only; it does NOT extend to 0.4. R-0065 S-19 all 16 groups have allocated/reserved ratios 1.0 against v0.3.0 and pass the unchanged 1.03 gates. Persistent cache mechanism remains; do not advertise the old exception as a new 0.4 allowance.
- A2-11（D-0007）：Checkpoint generation labels are caller-supplied, not self-proven by the checkpoint. Accepted design limitation remains.
- B8/S2-07（D-0007）：Lifting wall physics and dtype coercion remain deferred; do not reopen lifting or imply physical validation in 0.4.

## 资格导入与勘误

R-0065 PASS验证job10945436及目标7513787；V-64/V-65仅追加真实节点证据。全量原始数组2134组/6402文件，1993组/5979仅为.npy后缀子集。A-0075的757942计数只含工具目录外，完整清单769273。A-0079更正08:15Z系统删除：1232 cache +28 restart +14 trajectories。旧结果、A记录及封存归档不改。
