# PSSolver v0.2.0rc5 R5-11：Periodic 描述元数据与 strict fresh-state 合同

## N6 结论

Periodic 投影描述现在与 checkpoint forward-dynamics identity 使用同一权威映射：

- `full_complex`：`complete_conjugate_pair_projection_each_step.v2`；
- `hermitian_half`：`self_conjugate_stored_planes_projection_each_step.v2`。

projector execution metadata、production runtime metadata、functional runtime identity 和 checkpoint identity 不再各自维护字符串。rc1–rc3 已发布 checkpoint 的旧字面量 `self_conjugate_planes_each_step` 在 legacy canonicalizer 中显式冻结；带旧描述的当前 v3 functional checkpoint identity 会在比较前按 spectral storage 归一化。因此本次更正描述，不更改实际投影算法，也不让旧 checkpoint reader 再次失效。

## B20 结论

R5-11 选择合并报告允许的“修订合同与记录”，没有制造一个没有数学依据的拒绝规则。

lifted Plane 接口接收的是 cell-centered 内部配点值，不包含墙面样本，也不包含用户独立连续初态的 wall trace。对这些内部值减去 affine lift 后，任意有限 remainder 数组都定义一个齐次 DST 插值；该插值的数学墙面值为零。仅凭内部数组无法唯一反推出用户在墙面“原本想要”的连续值。若强行使用一阶或二阶有限差分外推，会把未注册的数值方法引入 fresh construction，并可能错误拒绝合法的谱初态。

因此 `strict_no_automatic_conditioning_v1` 的有效合同明确为：验证 shape、dtype、device 和有限性，执行 `r_conditioned = r_raw`，逐字节保留请求的内部配点值，不自动 taper，不在 restart 时重新调理，也不臆造 wall trace。若未来公共 API 增加显式 wall-trace 声明，可以在新的版本合同下验证该 trace；不能把这个新信息假装成当前数组已经携带的数据。

RC4.3.2 历史文件保持不变。本记录是前向 successor，明确覆盖其中“从内部样本推断并拒绝不兼容连续 wall trace”的不可实现表述。`localized_quintic_wall_transition_v1` 和 legacy sine 仍然只能显式选择。

## 范围

本阶段不修改科学方程、投影算法、checkpoint tensor schema、生产默认值、PSSolver-Control 或 nematics3d；不需要 H100，也没有 push。

## CPU 关闭证据

- 历史记录与计划门禁：206 passed；
- 完整源码 suite：3586 passed，7 skipped，8 subtests passed，0 failed；
- fresh installed-wheel 定向门禁：143 passed，0 skipped，0 failed；
- fresh installed-wheel 适用完整 suite：3579 passed，7 skipped，8 subtests passed，0 failed；
- installed 完整 suite 只排除了 7 个必须审计本地源码布局的 `tests/test_phase8_p861_local_cumulative_audit.py` 测试，没有新增 skip、xfail 或 deselect；
- wheel 从 qualified source commit `489655b260339b55e9b6ebdbb161dd464af0af18` 的 clean Git archive 构建，导入来自 fresh venv 的 `site-packages`，没有 source shadow import；
- `pip check` 通过。为满足继承的 `nematics3d` metadata，只在该隔离 venv 中安装 NumPy 2.3.2；没有修改全局环境或 nematics3d。

最终分类为 `PASS_R5_11_PERIODIC_METADATA_AND_STRICT_FRESH_STATE_CONTRACT_CPU_CLOSURE`。证据目录为 `/home/fansenwei/pssolver_rc5_r511_installed_cpu_489655b_20261015_v1`，manifest 为 9/9 PASS，`checksums.sha256` 自身 SHA-256 为 `e75ff2aae81823a03300d813b0cb292ef94552579e3536554130fd2a95d33542`。
