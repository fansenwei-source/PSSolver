# PSSolver RC4.2.1 类型化 checkpoint identity primitives

## 结论

RC4.2.1 已完成。它只建立未来 checkpoint 兼容裁决需要的类型、规范化哈希、冻结的 rc1–rc3 legacy schema registry 与独立 oracle；没有接入任何现有 reader、writer、restore 或求解器热路径。

因此这一步没有改变任何 checkpoint 的接受或拒绝结果，也没有改变公开 API、生产默认值、PSSolver-Control 或 `nematics3d`。RC4.2.0 的历史规划记录保持冻结且未改写；本文件和同名 JSON 是后续实施记录。

## 新的内部身份模型

内部模块 `pssolver/io/checkpoint_identity.py` 定义六种身份层：

- `run_provenance`：运行请求、初始化来源与输出策略；
- `forward_dynamics`：真实的下一步物理与数值映射；
- `derivative_dynamics`：functional JVP/VJP、伴随和转置语义；
- `state_layout`：持久张量含义、顺序与表示；
- `backend_restart`：exact restart 必须恢复的算法状态；
- `materialization_provenance`：实际设备和执行位置。

兼容性 aggregate 只允许 `forward_dynamics`、`derivative_dynamics`、`state_layout` 和 `backend_restart`。run/materialization provenance 无法被误放进兼容 digest。functional identity 必须包含 derivative dynamics；production identity 则明确拒绝它。

所有 payload 都先验证为有限、JSON-compatible 的数据，再按排序 key 的紧凑 JSON 规范化，并深度冻结。不同字典插入顺序得到同一 SHA-256；NaN、Inf、非字符串 key 和任意 Python 对象会 fail closed。

## 冻结的 legacy registry

registry 以五元组精确索引：

1. checkpoint family；
2. format version；
3. source release generation；
4. runtime path；
5. applicability class。

当前冻结 21 项：Plane 9 项，Periodic production、Channel production、Periodic functional、Channel functional 各 3 项，覆盖 rc1、rc2、rc3。每项都绑定完整 release commit、schema/canonicalizer ID 和从对应发布提交直接验证的 source schema SHA-256。

registry 只是目录，不会自行接受 checkpoint。所有项目的 disposition 仍是 `adjudication_required`；未知 key 直接拒绝。后续 RC4.2.2–RC4.2.4 必须提供实际 reader/migration 裁决，不能把“已登记”解释成“兼容”。

## Oracle 与隔离门禁

fixture 冻结了 registry 全量 metadata、registry digest，以及一个 Plane production identity 和一个 Channel functional identity。测试还验证：

- registry 与 payload 不可变；
- release source hash 来自 `git show <release>:<path>`，而不是当前 live run spec；
- 新模块不导入任何 `pssolver.*` live schema；
- 五条现有 checkpoint reader 源码均未接入新模块；
- `pssolver.io` 没有 re-export 新类型，公开 API 不变。

## 下一步边界

下一项是 RC4.2.2 的 Plane checkpoint identity 与 legacy upgrade path 规划/实现。当前尚未授权该实现，也未授权修改 reader/writer、原地改写 legacy checkpoint、放宽 cross-runtime restart、提交 H100、修改 PSSolver-Control、自动 merge、改变默认值或发布。
