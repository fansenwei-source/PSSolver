# PSSolver v0.2.x V-17 checkpoint 父目录符号链接修复

状态：完整本地 CPU 候选通过，尚未 push、未进行 installed-wheel、真实 NFS 或 H100 资格验证。

## 问题

v0.2.1 的 NFS 兼容发布路径使用
`target.parent.lstat().st_dev` 判断目标文件系统。当 checkpoint 的直接父目录
是指向另一文件系统的符号链接时，这会读取链接 inode 所在设备，而不是实际接收
staging 与最终 checkpoint 的目录设备，从而错误拒绝合法发布。

## 最小修复

保留 staging 目录的 `lstat(source).st_dev`，将目标父目录查询改为
`stat(target.parent).st_dev`。后者只跟随父目录符号链接，不解析或覆盖最终 checkpoint
名称，也不改变 no-clobber、占位目录所有权、lost-ack 回读或 fail-closed 语义。

## 冻结门禁

- Periodic functional `export_checkpoint` 经父目录符号链接成功发布并逐位恢复；
- Channel functional `export_checkpoint` 经父目录符号链接成功发布并逐位恢复；
- 公共 API 继续返回调用者请求的绝对路径，不擅自改为 resolved 路径；
- 真正不同的 resolved filesystem 继续 fail-closed；
- checkpoint bytes、schema、functional API、provider identity 与生产默认值不变；
- 不修改 PSSolver-Control 或 nematics3d。

## 本地验证结果

- 修复前定向复现：Periodic 与 Channel 两条 functional 路径 2 项失败，真实跨文件系统保护 1 项通过；
- 修复后 V-17 与 N-1 定向门禁：16 passed；
- checkpoint/functional 广泛门禁：96 passed；
- 历史证据 successor 定向门禁：144 passed；
- 完整 CPU suite：4168 passed、7 deselected、8 subtests passed、0 failed、0 skipped、0 xfailed；
- 34 个 forward-only successor source 已登记，其中 33 个是只更新解析链、不改写历史 JSON 的证据测试；
- `git diff --check` 通过。

实现提交为 `611d167d1b2c2d634222f608f4c9cd4dc671636d`，直接父提交为
`c0934ed20af3b33ad19c8872d1b3a520a1e9e245`。正式的 installed-wheel、真实 NFS
和 H100 资格状态仍为 pending。
