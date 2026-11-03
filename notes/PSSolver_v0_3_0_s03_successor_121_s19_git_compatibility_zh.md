# S-19 Git 1.8.3.1 兼容性后继登记

A-0070 的上传与全部输入哈希检查通过，冻结部署第 13 行因 HPCC 的
`/usr/bin/git` 1.8.3.1 不支持 `-C` 而停止；尚未创建 venv 或运行登录预检。
本记录不改写 R-0060、A-0069、A-0070 或 S-16 的已绑定记录。

私有支持模块 `_artifact_io._git_output` 仅将 `git -C` 改为
`subprocess.run(["git", ...], cwd=repository, check=True, ...)`。
身份字段与有效仓库上的输出保持不变。缺失目录和非目录分别抛
`FileNotFoundError`、`NotADirectoryError`，无效仓库抛 `CalledProcessError`；
三种情况均失败关闭，验证器不得继续到 venv 门禁或写出成功记录。

未来部署第 13 行由新命令清单改为子 shell 的 `cd && git remote remove origin`。
support 包和输入清单另命名，旧 support 树及 tar 保留；只有部署清单在先备份、
核对哈希之后替换。所有具体路径与新哈希由新准备关证据冻结，未经准备关验收
不得恢复 HPCC 写入。H100 提交仍须用户逐次确认。

运行时、三个 wheel、Provider tar、Consumer bundle、科学和性能合同、
阶段顺序、阈值、D-0022 精确例外、h100_plan、mirror_versions 和作业脚本不变。
本批仅为本地 Git 兼容修复、测试和证据，不是 S-19 最终资格通过。
