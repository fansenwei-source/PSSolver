# S-18 R4 等价性 helper 共同进程根更正

successor 101 以前向方式更正最终一处证据设置：Plane profile 会从 child
进程的 cwd 解析 no-output 目录，因此分别以父/候选源码根作为 cwd 时，仅 Plane
configuration identity 会不同。两个角色现在都从共同证据目录运行，同时仍用
各自的 `PYTHONPATH`、导入来源检查、提交和 Git 身份绑定实际源码。

此前尝试中，8 个 family 的状态与 refresh 进度始终逐位相同，两个 Plane 的
runtime identity 也相同。运行时实现没有改变。本 successor 还让 successor 99
的测试通过 resolver 接受合法的后续推进，并把 successor resolver 从文件名字典序
改为从文件名解析编号的数值顺序，使编号超过 99 后仍保持正确的前向链顺序。
第一次从 JSON 顶层读取编号的尝试不能处理较早的 import 型 successor 记录，已保留
为 superseded 提交；最终实现兼容全部既有记录。
