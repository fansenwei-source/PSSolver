# S-16 V-55 返工与绑定勘误

`625c3ca6b2c64bb674d43430d39ec3409f78c736` 删除了 V-54 棘轮测试对三个
仓库外临时目录的实时 `exists/stat` 读取。pytest 现在只核对 v2 记录中冻结的
路径字符串、大小、文件数和清单哈希；AST 回归守卫禁止重新引入这些外部状态读取。

同一提交新增前向勘误，不改写 v2 清单：Fig. 4 的测试没有绑定脚本哈希，因此该脚本
应归入已按字节冻结的 documented exception；v0.2.0 fixture 生成器的当前哈希由
successor 56 绑定，successor 53 只绑定了旧哈希。

`V-55` 已推进为 `claimed_fixed`，等待 `R-0047` 独立复核。本 successor 不修改
运行时、打包、数值实现、公开接口、门限、skip/xfail 或生产默认值。
