# S-07 资格支持修正

第一次完整 CPU suite 暴露了两类资格支持问题，而不是运行时或科学失败：安装版专用 gate 使用 `test_` 文件名而被源码 suite 收集；五个冻结历史测试仍直接比较旧 Channel/active-force 文件哈希，没有使用已经建立的 forward-only successor 链。

`b93037a` 将安装版 gate 改为只有显式调用才收集的 `qualify_` 文件名，并让五个历史测试在不改写历史记录的前提下解析显式登记的 successor。修正后定向源码测试 32 项以及安装版 wheel gate 7 项全部通过。没有修改运行时、打包配置、阈值、skip/xfail 或生产默认值。
