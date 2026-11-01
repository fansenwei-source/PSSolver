# S-18 R4 helper 前向绑定测试更正

successor 100 更正 successor 98 的测试：旧测试同时钉住了当时的中间
`current_sha256` 和仓库当前文件，所以 successor 99 合法推进 helper 后会失败。
更正后的测试保留对历史中间值的核对，并通过 `resolve_s03_successor` 核对当前
文件。没有改写旧记录，也没有改变运行时或资格规则。
