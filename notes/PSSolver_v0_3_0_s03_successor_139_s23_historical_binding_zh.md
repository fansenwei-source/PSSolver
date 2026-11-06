# S-23：RC4.4.5 测试来源的前向解析

da33e4f 的两轮全量均为 5650 passed、1 failed；4 个 teardown 错误已消失。
剩余失败是 RC4.4.5 记录测试只读取固定的旧后继列表，不能识别 138 对
test_phase9_p974_channel_functional_runtime.py 的测试收尾修正。

保留旧列表和逐段精确匹配，然后调用既有 resolve_s03_successor 解析
后续显式登记。未登记的哈希仍原样返回，最终必须等于实际文件 SHA；
新增反例证明不能用任意旧哈希通过。不改 resolver、历史记录或原始哈希。
本记录向前绑定这两个测试，137/138 与更早的记录保持原样。

对应 ARCHITECTURE §7；只改 notes/tests，不改资格工具、基准、运行时、
依赖、wheel、门限、skip/xfail。保留这两次失败证据，新提交的新克隆重跑。
