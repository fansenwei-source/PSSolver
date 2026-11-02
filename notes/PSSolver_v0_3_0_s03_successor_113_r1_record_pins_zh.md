# R1 全量门禁：历史源码绑定的向前修正

ee9ec8b 的三套新克隆全量各有两处记录测试失败：P8.4.4 与 RC4.3.2 已接入 successor 72，但把它的 current_sha256 当作终值，没有沿后续 successor 112 解析 simulation_binding.py。conda/blocked 为 5275 passed、2 failed；a2 为 5274 passed、1 个原有可选 skip、2 failed。checkpoint writer 子集中的 P8.4.4 同样失败。所有原始日志/JUnit 和已通过的两组数值逐位比较保存在 v1 失败归档，不作为最终门禁。

只修改这两条断言：保留 successor 72 的精确 historical_sha256 校验，将 current_sha256 输入现有 resolve_s03_successor。没有新增豁免条目，没有跳过检查；链中没有精确前驱时仍不接受新哈希。历史记录保持原样，生产与 benchmarks 字节不变。本提交重新绑定这两个测试，随后在 v2 中重跑全量与资格门禁；不得把 v1 的失败改写成成功。
