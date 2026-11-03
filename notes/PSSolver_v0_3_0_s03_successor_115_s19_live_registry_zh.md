# S-19 当前登记表断言前向更正

successor 114 新增 V-62 后，遗漏了当前登记表测试中另一处 claimed 集合断言。
本记录将它更正为 V-62；不改历史快照、successor 114 或任何运行时、门限。

ccbfe904308db9e447815a5ecce55b61d9f2856b 的仿 HPCC CPU 全量运行是
5300 passed、1 failed、1 optional skip、7 CUDA deselected。唯一失败是该断言。
原日志、JUnit 和检出身份保留在本轮本地准备归档的
`suites/candidate_mirror_initial`，不改写为成功结果。修正后须重新完整验证。
