# S-17 V-59 返工

`R-0050` 发现懒加载 package facade 丢失了 v0.3.0 普通包导入后可按属性访问的
19 个定义子模块，导致行为依赖导入顺序。提交
`223b86dde0eb22ff1816e5a91ee93195ff5718a7` 在 `functional`、`operators`、
`runtime` 和 `workflows` 的 `__getattr__` 中恢复精确子模块集合，按需导入时不发
弃用告警；facade helper 同时改用私有别名。

修复没有改变 Functional API 1.1、85 个 facade 导出、对象身份、定义模块路径、
数值算法、checkpoint schema、GPU 路径、门限、skip/xfail 或生产默认值。`V-59`
现为 `claimed_fixed`，等待 `R-0051` 独立复核；本记录不授权任何外部操作。
