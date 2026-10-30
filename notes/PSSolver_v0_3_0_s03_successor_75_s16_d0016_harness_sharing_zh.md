# Successor 75：S-16 / D-0016 harness 共享

本 successor 前向登记提交 `152922b` 对 D-0016（方案 b）的实现。
`benchmarks/_artifact_io.py` 是只使用标准库的资格支持模块，不属于
`pssolver/` 运行时或公开 API。

两个仍在使用的 S-16 verifier 改用共享的原子 JSON/文本写出、文件哈希
能力；相同输入下改前与改后的输出逐字节相同。保守 AST 清单逐项冻结或
说明了其余旧 helper，棘轮禁止新脚本再定义本地原子写或 provenance
helper。已哈希绑定的历史 harness 未修改，S-09/S-19 冻结 harness 未修改。

本提交不修改数值、运行时、GPU 专用路径、门限、skip/xfail、checkpoint
schema、provider identity 或生产默认值，不授权 push 或任何外部计算。
