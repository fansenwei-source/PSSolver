# PSSolver rc5 G-1 测量窗口修正 v9 实现记录

## 当前结论

预注册的 measurement-window 修正已经实现到 qualification-only profiler、runner 和 analyzer 中。当前仍未运行 H100，G-1 未完成，v7 失败和 v8 analysis-only 结论都没有被重分类。

## 实现边界

- 历史 profiler 的默认值仍为 `post_flush_rewarm_steps=0`；
- v9 command plan 才显式传入 `post_flush_rewarm_steps=3`；
- 3 个 rewarm steps 位于正式 timing、transform counter、compiler counter 和 steady-state memory peak 窗口之外；
- 正式 60 个 timing samples 全部保存并全部参与 CV；
- cold-recovery 显存作为独立 diagnostic 保存；
- 独立 fresh-runtime replay 使用相同 seed、配置和初态，并要求最终状态 SHA 与正式运行一致；
- replay 只用于解释剩余波动，不能进入性能或显存裁决；
- public pressure diagnostics 不可用时必须显式记录原因，不允许伪造数值；
- `nvidia-smi` 遥测只在正式计时窗口之外采集。

## 未改变内容

没有修改 PSSolver production solver/runtime、科学方程、默认值、性能阈值、显存阈值、CV 阈值、trial 数量或 outlier 规则。历史 runner/analyzer 的默认入口保持原合同。

## 下一门禁

先完成本地和 HPCC Python 3.10 的 CPU/synthetic replay，并把实际计数写回本记录。只有这些门禁通过后，才能向用户申请一次新的 H100 recovery 授权；当前没有 H100 授权。
