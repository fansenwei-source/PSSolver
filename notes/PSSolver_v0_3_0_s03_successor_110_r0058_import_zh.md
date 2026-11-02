# R-0058 独立复核导入（S-18 / R5）

R-0058 PASS：V-61 推进为 verified，修复源 3c519cc 不变；资格源仍是
77a6eb1。本提交只登记证据，不重新声明运行时资格，不修改既有记录。
R5 已完成独立验收，dev/0.4 和远端暂留 d84e883；推送、临时 registry ref
清理以及下一批 R1 均等待相应流程。本轮没有远端写入或 GPU 操作。

## 证据摘要

独立故障探针 144 次运行；36 个组合与父提交及 v0.3.0 逐字段一致，
32 个进入注入路径的负对照错误接受。四路径、S-06 2160/360 行回放、
两侧各 40/40 的数值比较通过。全量 conda 5131、a2 5130 + 1 个可选
nematics3d skip、屏蔽外部路径 5131；原 7 个 CUDA deselect 不变。
Control main 对已安装候选为 298 passed / 6 skipped。本导入不重跑
数值或已安装资格，未修改的结果继续由 R-0058 绑定。

## I-1 向前勘误：缓存位置

原 v2 归档 PRUNED.txt 关于缓存全部重定向到克隆外的表述不准确。
路径级测试的 python -I 子进程会忽略 PYTHONDONTWRITEBYTECODE 与
PYTHONPYCACHEPREFIX；原 prune_inventory.json 已记录三个测试克隆各
116 个树内忽略的 .pyc。删除克隆时一并删除这些缓存。原归档不改写；
新记录绑定原 PRUNED.txt 和盘点的哈希。今后可以另批加入 -B，本次不改。

R-0057 I-1/I-2/I-3 仍有效：冻结 S-09 proxy 仅用于历史重放；S-19 如需
缓存关闭对照必须新建适用的 proxy；旧错误标签及 implicit RuntimeError
不是本次修复对象。R1-R9 聚合条目仍 deferred，下一批为 R1 影子注册表。
