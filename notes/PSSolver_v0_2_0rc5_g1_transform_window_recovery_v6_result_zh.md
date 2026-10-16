# PSSolver rc5 G-1 transform measurement-window recovery v6 结果

最终分类：`STOPPED_RC5_G1_TRANSFORM_WINDOW_RECOVERY_V6_AT_FROZEN_PROFILE_PERFORMANCE_MEMORY_GATE`。

本轮没有提交 H100，只在 CPU 上回放 recovery-v5 的 96 份冻结 H100 profile。旧归档在回放前重新验证为 26/26 checksum PASS，原始 profile 未修改。

transform measurement-window 门禁已经解决并通过：96/96 raw/rate 一致；24 份 legacy/compiled Channel profile 精确满足 `663 = 11×60 + 3`，3 次调用来自计时窗口内的 scheduled spectral refresh；inverse 精确为 `2160 = 36×60`。其他 72 份没有额外调用。

继续执行未放宽的原性能和显存门禁后，发现四个失败 group：

- `plane_compiled_v2 / plane_R128`：max paired ratio `1.0563119777324725 > 1.05`；
- `channel_compiled_v2 / channel_R128x20x20`：max paired ratio `1.0584790919925313 > 1.05`；
- `periodic_functional / periodic_R128`：max paired ratio `1.066582662553552 > 1.05`；
- `channel_functional / channel_R512x40x40`：peak reserved ratio `1.0402010050251256 > 1.03`。

因此 G-1 尚未完成。没有修改阈值、求解器、生产默认值或 profile，也没有写入 `COMPLETE`。下一步应只分析这三个 paired-time outlier 和一个 reserved-memory ratio；在获得新的证据合同前不能把累计资格判为 PASS。
