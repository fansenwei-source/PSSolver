# PSSolver v0.2.0rc4 RC4.5.8 transform-counter layer recovery

## 状态

`QUALIFICATION_ONLY_RECOVERY_READY_FOR_CPU_REPLAY`

本恢复只修正 RC4.5.8 外部 analyzer 对 transform counter 层级的解释，不修改
PSSolver runtime、积分器、谱变换、模型、边界条件、性能门限、生产默认值或已有 H100
profile。

Job `10879606` 已经完成 96/96 个 H100 profiles。Plane、Periodic、
complete-stress 和 functional 的 72 个 profiles 与冻结计数完全一致。其余 24 个
legacy/compiled Channel profiles 在 baseline/candidate、两个网格和三个 trials 上均为：

```text
timed steps                         60
base forward calls        11 * 60 = 660
scheduled refresh calls              3
raw forward calls                   663
raw forward calls/step            11.05
inverse calls             36 * 60 = 2160
```

三次 refresh 来自生产 interval=20，在 10 个 warmup 之后的 60-step 测量窗口内发生
于 completed steps 20、40 和 60。该语义已由 P7.6 和 RC4.4.7 资格合同冻结。

## 恢复方法

新增 successor analyzer：

`benchmarks/analyze_rc458_transform_counter_recovery.py`

它保持原 96 个 profile 文件只读，并逐份执行：

1. 验证 raw total 与 raw-per-step 自洽；
2. 对 legacy/compiled Channel 精确验证 interval、warmup、timed steps、三次 refresh、
   一个 dynamic transform group、raw `663/2160`；
3. 计算但不写回 base forward `660/60 = 11`；
4. 对其余六个 family 禁止扣除 refresh；
5. 在临时目录建立 analyzer-only JSON 副本；
6. 复用原 analyzer 的 identity、finite、fallback、graph break、state SHA、performance、
   memory、Consumer 与 CUDA 门禁；
7. 原 profile、旧失败归档和原 analyzer 均不修改。

该 CPU-only replay 不重新运行 H100 timestep，不能自行伪造新 H100 证据。只有冻结
profile 的 checksum、原 v2 归档和 successor analyzer 全部通过后，才可形成最终累计
资格结论。

## 授权边界

- 本提交不修改 PSSolver 数值实现；
- 不修改 PSSolver-Control 或 nematics3d；
- 不改变性能、显存、状态 SHA 或 transform 门限；
- 不改变生产默认值；
- 不授权新的 H100 submission；
- 不授权自动 merge 或默认提升。
