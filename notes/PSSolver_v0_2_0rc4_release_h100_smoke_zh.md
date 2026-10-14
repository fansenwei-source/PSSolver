# PSSolver v0.2.0rc4 installed-wheel H100 release smoke

最终分类为
`PASS_V0_2_0RC4_INSTALLED_WHEEL_SINGLE_H100_RELEASE_SMOKE_WITH_AUTHORIZED_TF32_CHILD_BOOTSTRAP_RECOVERY_V2`。

正式 release source 仍是
`658869456d762c917ca4409b091539c3733df2fb`。最终 source archive、sdist 和
wheel 都由该提交生成；本记录只是后继证据，不是新的 qualified source，也不要求重跑
H100。annotated tag 已明确指向 `6588694`，没有误指向证据登记提交。

## CPU 和 package

定向 CPU suite 为 134 passed；完整 CPU suite 为 3399 passed、一个允许的
`nematics3d` optional-dependency skip、七个 CUDA-only deselect、零 failure。最终
wheel 的 SHA-256 为
`1bf31f324180dac58777a23785fb4e1424d3cc589099cf8c6328e2f6c74d8cc5`；
隔离安装、`pip check`、无 source shadow 和 CPU 一步运行均通过。

## H100 recovery

第一次 Job `10881417` 因外部 pytest 子进程没有在自身 Python runtime 中设置
cuDNN TF32 flag 而停止；该失败归档保持原样。单独授权的 recovery Job
`10881426` 在 `gpu-h100-4-0` 上以 `COMPLETED / 0:0` 完成，用时 31 秒，没有
自动重试或第二个 recovery Job。

recovery runner 在调用 `pytest.main()` 的同一个进程内设置并持久化三个精度状态。
七个 CUDA-only tests 全部通过；最终 installed wheel 随后完成一次 Plane CUDA
timestep，Q、速度和压力均有限、形状正确，并写入运行级 `COMPLETE`。

最终归档为：

```text
/home/fansenwei/pssolver_v0_2_0rc4_release_h100_6588694_20261013_recovery_v2
```

归档 checksum 为 895/895 PASS，manifest SHA-256 为
`636e56b92170e99a3928818a2761f65e3e47a8dd9cebdf490311cd6bbaf1e9bd`。

## GitHub prerelease

annotated tag `v0.2.0rc4` 的 tag object 为 `8822ab370e252d76549c6365dc5f5023ec3231c4`，
解引用目标精确为 qualified source `658869456d762c917ca4409b091539c3733df2fb`。
GitHub prerelease 已发布：

```text
https://github.com/fansenwei-source/PSSolver/releases/tag/v0.2.0rc4
```

Release 为非 draft 的 prerelease。上传的 sdist 为 872255 bytes，wheel 为 1094307
bytes；GitHub 返回的两个资产 digest 与资格记录中的 SHA-256 精确一致，状态均为
`uploaded`。

## 发布边界

本次只创建 annotated tag 和 GitHub prerelease，没有发布 PyPI 包。生产默认值、
functional API 1.0、checkpoint 兼容范围、batch-one 能力、PSSolver-Control 和
`nematics3d` 均未改变。
