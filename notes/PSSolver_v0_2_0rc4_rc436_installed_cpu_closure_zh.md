# PSSolver RC4.3.6 installed-wheel lifting CPU 关闭

最终分类为 `PASS_RC4_3_6_INSTALLED_WHEEL_LIFTING_CPU_CLOSURE`。资格支持提交 `72225698005897cbc8559a17628c387f7a20b3a6` 的 clean Git archive 成功构建 wheel，并在源码 checkout 之外的新 Eva-derived venv 中以 `--no-deps` 安装；`pip check` 通过，`pssolver` 精确来自 venv purelib，没有 source shadow。资格环境没有安装或修改 `nematics3d`。

专用 source gate 与 installed-wheel gate 均为 `11 passed`，没有 failure、skip、xfail 或 collection error。它们验证了 public lifted Plane 对两种 runtime 与两种 molecular-field request 的 4/4 fail-closed 行为、受支持 homogeneous-Neumann Plane 的 4/4 requested/effective identity、localized quintic 的 fresh-only 和 interior byte identity，以及内部 lifted Plane 四步 continuous 与 2+2 文件 restart 的全部状态逐字节一致。

独立 field-neutral lifting pipeline 在 wall resolution 16、32、64 上的误差为 `5.598158649073774e-3`、`1.4008891512969812e-3`、`3.503066853420365e-4`，相邻阶数为 `1.9986095691699255` 与 `1.9996523773159034`，均高于冻结门限 1.95。这项结果不重新开放、不资格化 RC4.3.3 已明确拒绝的 public lifted Beris–Edwards 组合。

最终完整 source CPU suite 为 `3025 passed, 7 deselected, 8 subtests passed`，无 skip、failure 或 xfail。七个 deselect 精确对应既有 CUDA-only node IDs。离线资格 venv 在正式收集前曾两次因遗漏 `_pytest` 与 `py.py` 停止；补齐同一只读离线依赖源后才首次生成正式结果 JSON，这两次事件不是 solver 或 scientific failure。

控制目录为 `/home/fansenwei/pssolver_rc436_installed_cpu_7222569_20261010_v1`，checksum 为 `23/23 PASS`，`checksums.sha256` 自身 SHA-256 是 `6958b80060ff8b4be4e3a07d375a7df5f134aa2745184c2dfb5e0254964b51a5`，`COMPLETE` 已写入。

本阶段没有修改 runtime 热路径、checkpoint 接受规则、PSSolver-Control、`nematics3d`、生产默认值或远端分支，也没有提交 H100。RC4.3.6 已完成；下一阶段是 RC4.3.7 single-H100 lifting closure，需要用户单独授权。
