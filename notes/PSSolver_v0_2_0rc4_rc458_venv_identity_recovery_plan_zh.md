# PSSolver v0.2.0rc4 RC4.5.8 installed-venv identity recovery v2

## 结论

本轮只准备 qualification-only recovery，不提交 H100 Job。Job 10879343 的失败来自外部 runner 对 venv 解释器路径的错误归一化，不是 PSSolver 数值实现、CUDA、性能或科学失败。

## 根因

`venv/bin/python` 通常是指向基础解释器的符号链接。失败 runner 在解析参数时调用了 `Path.resolve()`，随后把解析后的基础 Conda 解释器父目录当作 venv 根目录。实际 `pssolver` 已从 `/scratch1/.../venv/lib/python3.10/site-packages` 导入，但错误门禁拿它与 `/home/fansenwei/miniconda3/envs/Nematics3D` 比较，因此在任何 CUDA-only test 或 profile 开始前错误拒绝。

## 修正语义

修正后的门禁区分两种身份：

1. lexical invocation path：保留调用时的 `.../venv/bin/python`，用于确定被请求的 venv；
2. resolved interpreter target：单独记录符号链接最终指向的基础解释器，只作为 provenance，不用于确定 venv 所有权。

门禁还必须同时验证：

- venv 根目录存在普通 `pyvenv.cfg`；
- 运行时 `sys.prefix` 与 lexical venv 根目录一致；
- `sysconfig` 给出的 `purelib` 位于该 venv 内；
- `pssolver` 或 `pssolver_control` 的实际导入文件位于该 `purelib` 内；
- source checkout 不能遮蔽 installed wheel。

候选 Provider、Consumer venv 中的 Provider 和 Consumer 导入都采用同一规则。任何路径或归属不一致都 fail closed。

同样的 lexical-path 规则也应用于 96-cell profile matrix 的 A/B 子进程。为保持失败尝试所绑定的原 matrix runner 哈希不变，本轮新增 recovery-only matrix 入口，而不改写旧文件。新入口仍会解析最终目标以确认解释器文件存在，但写入命令计划和实际启动子进程时保留 A/B 各自的 `venv/bin/python`，避免二者退化为同一个基础解释器。

## 冻结边界

- A：`51f47e9f06bd3e019d785305bff47c80f5519b10`
- B：`a195bd05992bc6553c72c400587d2b07afe722d0`
- 失败 qualification commit：`5c286f8671151d43bb05674984a0f1de94e431b8`
- Consumer qualified source：`b8539e0ec4c137a6c13746f6832e26e74be24515`
- Consumer evidence tip：`0e3895d664d1b81925127848b2dfde638f39763d`
- 旧归档 checksum：4443/4443 PASS；manifest SHA-256 为 `c31f1ccc06ef815c582d8cd0f784aad2864e3569d710cfa10e8af337e9c3595b`。

旧控制目录和 scratch 保持只读。下一次只能使用新的 `recovery_v2` 控制目录和 scratch。若复用三个 wheel，必须先重新验证精确 SHA-256。

## 无 GPU 预检

已在 HPCC 登录节点使用现有 installed-wheel venv 实测新门禁：candidate venv 的 Provider 导入、Consumer venv 的 Provider 导入和 Consumer 导入共 3/3 PASS。三者都保留了 `/scratch1/.../venv/bin/python` 的 lexical 身份，同时单独记录解析目标 `/home/fansenwei/miniconda3/envs/Nematics3D/bin/python3.10`；运行时 prefix、purelib 和实际导入文件的包含关系全部正确。该预检没有申请 GPU，也没有调用 `sbatch` 或 `srun`。

## 授权状态

当前没有新的 push 或 H100 recovery submission 授权。不会调用 `sbatch`，不会自动重试，不会修改求解器数值实现、阈值、checkpoint schema、Control 算法、nematics3d 或生产默认值。
