# S-19 登录/计算节点命令边界修正

依据 R-0063：nvidia-smi 仅在真实作业模式检查，位于 --preflight-only 退出之后、node_root 创建之前；缺失仍 exit 2。共享块其余检查逐字节保留。

本轮 jobs/tests/notes-only；benchmarks、运行时、Control、矩阵、门限及 wheel 不变。V-64 保持 claimed_fixed；V-65 按导入要求保持 open，待准备关复核。

一次登录节点只读 command-v 探查已归档：未 module load、未写入。命令表区分登录/计算节点，并补充 uname -p 与 fc-list 的软调用。新冻结提交包装采用 queue="$(squeue -h -u "$USER")"; test -z "$queue"，squeue 非零必须失败关闭。

旧 frozen_0620103_v2 保留且被新冻结取代，不改写旧文件。新旧完整 CPU 演练不重跑的前提是 benchmarks/ 与运行时、打包均无差异；沿用 R-0063 已验证的 6675 个数组逐字节结果，不能把本轮称为 GPU 资格。真实旧工具部署重放必须调用未改写的 bash 作业脚本 --preflight-only，禁止 ready adapter。
