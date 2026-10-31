# S-03 successor 85：S-17 跨仓库提交身份更正

successor 83 和 84 有三个 `git_identity` 字段明确指向 PSSolver-Control，而不是
PSSolver 提交。旧的 S-08 守卫把所有完整提交号都当作 PSSolver 对象解析，导致只含
PSSolver 可达对象的新克隆失败。

本 successor 逐字段固定记录路径、字段名、完整提交号和仓库；守卫只对这三项接受
`PSSolver-Control` 身份，并要求它们不能在 PSSolver 仓库中解析。其他提交身份仍必须
解析为 PSSolver 提交或命中已有的精确前向勘误，规则没有被泛化或跳过。
