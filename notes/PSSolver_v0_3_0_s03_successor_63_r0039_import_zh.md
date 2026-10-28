# R-0039：v0.3.0rc1 独立发布后验收导入

R-0039 的独立结论为 PASS。GitHub Release `427202656`、annotated tag
`v0.3.0rc1`（tag object `3fb93502…`，release source `c031cda…`）、两个线上
资产以及远端 `dev/0.3` / `release/pssolver-v0.3.0rc1` 身份均与授权一致。
没有发布阻断项、登记表更新或新发现；v0.3.0rc1 保持原样，S-11 完成。

首次发布目录
`/home/fansenwei/pssolver_v0_3_0rc1_publication_c031cda_20261028_v1`
中的 `release_notes.md` 是 GitHub Release 正文的原始来源，SHA-256 为
`3ec58dc33449677f20857ac4c2c11c097703a3916e40ec586ffaac7eda7e436c`。
该目录没有封印，并且保留了首次请求不受支持字段后产生的空输出；因此只把它作为
正文来源的 provenance 原样保留，不把它改写或提升为发布成功证据。成功结论由
R-0039 的独立归档以及 A-0048/A-0049 记录链提供。

本提交只新增记录和绑定测试，不修改运行时、打包输入、阈值、skip/xfail、tag、
GitHub Release 或线上资产，也不授权 push。后续工作转入 S-12：冻结 final
v0.3.0 发布源、生成本地资产并提交独立候选验收请求。
