# S-16 R8 本机证据核对迁移对照表

本表执行 D-0015（选项 C，加必须运行的本机核对工具）。历史记录和绝对路径不改写；pytest 只核对仓库内冻结合同，本机文件由 `benchmarks/verify_v0_4_local_evidence.py` 只读核对。

- 迁移测试：35 个（27 个文件）
- 本机检查项：103 个
- 语义复算：`S09_STOP_ANALYSIS`
- 缺失、大小变化、SHA-256 变化、清单条目变化、COMPLETE 缺失或字段来源变化均失败关闭。

## pytest → verifier

| pytest 节点 | verifier 检查项 | 语义复算 |
|---|---|---|
| `tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::test_s09_stop_analysis_does_not_overclaim_unique_cause` | `C005`, `C023`, `C078`, `C079`, `C080`, `C081`, `C087`, `C088`, `C089` | `S09_STOP_ANALYSIS` |
| `tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::test_s09_stop_analysis_preserves_failure_and_uses_no_new_profiles` | `C005`, `C023`, `C078`, `C079`, `C080`, `C081`, `C087`, `C088`, `C089` | `S09_STOP_ANALYSIS` |
| `tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::test_s09_stop_analysis_separates_transient_from_iteration_count` | `C005`, `C023`, `C078`, `C079`, `C080`, `C081`, `C087`, `C088`, `C089` | `S09_STOP_ANALYSIS` |
| `tests/test_rc5_r56_public_error_boundary_record.py::test_r56_record_binds_installed_wheel_identity_and_cpu_results` | `C005`, `C023`, `C024`, `C025`, `C026` |  |
| `tests/test_rc5_r57_device_normalization_record.py::test_r57_record_binds_installed_wheel_artifacts_and_identity` | `C005`, `C023`, `C027`, `C028`, `C029`, `C030` |  |
| `tests/test_v0_3_s08_r0021_import.py::test_r0021_result_and_independent_archive_are_exact` | `C005`, `C006`, `C023` |  |
| `tests/test_v0_3_s09_cpu_6x60_preregistration.py::test_s09_6x60_contract_preserves_the_failed_5x30_sample` | `C005`, `C023`, `C087`, `C088` |  |
| `tests/test_v0_3_s09_cpu_6x60_stop_record.py::test_s09_6x60_archive_and_matrices_are_content_addressed` | `C005`, `C023`, `C078`, `C079`, `C080`, `C081` |  |
| `tests/test_v0_3_s09_cpu_6x60_stop_record.py::test_s09_6x60_sources_and_historical_failure_remain_bound` | `C005`, `C023`, `C087`, `C088` |  |
| `tests/test_v0_3_s09_cpu_diagnosis_record.py::test_s09_evidence_binds_every_measured_matrix` | `C005`, `C023`, `C087`, `C088`, `C089`, `C090`, `C091`, `C092`, `C093`, `C094` |  |
| `tests/test_v0_3_s09_d0013_closure.py::test_closure_adjudication_changes_no_analyzer_threshold_or_scope` | `C005`, `C023`, `C074`, `C075`, `C077` |  |
| `tests/test_v0_3_s09_d0013_closure.py::test_d0013_and_independent_reviews_are_exactly_bound` | `C004`, `C005`, `C016`, `C017`, `C018`, `C019`, `C023` |  |
| `tests/test_v0_3_s09_d0013_closure.py::test_noninterrupting_recompute_proves_the_exception_is_exhaustive` | `C005`, `C023`, `C074`, `C076`, `C077` |  |
| `tests/test_v0_3_s09_d0013_closure.py::test_v1_archive_is_preserved_while_v2_binds_the_real_job_plan` | `C005`, `C023`, `C072`, `C073` |  |
| `tests/test_v0_3_s09_failure_analysis_record.py::test_s09_failure_analysis_archive_is_content_addressed` | `C005`, `C023`, `C082`, `C083`, `C084`, `C085`, `C086` |  |
| `tests/test_v0_3_s09_r0022_import.py::test_r0022_import_binds_the_rework_result_and_archive` | `C005`, `C007`, `C023` |  |
| `tests/test_v0_3_s09_r0023_import.py::test_r0023_import_binds_the_exact_review_result` | `C005`, `C008`, `C023` |  |
| `tests/test_v0_3_s09_r0024_import.py::test_r0024_import_binds_the_exact_passing_result_and_archive` | `C005`, `C009`, `C023` |  |
| `tests/test_v0_3_s09_r0025_import.py::test_r0025_result_and_archive_are_bound` | `C005`, `C010`, `C023`, `C041`, `C042` |  |
| `tests/test_v0_3_s09_r0026_candidate.py::test_r0026_candidate_binds_final_archive_plan_job_and_rehearsal` | `C001`, `C002`, `C005`, `C023` |  |
| `tests/test_v0_3_s09_r0026_import.py::test_r0026_h100_citations_are_exact` | `C005`, `C011`, `C023` |  |
| `tests/test_v0_3_s09_r0026_import.py::test_r0026_result_and_archive_are_bound` | `C005`, `C011`, `C023`, `C045`, `C046` |  |
| `tests/test_v0_3_s09_r0027_import.py::test_r0027_result_archive_and_decision_are_bound` | `C003`, `C005`, `C012`, `C023`, `C043`, `C044` |  |
| `tests/test_v0_3_s09_r0028_rework.py::test_r0028_rework_result_and_archive_are_bound` | `C005`, `C013`, `C023`, `C035`, `C036` |  |
| `tests/test_v0_3_s09_r0029_candidate.py::test_r0029_local_archive_and_cpu_claims_are_exact` | `C005`, `C023`, `C059`, `C060`, `C061`, `C062`, `C063` |  |
| `tests/test_v0_3_s09_r0029_rework.py::test_r0029_result_and_independent_archive_are_bound` | `C005`, `C014`, `C023`, `C039`, `C040` |  |
| `tests/test_v0_3_s09_r0030_candidate.py::test_r0030_artifacts_and_archive_are_exact` | `C005`, `C023`, `C064`, `C065`, `C066`, `C067`, `C068`, `C069`, `C070`, `C071` |  |
| `tests/test_v0_3_s09_r0030_import.py::test_r0030_citable_identities_match_the_passing_result` | `C005`, `C015`, `C023` |  |
| `tests/test_v0_3_s09_r0030_import.py::test_r0030_result_and_independent_archive_are_bound` | `C005`, `C015`, `C023`, `C037`, `C038` |  |
| `tests/test_v0_3_s09_r0035_import.py::test_r0035_result_and_independent_archive_are_bound` | `C005`, `C020`, `C023`, `C031`, `C032`, `C033`, `C034` |  |
| `tests/test_v0_3_s09_transient_telemetry_record.py::test_transient_telemetry_archive_and_sources_are_content_addressed` | `C005`, `C023`, `C095`, `C096` |  |
| `tests/test_v0_3_s10_r0033_import.py::test_r0033_import_binds_the_exact_independent_review` | `C005`, `C018`, `C023`, `C055`, `C056`, `C057`, `C058` |  |
| `tests/test_v0_3_s10_r0036_rework.py::test_r0036_rework_binds_the_exact_independent_review` | `C005`, `C021`, `C023`, `C047`, `C048`, `C049`, `C050` |  |
| `tests/test_v0_3_s10_r0037_import.py::test_r0037_import_binds_the_exact_independent_pass` | `C005`, `C022`, `C023`, `C051`, `C052`, `C053`, `C054` |  |
| `tests/test_v0_4_s13_s14_correction_source.py::test_s13_archive_is_complete_and_content_addressed` | `C005`, `C023`, `C097`, `C098`, `C099`, `C100`, `C101`, `C102`, `C103` |  |

## verifier 检查项

| ID | 类型 | 历史路径 | 字节 | SHA-256 |
|---|---|---|---:|---|
| `C001` | `file` | `/home/fansenwei/.pssolver_s09_r0026_local/provider_support_719ae76.tar` | 21463040 | `7500e889a01733d998ae08c121f12de00bf1d5a831f9eefdcc14fd763843b3cd` |
| `C002` | `file` | `/home/fansenwei/.pssolver_s09_r0026_local/rehearsal_v3/cpu_rehearsal.json` | 12355 | `a1bcf41e50dc1064e0772baadd048119ec4a4515f8c106c27914a43ca8353d2c` |
| `C003` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/decisions/D-0012.json` | 1939 | `fabffcace193d72dabc322b22e05aecbb7489fb3a5da833246a228b4e86d1936` |
| `C004` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/decisions/D-0013.json` | 3356 | `6ecd1b92a19fae5c0e66895b9b3216996a2a58d1c6d37fe8358331c8a6fd6f1f` |
| `C005` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/decisions/D-0015.json` | 3429 | `69eaded6bff3a9ddcea173f1924f3d272199e00893b8fa483936bd3be3af56b6` |
| `C006` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0021.result.json` | 7444 | `5f46e5991639f9c014ccff3cf8c43dabad8dcf59b426ec68dbd2b0add30c98f5` |
| `C007` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0022.result.json` | 15574 | `9522b71a0b944319b377a205d528546b3bed7589230cee7eb06bde3ea2b3901a` |
| `C008` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0023.result.json` | 11586 | `4570f85423eb0c6008eeaa141889afcb78e751f9cad003f887c222ae121ca3fc` |
| `C009` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0024.result.json` | 12682 | `2b172299be82ac1141badd4bbf099b8865a467926122cabb2a330d564037f968` |
| `C010` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0025.result.json` | 11940 | `fffe1689735b9a197bb0c8703fcfbfac3a1426c9a180cc97e2de7e6803e620f5` |
| `C011` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0026.result.json` | 11197 | `9bc8bfa136c96e217ec731b812a512c96c72e76e4e0f163caa7b217c9cc34732` |
| `C012` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0027.result.json` | 10557 | `abf7ec135a9ab5b91c71929d3ab89049e254149fba16ee9ba66e62ac6ef01dbd` |
| `C013` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0028.result.json` | 6953 | `de2a2ea63f3ed2dbda06f70d580372af4a513b4e6e1efe77916aec525bb27bd3` |
| `C014` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0029.result.json` | 9665 | `58d21c9e27fb1680b952a7171bf8eeb5884c7b6e3b6a37da67223030689ec2de` |
| `C015` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0030.result.json` | 12401 | `6685b4cfede76b5d36ea48fbdcc05e6ae6f1d6673c84749f5cb547e5b6a8474d` |
| `C016` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0031.result.json` | 12036 | `68135c2c6014a3b25e2e04bd43e78e4244e8ecf2b8f8ff0d997a9cb63705bff8` |
| `C017` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0032.result.json` | 10140 | `ce87ee51b3f423f6fb3cb1ddb56f7b09dca96dd3d608bf797e50bd58f9e5aa4e` |
| `C018` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0033.result.json` | 8499 | `8b0f709dc37c1bb631b4efb9f82f47353fc2e06485e7b6c6ee7613536988bd49` |
| `C019` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0034.result.json` | 9485 | `0d242ae40dc8a0d5f5a67214995654d47ae4dfeb184fecfd25c0a7db45b57d30` |
| `C020` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0035.result.json` | 9536 | `f6f94781b4681b7ed1ee2ddf33cfc259aa220666372c3f082ece32da49043c3c` |
| `C021` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0036.result.json` | 10311 | `d7572772516ba1d6adbe0e34359cb1a8d9778425cdac47dcf75977964288580e` |
| `C022` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0037.result.json` | 10016 | `878feb76d315a9292ebbb1883a3c4699771ffa354d585a444c0ac2b7bfa57c95` |
| `C023` | `file` | `/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0043.result.json` | 14210 | `c2524ea963cb4c6d6a91d80e197e16dc8899613e42c01bb66e1b3468b90671ef` |
| `C024` | `manifest` | `/home/fansenwei/pssolver_rc5_r56_installed_cpu_cf572bc_20261014_v1/checksums.sha256` | 1112 | `a8bcc1dd3d52c0c27cd3a86d8ea499c09206a3229880451ba97e15c7b24ff084` |
| `C025` | `file` | `/home/fansenwei/pssolver_rc5_r56_installed_cpu_cf572bc_20261014_v1/package/dist/pssolver-0.2.0rc4-py3-none-any.whl` | 787958 | `d427a3a2cf8aa7a5b92f2018f4d5a28e88eaf0ac23d1336e58bccc1591549666` |
| `C026` | `file` | `/home/fansenwei/pssolver_rc5_r56_installed_cpu_cf572bc_20261014_v1/package/provider_cf572bc.tar` | 21821440 | `c22b08e6a024245c55a7fb5439599596ad29d6261059eb393ce83691c5fb623e` |
| `C027` | `manifest` | `/home/fansenwei/pssolver_rc5_r57_installed_cpu_2d67ba2_20261014_v1/checksums.sha256` | 453 | `17def36096bca45194e35b1d47ef132874dde85a66e5a25f32251777bba79c6f` |
| `C028` | `file` | `/home/fansenwei/pssolver_rc5_r57_installed_cpu_2d67ba2_20261014_v1/package/dist/pssolver-0.2.0rc4-py3-none-any.whl` | 788670 | `e7f0515bae2a2d25e36f38e07ef39a8398b1db57d55c6501f8cdd1c5000fbc65` |
| `C029` | `file` | `/home/fansenwei/pssolver_rc5_r57_installed_cpu_2d67ba2_20261014_v1/package/provider_2d67ba2.tar` | 21862400 | `9ebe1e41f457879cd78400b39ff8d09f5afd6e340e01888cb69d50c49261c403` |
| `C030` | `file` | `/home/fansenwei/pssolver_rc5_r57_installed_cpu_2d67ba2_20261014_v1/venv/lib/python3.12/site-packages/pssolver/__init__.py` | 3837 | `cc70ae9bd8667936f7b5c78f17f1a718341b1d9f8d4b63f649555380f01a5180` |
| `C031` | `complete` | `/home/fansenwei/pssolver_s09closure_rw_claude_review_20261028_v1/COMPLETE` | 91 | `a2ad0543eff6f83d2347f877637e0132d892c5ee7a75aadab5107c60e3a5c651` |
| `C032` | `manifest` | `/home/fansenwei/pssolver_s09closure_rw_claude_review_20261028_v1/checksums.sha256` | 4717 | `998317f64ac3aa184ed112b740a838a51acaf454e34848c88c13c86828317327` |
| `C033` | `file` | `/home/fansenwei/pssolver_s09closure_rw_claude_review_20261028_v1/final_report.md` | 9749 | `baa1421b9bafeee9503a3e50d307a6cfc4169da872c652b1052c5d29f2975207` |
| `C034` | `file` | `/home/fansenwei/pssolver_s09closure_rw_claude_review_20261028_v1/summary.json` | 1297 | `a62b2680859f85692551ca561483727d5173fdfbbcd3c2723998d5404e1b033b` |
| `C035` | `complete` | `/home/fansenwei/pssolver_s09diag_claude_review_20261026_v1/COMPLETE` | 136 | `9703db443343ac9515f2265cfaa76ff48e07d3ca3acfa54b0dfc08acf28039f8` |
| `C036` | `manifest` | `/home/fansenwei/pssolver_s09diag_claude_review_20261026_v1/checksums.sha256` | 163446 | `cebf6521dd8102c93b24a1eee7a8ac1685bd452831c4afaf786b895023a85a5f` |
| `C037` | `complete` | `/home/fansenwei/pssolver_s09diagrw2_claude_review_20261026_v1/COMPLETE` | 169 | `acce1f35abc6159d0c48cac5c224bc71f03dbc65597e3502af44dc054e406efe` |
| `C038` | `manifest` | `/home/fansenwei/pssolver_s09diagrw2_claude_review_20261026_v1/checksums.sha256` | 53751 | `78b2192d2b866ac0e7acb875f9be364deccfb09c34bb79d031788485eca25bb6` |
| `C039` | `complete` | `/home/fansenwei/pssolver_s09diagrw_claude_review_20261026_v1/COMPLETE` | 128 | `c294a6fcae74a6e5e4656354ac9a192c2ac8fb7421816d2185c25aeab20214ce` |
| `C040` | `manifest` | `/home/fansenwei/pssolver_s09diagrw_claude_review_20261026_v1/checksums.sha256` | 290161 | `dc69004bd8160efeef035b5cefe27f3029845b0c111710f62b9c1b10d86eb52f` |
| `C041` | `complete` | `/home/fansenwei/pssolver_s09h100_claude_review_20261026_v1/COMPLETE` | 205 | `dbf4bfb1adf77fde0ada765ddb77e5fd4b2b273927d40577d5cdd1f5d3d5ff3b` |
| `C042` | `manifest` | `/home/fansenwei/pssolver_s09h100_claude_review_20261026_v1/checksums.sha256` | 762956 | `9a2fed5fcc2153d0f77d4910341695ed134000670f02a34b13c20fbbfbe11413` |
| `C043` | `complete` | `/home/fansenwei/pssolver_s09h100rv2_claude_review_20261026_v1/COMPLETE` | 136 | `6218cda8d30082419c50929f7c1716c481d0176600756cb7f1641dd8449ab4e8` |
| `C044` | `manifest` | `/home/fansenwei/pssolver_s09h100rv2_claude_review_20261026_v1/checksums.sha256` | 524665 | `81081de4155932d5adcb54c585fa2ec0de5060330c98cc785cf5979096ec1819` |
| `C045` | `complete` | `/home/fansenwei/pssolver_s09recov_claude_review_20261026_v1/COMPLETE` | 194 | `9d615b361e094dbecf9acc4e1c8008a23de691556f6c3b5ec4965d426d1e7e74` |
| `C046` | `manifest` | `/home/fansenwei/pssolver_s09recov_claude_review_20261026_v1/checksums.sha256` | 1077783 | `495b2978649b6dcb158992ca0769e95385cbde5bcfb95281f2b7fab52bd062c8` |
| `C047` | `complete` | `/home/fansenwei/pssolver_s10integ_claude_review_20261028_v1/COMPLETE` | 120 | `2ef85eb3e702d6e908fe9a90e7d68cc03316757ed5022b2e66f28a5a1f468ff4` |
| `C048` | `manifest` | `/home/fansenwei/pssolver_s10integ_claude_review_20261028_v1/checksums.sha256` | 5393 | `206988e0774664d5e49a9ae3838c0fbd5ddf2173726d3470a77505a448616445` |
| `C049` | `file` | `/home/fansenwei/pssolver_s10integ_claude_review_20261028_v1/final_report.md` | 9221 | `2bf77da47d4067cdbfdd2d4c07411eba951cbd406fdd9d013b055d25d6338acb` |
| `C050` | `file` | `/home/fansenwei/pssolver_s10integ_claude_review_20261028_v1/summary.json` | 1139 | `83f285b10e52ed0f9ca8cdbd6388e5d5576b7e141d475640b01a06f46e82fc8c` |
| `C051` | `complete` | `/home/fansenwei/pssolver_s10reach_claude_review_20261028_v1/COMPLETE` | 96 | `5740b1b3e5354c5d0453611f7f91d8d548f4ecd00f01a408fdf8f4b1326b4ec4` |
| `C052` | `manifest` | `/home/fansenwei/pssolver_s10reach_claude_review_20261028_v1/checksums.sha256` | 4862 | `687b0a77a6aac5a5058a90f8d6eb877dbcc3a7917b3270c5c1c068d13b0d423c` |
| `C053` | `file` | `/home/fansenwei/pssolver_s10reach_claude_review_20261028_v1/final_report.md` | 14562 | `ed0deeba504a15336f4d1dd5c1f9b260c0e53b0beda6b6155590b2c91b3b1dd0` |
| `C054` | `file` | `/home/fansenwei/pssolver_s10reach_claude_review_20261028_v1/summary.json` | 2585 | `b162433e52bbeea8a255a214e8a2ca6dcf62e0c570d4510cdc6e7e6c0e33b88d` |
| `C055` | `complete` | `/home/fansenwei/pssolver_s10rw_claude_review_20261027_v1/COMPLETE` | 116 | `e9c13586e1ec77b4873e61bfc364f5fbf195728f996fa4df520b2a0478385fad` |
| `C056` | `manifest` | `/home/fansenwei/pssolver_s10rw_claude_review_20261027_v1/checksums.sha256` | 3476 | `3e063a04821dbf75464137fa146d46cc1caf78b7f5a1c39347e33c9c58abc2ee` |
| `C057` | `file` | `/home/fansenwei/pssolver_s10rw_claude_review_20261027_v1/final_report.md` | 7188 | `8806bdf5b3385efde70eaf27b3182dedecdb894a29cd0759177e39618f66f550` |
| `C058` | `file` | `/home/fansenwei/pssolver_s10rw_claude_review_20261027_v1/summary.json` | 1622 | `10fb50725de7d7fa07c9f3eb527dafa998b46caf3fbc31339f01408a2036f770` |
| `C059` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2/COMPLETE` | 234 | `044e858fed1ed6c24679f97452c766ea77c4ee6f3c11de2da47e2446ca8f18c2` |
| `C060` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2/checksums.sha256` | 52638 | `fe9c9ba3742c297ea7091d4733d1a0d94a74efc496450288a3d0b65e068eab69` |
| `C061` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2/final_report.md` | 4245 | `406c52abb2ab72d2dd3c2a007647a7792e40a728376219a4649d01498a8920d9` |
| `C062` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2/results/cpu_memory_comparison.json` | 1216 | `51fc407ff68055025fddad92f5645895e199ed05559e12b4059055221b1b99ee` |
| `C063` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2/summary.json` | 1737 | `989f2d97aa113b9934eacf72721b545cd518f78737b8bff8fa97ac54ca3fc47f` |
| `C064` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/COMPLETE` | 188 | `aa6ef2bdd45a4ddba8bfd36432c9299908638d185b2561a39b6eecf5a15d0f72` |
| `C065` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/artifacts/provider_f94b9be.tar` | 26767360 | `0e31ef0bd9b1ff5283b23c1fbd4237975ebebf5be9460d9c4bef3fc365cfaa07` |
| `C066` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/artifacts/provider_support_658baca.tar` | 2457600 | `d3fb444eb5ba627a5e9b41b241c40a7c14ef799eed04bfb5e45611a55fa6f561` |
| `C067` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/artifacts/pssolver-0.2.2-py3-none-any.whl` | 801349 | `a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e` |
| `C068` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/artifacts/pssolver-0.3.0.dev0-py3-none-any.whl` | 646534 | `4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8` |
| `C069` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/checksums.sha256` | 2773 | `839498706d15bbad2c41fa607159bb242d66cea862743f632da003c7f88815a3` |
| `C070` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/plans/cpu_rehearsal_plan.json` | 82807 | `6a8a81115d20c4242191da475cf8506cc716fb2c0071a001e46963509c6f80a9` |
| `C071` | `file` | `/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1/plans/h100_plan.json` | 82978 | `57a5bdbf255f7e63376f633401050619db9df21057d6a3aa51fb0466540094aa` |
| `C072` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v1/COMPLETE` | 75 | `6670588e5138effa8a0812f2b0d19a40db7ebd92279283e71c516e9c3ad8c2ee` |
| `C073` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v1/checksums.sha256` | 2289 | `0860fe21b2be56744592aea9633f9fda5b4dfcfa45d1b0460908f76414500b3c` |
| `C074` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v2/COMPLETE` | 226 | `b5b7b1f438485c544dee914af6f5201b1d46f45a8761cfaa9d427317b4e6418e` |
| `C075` | `file` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v2/analysis/closure_adjudication.json` | 2995 | `4969cdc062d6cab301f3e3b5f0c411da02a37f997195eeaf6316b6c13c6e032f` |
| `C076` | `file` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v2/analysis/noninterrupting_recompute.json` | 14965 | `1757374241e033c77a5fbe0b31df29e6f33a0dcae3e777c4ae45754d5927df01` |
| `C077` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v2/checksums.sha256` | 3171 | `0212cf6b98eb06d4f64aa6c5509209714b3c0b5aab5fa2ab51368ce519d80895` |
| `C078` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3/COMPLETE` | 133 | `78b251e2ee0355819b0edc2a961352e83d40b23a3a29aab0f54b31ca2921f424` |
| `C079` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3/checksums.sha256` | 45859 | `8311695d7dc3031b7aed0574a0239354b35efad70c4b6d2a3223f8795accf1ac` |
| `C080` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3/results/candidate_bound/matrix.json` | 684760 | `f4e42828e2c5dc1caad9729fb1b2b005d08a4f53cc4b3e82274dae436ae71085` |
| `C081` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3/results/self_control/matrix.json` | 682756 | `5f652cdc9c79093bbae4862fc3080a839cec7c6969e0e3b4bc76d7c4c0a710d2` |
| `C082` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4/COMPLETE` | 121 | `911ee93fa7daad7cc555de8f2a1168da3a2d823d34c43019e24ca8cfd35e41e8` |
| `C083` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4/checksums.sha256` | 2007 | `b35113dfc0bb6b0acd3cc58b402a0ac460c85850957a7a9c3451e6734c6c71d8` |
| `C084` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4/logs/pytest.junit.xml` | 817 | `f1431e7daacff3820c009244d861115dc563b44b5659e96c6407bcd5485dd553` |
| `C085` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4/logs/pytest.log` | 98 | `cf0c2cc2867610db68ff0a051d55d0e206ddaa505c66c6ee68075be937872047` |
| `C086` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4/results/analysis.json` | 2727 | `d0d5b00ca1b670d747826b9b122c8d38557a1e3a70512404895e9a2653221abe` |
| `C087` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/COMPLETE` | 92 | `f448526454b1ac79b110c6238d81295c7fff03416ba356e89a0ee8d92e3e1b10` |
| `C088` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/checksums.sha256` | 111191 | `679ba7adaceddceaebe55769b1d2bc81a7c4c47c680525465a9884ea482aa02d` |
| `C089` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_c105_cost_5x30/matrix.json` | 245166 | `ecf2638b45ab5e4febd61d0b5c25fa2f13936aace40085704dbf078777506ea2` |
| `C090` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_optimized_bound_2573dec_5x30/matrix.json` | 483945 | `f050556410905a56b53d53fa218b90a43d8cf63ef81e9c867cb6cf2636802953` |
| `C091` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_optimized_final_5x30/matrix.json` | 483699 | `3f9bc9ab30efe70c997731dfb3adbbcb2356e1c8e79455a505187d5b4ccc8a17` |
| `C092` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_s06_pre_post_5x30/matrix.json` | 486665 | `b551b9d8a5ff344106996647d6fe324a41764877a2293db9a9bc34fab156b5b2` |
| `C093` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_v022_self_5x30/matrix.json` | 484014 | `c4cc3dc2892b7002e707fb93067e87ed61f6dcd0811dc22e2a4b22317c640fc3` |
| `C094` | `file` | `/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2/results/s09_diagnosis_v022_self_repeat_5x30/matrix.json` | 484741 | `1492a54abe3842d0af74c4bad70c46d894ba2f5373fd38d207167a7f74326299` |
| `C095` | `complete` | `/home/fansenwei/pssolver_v0_3_s09_cpu_transient_telemetry_d568039_20261023_v5/COMPLETE` | 145 | `5860465addbcb45f2236e7910cc10d514484dacb59ac3997a2e78b948dcf5f17` |
| `C096` | `manifest` | `/home/fansenwei/pssolver_v0_3_s09_cpu_transient_telemetry_d568039_20261023_v5/checksums.sha256` | 3115 | `ca878706bb87c5f14b6956a289d06a523241e82d78147cf6eb1a070f4ef25291` |
| `C097` | `complete` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/COMPLETE` | 240 | `bf5077c63d3a1c94618f6a5489302fedc19455bac772a9af58a6ceabf900f987` |
| `C098` | `file` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/PRUNED.txt` | 2627 | `13bb65414c8f40172da876e034f961a8cd0f8b9c7039c52b655fdb606033ce88` |
| `C099` | `file` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/STOPPED_PRE_DELETION.md` | 1049 | `24cd863ef3eeb47245e19e9ef61fd9a02e3cf379d97dff8c614ba548a1c6aa1c` |
| `C100` | `manifest` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/checksums.sha256` | 1152 | `261108f0a2a473e89de0fb76bd49fb670e14d56be4b1d4ad4893cef5e37ae269` |
| `C101` | `file` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/post_cleanup.json` | 1301 | `7f9ea6eebd97a122589af2400a736dc972f72de9131ac8128e2d96acea3f1ec7` |
| `C102` | `file` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/pre_deletion_inventory.json` | 116868 | `3d80c339667a63a69d0b3fae3e64b34b590d428546b0141acf74f14522e7101d` |
| `C103` | `file` | `/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1/worktree_inventory.json` | 16497 | `94472051c796928e9109a48bbaf9973ce91e6665e7357cf38156faa4839f54a1` |

## 边界

- JSON/文本字段检查由文件的精确 SHA-256 绑定；S-09 stop analyzer 另外原样复算关键语义。
- `checksums.sha256` 不只核对自身 SHA-256 和行数，还逐项重算清单内文件。
- support tar 另外检查不得包含 `/pssolver/` 或 `/pssolver_control/` 成员。
- 不复制 PSSolver_sync 或历史归档内容，不修改、移动或删除本机证据。
