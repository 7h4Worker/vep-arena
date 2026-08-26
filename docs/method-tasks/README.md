# Method Task 规格文档（与 tasks/methods/ 一一对应）

每个 `MNN_xxx.md` 对应 `tasks/methods/MNN_xxx/` 一个实现任务。
文档 = 论文规格 + curated 包索引 + 验证协议 + 期望结果。

## 第一批：空间滤波类（TRCA/CCA 骨架扩展）—— 9 个任务

| 任务 | 方法 | 论文 | curated 包 | 状态 |
|---|---|---|---|---|
| M01 | RESS | Xu/Ke 2024 TNSRE | `28_2024_xu_rhythmic_entrain` ✅ | 待实现 |
| M02 | PRCA | Ke 2024 TBME | `19_2024_ke_bprca` ✅ | 待实现 |
| M03 | sTRCA (time-filter+sim) | Measurement 2024 | `33_2024_yin_strca` ✅ | 待实现 |
| M04 | LA-TRCA | TNSRE 2022 | `34_2022_huang_latrca` ✅ | 待实现 |
| M05 | Multi-objective high-pass | TIM 2022 | `35_2022_zhang_mohp` ✅ | 待实现 |
| M06 | Sinusoidal-referenced TRCA | Infocomm 2023 | `36_2023_wang_sinref_trca` ✅ | 待实现 |
| M07 | xTRCA | NeuroImage 2019 | `37_2019_tanaka_xtrca` ✅ | 待实现 |
| M08 | gTRCA | Sci Reports 2020 | `38_2020_tanaka_gtrca` ✅ | 待实现 |
| M09 | scTRCA | JNE 2021 | `39_2021_sun_sctrca` ✅ | ✓ |

> curated 包索引规则：`curated/papers/<分类>/<NN_YYYY_author_topic>/paper_record.json`
> 完整路径前缀：`D:\ProjData\literature_workspaces\ssvep_classic_reproduction_20260729`
