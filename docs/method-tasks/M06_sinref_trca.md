# M06 — Sinusoidal-Referenced TRCA

## 论文
- 标题: Enhance detection of SSVEPs through a sinusoidal-referenced task-related component analysis method
- 期刊: IEEE Infocom Workshops, 2023
- DOI: 10.1109/INFOCOMWKSHPS57453.2023.10226001

## Curated 包索引
- **未提取**（DOI 10.1109/INFOCOMWKSHPS57453.2023.10226001）

## 核心算法
正弦参考 TRCA：
- TRCA 空间滤波 + 正弦/余弦参考信号融合
- 把 CCA 的参考信号优势并入 TRCA 的模板匹配

## 接口与验证
- 对齐 `TRCA`；Benchmark 9ch
- 写 `tasks/methods/M06_sinref_trca/results/`
