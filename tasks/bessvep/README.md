# BesSSVEP — 私有双目 SSVEP 数据集

BesSSVEP 系统的全部评估和分析，包含三个数据集配置和跨数据集分析。

## 数据集

| ID | 配置 | 目标数 | 被试 | 来源 |
|----|------|--------|------|------|
| BS01 | EMBC 9 目标 | 9 | 8 | EMBC 投稿 |
| BS02 | JBHI 16 目标 | 16 | 13 | JBHI 投稿 |
| BS03 | JBHI 35 目标 | 35 | 6 | 诊断扩展（未发表） |

## 脚本

### Baseline 评估
- `run_embc_9t.py` — BS01 标准基线
- `run_16t.py` — BS02 标准基线
- `run_35t.py` — BS03 标准基线
- `run_35t_five_subject_*.py` — BS03 五被试细分实验
- `run_35t_scoring_smoke.py` — BS03 评分验证

### 跨数据集分析
- `analyze_codebook.py` — BS04 码本结构分析（跨 9/16/35t）
- `analyze_codebook_signal.py` — BS04 信号层码本分析
- `run_codebook_tdca.py` — BS04 TDCA 码本评估
- `plot_fft_features.py` — BS05 FFT 频谱特征
- `build_receiver_matrix.py` — BS06 接收算法对比矩阵

## Results 目录

```
results/
├── BS01_embc_9t/       EMBC 9t baseline 产出
├── BS02_16t/           JBHI 16t baseline 产出
├── BS03_35t/           JBHI 35t baseline 产出
├── BS04_codebook/      码本分析产出
├── BS05_fft/           FFT 特征图
└── BS06_receiver/      接收算法对比产出
```
