# S1 阶段记录 — 经验决策信道效率

> 执行器 wsl-sandbox-conda | 走 vep_arena/channel/capacity.py, 无重实现

## 结果 (eta=MI_uniform/C0, 5 范式切片)
               paradigm  n_classes  window  accuracy  MI_uniform     C_BA      eta     n
         FDMA/Benchmark         40     1.0  0.936786    4.752702 4.766678 0.893041  8400
DualFreq/Binocular-SFSP          8     1.0  0.817411    1.822564 1.830683 0.607521  2240
DualFreq/Binocular-DFDP          8     1.0  0.954464    2.625519 2.627586 0.875173  2240
            CDMA/JFPM-8         40     2.0  0.544833    2.046627 2.183328 0.384565 12000
            CDMA/NBRS-8         40     2.0  0.568750    2.230134 2.294608 0.419046 12000

## 已证实
- S1 方法学跑通: predictions.csv -> 经验决策信道 P(mhat|m) -> C1/MI_uniform/C_BA, 全走现有干净核心。
- **G_dof 首个真结果(可比)**: Binocular 同数据/同接收机ETRCA/同窗口1.0s/同K=8, SFSP->DFDP 的 eta 60.8%->87.5% (+26.8pp)。加频率+相位两自由度显著提升编码效率——受控单码增益, 非分类准确率。

## 已暴露可比性陷阱 (印证 comparability_key)
- CDMA K=40 win=2.0s 与 FDMA K=40 win=1.0s 不同 comparability_key, eta 不可直接横比。图(a)已标注。
- 跨范式横比必须落到 S4 固定物理时长口径 + 同 receiver。

## pilot 反哺框架 #1 (实证)
- predictions 列名不统一: Benchmark/Binocular=`true`, JFPM/NBRS=`target`。已加 norm_cols 适配。
  -> manifest/pipeline 必须声明 label 列名或强制标准列。由真实分析逼出, 直接进 schema。

## 下一步
- S2 G_repeat: block 结构构 I_1 vs I_R + 独立性检验。
- S4 时间双口径: 裁决 CDMA 长窗口是码字展开还是重复。
