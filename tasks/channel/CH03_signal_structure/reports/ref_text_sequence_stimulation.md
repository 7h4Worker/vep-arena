# 参考文献: Text Sequence Stimulation for SSVEP BCI

> **论文**: Li, Zhang, Song, Zhang, Chen, Wang, Gao. "Text Sequence Stimulation for High-Speed and Comfortable SSVEP BCI." *Cyborg and Bionic Systems*, 2025.
> **关键词**: SSVEP, Text Sequence, TDCA, 腹侧视觉通路, 舒适性

---

## 核心贡献

提出**文本序列刺激** (Text Sequence Stimulation) 范式, 用中文字符序列替代传统亮度闪烁, 实现:
- **235.12 ± 30.12 bits/min** 在线 ITR (40 目标)
- 比传统亮度刺激更舒适 (Likert 评分 3.9 vs 2.0, 5=most comfortable)
- 利用腹侧视觉通路 (occipito-temporal) 而非仅枕区

## 刺激参数

| 参数 | 值 |
|------|-----|
| 频率范围 | 3.4–11.2 Hz (0.2 Hz 间隔) |
| 相位间隔 | 0.35π |
| 目标数 | 40 |
| 刺激时长 | 0.7s (在线) |
| 编码方式 | JFPM (Joint Frequency-Phase Modulation) |
| 字符序列 | 中文字符, 帧周期内切换 |

## 解码器配置

| 参数 | 值 |
|------|-----|
| 方法 | TDCA (Task-Discriminant Component Analysis) |
| 导联数 | 39 (后部) |
| 子空间分量 | N_k = 6 |
| 延迟增广 | lag = 4 |
| 滤波器组 | 5 个子带 |
| 滤波器权重 | w(n) = n^(-a) + b, a=2, b=0.1 |
| 训练 | 5 blocks (leave-one-out) |

## 频率依赖的空间模式

关键发现: **低频 (3–8 Hz) 激活 occipito-temporal 区域, 高频 (>10 Hz) 集中于 occipital**

- 3.4–8 Hz: 明显的颞叶后部响应 → 腹侧视觉通路 (ventral stream)
- 10–11.2 Hz: 标准枕区响应
- 这解释了为何文本序列刺激可用更低频率: 利用了亮度刺激未触及的脑区

## 与传统亮度刺激的对比

| 指标 | 文本序列 | 传统亮度 |
|------|---------|---------|
| 频率范围 | 3.4–11.2 Hz | 8.0–15.8 Hz |
| 在线 ITR | 235 bpm | ~250 bpm (Benchmark 最佳) |
| 舒适性 | 3.9/5 | 2.0/5 |
| 激活区域 | occipito-temporal + occipital | 主要 occipital |
| 低频利用 | 可用 3–8 Hz | 3–8 Hz 效果差 |

## 对本分析的启示

1. **频率依赖的空间模式**: 信道矩阵应按频率段分组分析, 低频和高频可能有不同的信道特性
2. **39 导联 vs 9 导联**: 论文使用 39 导联覆盖 temporal 区域, 对应本分析中 32ch/66ch 配置的优势
3. **TDCA 参数参考**: N_k=6, lag=4 是经过优化的参数, 可作为信道容量分析的基准
4. **低频段的潜力**: 3.4–8 Hz 频段在传统分析中被忽略, 但在文本序列范式中有效 — 信道容量可能被低估
5. **舒适性-性能权衡**: 略低的 ITR 换取显著更高的舒适性 — 实际应用中的关键维度
