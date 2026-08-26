# M07 — xTRCA（Cross-correlation TRCA）

## 论文
- 标题: Cross-correlation task-related component analysis (xTRCA) for enhancing evoked and induced EEG responses
- 作者: Tanaka T, Miyakoshi M
- 期刊: NeuroImage 197:672-687, 2019
- DOI: 10.1016/j.neuroimage.2019.04.049

## Curated 包索引
- 包: `37_2019_tanaka_xtrca`
- 位置: `curated/papers/03-vep-algorithm/11-spatial-filter/37_2019_tanaka_xtrca/`

## 核心算法
xTRCA 扩展 TRCA 到**诱发 + 诱发（evoked & induced）响应**：
- TRCA 用试次间协方差（互协方差）提取任务相关成分
- xTRCA 引入**信号与其时间延迟版本的交叉相关**，捕捉诱发放电的时变结构
- 数学上：最大化 $\sum_{i\neq j} \text{Cov}(X_i, X_j^{(\tau)})$（含延迟 $\tau$ 的互协方差）

## 接口与验证
- 对齐 `TRCA`，新增延迟 $\tau$ 参数
- Benchmark 9ch；与 TRCA/eTRCA 对比

## 验收标准
- Benchmark 结果写入 `tasks/methods/M07_xtrca/results/`
