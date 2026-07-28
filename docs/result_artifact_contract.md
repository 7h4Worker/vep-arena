# 结果产物契约

本文档规定 VEP Arena 实验输出的项目级标准。

## 完整运行目录

完整算法运行目录应包含：

- `manifest.json`：数据集、协议、方法、受试者、时间窗、通道、代码入口、缓存/预处理说明、状态和时间戳；
- `trials.csv`：聚合指标使用的逐单元记录，通常为 method/window/subject/block；
- `predictions.csv`：逐 trial 或逐 command 的分类记录，用于混淆矩阵和错误结构分析；
- `summary.csv`：方法/时间窗聚合结果；
- `subject.csv`：受试者级聚合结果；
- `block.csv` 或同等 split/session 表；
- `runtime.csv`：具有明确 fit/predict 阶段时的运行时间记录；
- `confusion_*.npy` 或 `confusions/*.npy`：由 `predictions.csv` 生成的混淆矩阵；
- 由 CSV/NPY 重新生成的图，不依赖隐藏的内存状态。

任何需要进入准确率聚合之外科学分析的结果都必须包含 `predictions.csv`。如果 runner 无法写出预测，manifest 必须说明原因，并将目录标记为 aggregate-only。

## 仅聚合目录

聚合或对比目录可以只包含：

- `summary.csv`；
- 可选的 `subject.csv`、`block.csv`、`paired_stats.csv` 或来源清单；
- `manifest.json` 或明确的来源账本。

这类目录是派生视图，不能当作原始实验运行目录。可以据此绘图，但不能据此进行混淆矩阵或错误结构分析。

## 预测表必需字段

任务适用时统一使用：

- `method`；
- `subject`；
- `block` 或 `session`；
- `trial_index`；
- `true`；
- `pred`；
- `correct`；
- `window` 或 `window_ms`；
- `targets`、`channels`、`time_samples` 等数据集上下文。

如果数据集原始标签为面向使用者的一基标签，输出应保持一基。内部使用零基标签时，manifest 必须明确记录标签约定。

## 版本控制边界

结果表、逐 trial 预测、图片、模型 checkpoint、滤波缓存以及原始/派生 EEG 都属于本地产物，即使它们是 task 验证所必需，也必须保持 Git ignored。

进入版本控制的是源码、测试、task 协议、NOTES 和报告模板，而不是运行生成物。
