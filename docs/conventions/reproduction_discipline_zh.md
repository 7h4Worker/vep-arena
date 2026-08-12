# 复现任务纪律

本仓库的复现任务必须把“官方复现”和“工程近似/诊断运行”分开，避免结果混用。

## 结果分级

- `official_*`：只能使用论文或公开代码明确指定的协议、依赖、滤波器、通道、窗口、交叉验证和指标。若官方依赖缺失，任务必须失败并给出错误信息。
- `legacy_*`：历史结果或已知不再作为正式结论使用的结果。必须在目录名或报告中说明原因。
- `smoke_*`：小样本验证，只能用于检查数据读取、形状、协议和基本数值趋势。
- `diagnostic_*`：工程诊断或替代实现，用于定位问题，不作为论文复现结果。

## 依赖和后端

- 官方代码使用的关键后端不得被静默替换。例如官方 FBDCCA 使用 MNE FIR 时，默认 task 必须使用 MNE FIR。
- 替代后端必须显式命名，例如 `scipy-fir-legacy`，并写入 `manifest.json`、`summary.csv` 或 task 报告。
- 如果当前 Python 环境缺少官方依赖，默认行为是失败；只有用户明确要求 fallback 时才允许运行替代后端。

## Python 执行器

- 正式复现命令不得使用裸 `python`，也不得假设当前 shell 已激活正确环境。
- README、NOTES 和批处理命令必须写出明确解释器。当前默认使用 `D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe`，或使用项目选定的统一 wrapper。
- `manifest.json` 必须记录关键环境信息，至少包括 Python 路径、关键 backend、依赖缺失时的行为。
- 若 `official_*` task 依赖 MNE、PsychoPy、MATLAB 或 CUDA，对应环境必须在 task README 中显式说明；缺失时失败，不自动降级。
- `.venv` 已通过 `uv sync --extra all` 补齐 MNE、torch 和报告依赖，是 Arena 离线复现默认执行器；若后续任务需要 PsychoPy/在线实验，可显式切到历史 conda 环境。

## 产物要求

每个正式 task 至少写出：

- `manifest.json`：数据集、subjects、窗口、方法、通道、CV、关键后端、运行状态。
- `unit_manifest.csv`：每个 subject/paradigm/method/window 组合的完成状态。
- `summary.csv` 和 `subject.csv`：聚合结果。
- `predictions.csv`：后续决策信道/混淆矩阵分析所需的 trial-level 预测。
- `report.md` 或中文报告：说明协议、关键差异、官方对照和限制。

## 命名规则

- 正式结果目录不能包含未说明的近似实现。
- 若发现正式目录中混入了近似结果，应移动为 `*_legacy_YYYYMMDD`，再重新生成正式目录。
- 结果目录继续由 `.gitignore` 忽略；代码、runner、README、NOTES 和报告模板可以纳入版本。
