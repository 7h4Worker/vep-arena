# 2026-08-11 分支清理

## 清理前状态

main 分支 `d5a68c7`，ahead 1（`整理 docs/ 目录结构`）。

| 分支 | 基点 | 独占内容 | 处理 |
|------|------|---------|------|
| `feat/jbhi-bessvep-evaluation` | `5643d5f` | 无（已合并到 main） | ✅ 已删除 |
| `pr/arena-baseline-contracts` | `19c14a8` | 无 — 文档/规范层改动，已被 main `3259d6d` 覆盖 | ✅ 已删除 |
| `pr/binocular-methods-datasets` | `19c14a8` | 无 — ecca/fbdcca/sscor + binocular 数据加载器 + 3 个 task 目录，main 均已有更新版本 | ✅ 已删除 |
| `feat/theory-p0-p2-survey-v2` | `ccd5979` | **有** — `tasks/ssvep_neural_communication_survey_v2/`（P0 + P1 容量分析），main 无此内容 | 🔒 保留，待 T1 重组时合入 |

## feat/theory-p0-p2-survey-v2 详情

4 个未合并 commit：

```
ccd5979 Add P1 short-window capacity analysis
763c0e8 Record decoder source run names from inputs
90ca27a Allow fixed canonical epoch cache windows
24a7393 Add unified SSVEP P0 decoder analysis
```

独占文件：

```
tasks/ssvep_neural_communication_survey_v2/README.md          (40行)
tasks/ssvep_neural_communication_survey_v2/run_p0_analysis.py (190行)
tasks/ssvep_neural_communication_survey_v2/run_p1_frontload_analysis.py (324行)
tests/test_survey_v2_p0.py                                    (49行)
tests/test_survey_v2_p1.py                                    (38行)
```

另改动 `scripts/run_traditional_benchmark.py` 1 行（scripts/ 本身待归档，可忽略）。

**计划**：T1 tasks/ 重组时，将 survey_v2 的脚本纳入信道分析主题目录，合入后删除该分支。
