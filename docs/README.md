# docs/ 导航

**入口**：[project_rules.md](project_rules.md) — 项目活文档，包含仓库边界、目录架构、命名规则、待办事项。

`docs/` 只放项目自身的结构文档和工作记录。方法参考、论文笔记、演示材料等在 `notes/`。

## 目录结构

```
docs/
├── project_rules.md       项目规则入口（活文档）
├── README.md              本文件
├── conventions/            项目约定与契约
│     architecture.md         层级架构
│     environment.md          Python 环境策略
│     git_pr_workflow_zh.md   Git/PR 工作流模板
│     preprocessing_contracts_zh.md   预处理溯源契约
│     project_scope_zh.md    项目范围定义
│     reproduction_discipline_zh.md   复现纪律分级
│     result_artifact_contract.md     产出物标准
└── journal/                工作日志（按日期）
      README.md               日志约定
      20260617_…               实验计划 / TDCA 回顾
      20260618_…               DNN 交接
      20260709_…               Session 交接
      20260727_…               EMBC/JBHI 数据审计
      20260809_…               提交整理计划
      20260810_…               项目整理与清理
```

## 不在 docs/ 的内容

| 位置 | 内容 |
|------|------|
| `notes/references/` | 方法参考文档（CCA, TRCA, MVMD, Wong 系列等） |
| `notes/paper_notes/` | 论文阅读笔记与提取文本 |
| `notes/papers/` | 参考论文 PDF |
| `notes/datasets/` | 数据集笔记 |
| `notes/sibling_snapshots/` | DNN 兄弟项目快照 |
| `notes/unification/` | 评估管线统一设计工作区 |
| `notes/plans/` | 规划文档 |
| `notes/ppt_figures/` | 演示图片 |
| `notes/archive/` | 已归档文档 |

`notes/` 整体 gitignored，不推 remote。
