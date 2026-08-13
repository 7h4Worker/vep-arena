# VEP Arena 项目规则

> 本文档是项目的**活文档**，随项目演化持续更新。
> 最后更新: 2026-08-12

---

## 1. 仓库边界：公开 vs 本地

远端仓库（GitHub）只包含**可公开的基础设施层**，不包含未发表的分析结论。

| 层级 | 推送到 remote | 说明 |
|------|:---:|------|
| `vep_arena/data/` | ✅ | 数据集加载器（公开数据集的接口代码） |
| `vep_arena/methods/` | ✅ | 算法实现（CCA, TRCA, TDCA, bPRCA 等） |
| `vep_arena/signal/` | ✅ | 信号处理工具 |
| `vep_arena/channel/` | ✅ | 信道容量计算工具（通用数学工具） |
| `tests/` | ✅ | 单元测试 |
| `configs/` | ✅ | 配置模板（不含实际路径） |
| `docs/conventions/` | ✅ | 工作流、契约、架构 |
| `tasks/` baseline 脚本 | ✅ | 各数据集的标准算法评估脚本 |
| `tasks/` 分析类脚本 | ❌ | 决策信道、跨范式、码本等分析（论文相关） |
| `outputs/` | ❌ | 跨任务综合产出（见§3） |
| `docs/journal/` | ❌ | 工作日志（含未公开研究进度） |
| `.cache/` | ❌ | 运行期缓存（epoch cache 等，已在 .gitignore） |

**判断标准**：如果某文件的存在会泄露未发表论文的核心思路或结论，就不推。

### 当前 remote 状态

- 默认分支: `main`（origin/HEAD）
- PR #1-5 已合并，#6-7 已关闭
- Baseline 里程碑已推送（commit `3259d6d`，84 文件）
- 后续新增功能以 PR 组织

---

## 2. 目录架构

```
vep_arena/                     # Python 库（可 import，推 remote）
├── data/                      #   数据集加载器
├── methods/                   #   算法实现
├── nn/                        #   DNN 模型
├── signal/                    #   信号处理
├── channel/                   #   信道容量数学工具
├── plots/                     #   绘图工具
├── neuroviz/                  #   MNE 桥接
└── metrics.py                 #   评分指标

tasks/                         # 分析任务（脚本 + 本地结果）
├── baselines/                 #   公开数据集标准评估（BL01–BL12，推 remote）
├── bessvep/                   #   BesSSVEP 私有数据集（BS01–BS06，本地）
├── channel/                   #   信道容量理论分析（CH01–CH07，本地）
├── probes/                    #   探索性分析（PB01–，本地）
├── _shared/                   #   跨任务共享模块
└── _legacy/                   #   已归档

.cache/                        # 运行期缓存（epoch cache 等，本地）
outputs/                       # 跨任务综合产出（本地，见§3）

docs/                          # 文档（项目自身结构与进程）
├── project_rules.md           #   本文档（根入口）
├── conventions/               #   项目约定与契约（酌情推 remote）
└── journal/                   #   工作日志（本地）

notes/                         # 研究参考材料（本地，gitignored）
├── references/                #   方法参考与复现笔记
├── paper_notes/               #   论文阅读笔记
├── papers/                    #   参考论文 PDF
├── datasets/                  #   数据集笔记
├── sibling_snapshots/         #   DNN 兄弟项目快照
├── unification/               #   评估管线统一设计
├── plans/                     #   规划文档
└── archive/                   #   已归档文档

configs/                       # 配置模板（推 remote）
tests/                         # 测试（推 remote）
```

---

## 3. 产出物流向

```
tasks/ 各任务独立运行
    ↓ 产出 predictions.csv, capacity.csv 等（在各自 results/ 下）
    ↓
outputs/ 综合消费
    ↓ 跨任务的 plot 脚本读取多个 tasks/*/results/
    ↓ 产出综合 figures/, tables/, reports/
    ↓
docs/journal/ 记录决策
    引用 outputs/ 中的图表和结论
```

**核心规则**：
- `tasks/*/` 只关心自己的数据集和分析，产出到自己的 `results/`
- 需要横跨多个 task 的比较图、汇总表、综合报告 → `outputs/`
- `outputs/` 中的脚本可以 import `vep_arena` 库，可以读取任意 `tasks/*/results/`
- `outputs/` 整体 gitignore，不推 remote

---

## 3b. 工作区清洁规则

| 产物类型 | 正确位置 | gitignore |
|----------|---------|-----------|
| 任务产出 figures/tables | `tasks/*/results/` 下 | ✅ 已覆盖 |
| 跨任务综合图表 | `outputs/` | ✅ 已覆盖 |
| 版本化 figures 快照 | `tasks/*/results/figures_v*/` | ✅ `tasks/**/figures_*/**` |
| `_` 前缀探索脚本 (.py) | 允许存在，但提交前审查 | ❌ 不自动忽略 |
| `_` 前缀临时数据 (.csv/.txt) | 不应提交 | ✅ `tasks/**/_*.csv` `_*.txt` |
| 参考 PDF/mat/临时文件 | `tmp/` | ✅ 已覆盖 |
| 论文提取文本 | `tmp/` 或不跟踪 | ✅ |

**`_` 前缀 .py 脚本规则**：允许在开发中存在，但提交时必须做以下二选一：
1. 有保留价值 → 去掉 `_` 前缀，归入正式脚本
2. 纯临时探索 → 移入 `tasks/*/results/_explore/` 或删除

---

## 4. 命名规则

### tasks/ 下的目录

- `baselines/` 下: `{paradigm}_{dataset}` — `ssvep_benchmark_40t`, `cvep_jfpm_nbrs`
- 其他主题下: 描述性短语 — `dmc_benchmark`, `dmc_envelope`, `info_accumulation`
- 同一父目录下所有子目录遵循同一字段模式

### 分支命名

```
data/{topic}           数据加载器
feature/{capability}   新功能/算法
experiment/{target}    实验任务
docs/{topic}           文档
fix/{issue}            修复
```

### Commit message

中文，一句话概括改动。如果有 PR 关联，commit 里不需要重复 PR 描述。

---

## 5. PR 工作流

遵循 [`docs/conventions/git_pr_workflow_zh.md`](conventions/git_pr_workflow_zh.md) 模板：

```markdown
## 背景
## 改动
## 验证
## 影响范围
## 未完成/后续
```

PR body 末尾链接 journal：`📓 docs/journal/YYYYMMDD_slug.md`

---

## 6. 待办事项

> 以下事项来源于 `docs/journal/20260810_项目整理规划与工作区清理.md`，
> 每项可作为独立 session 推进。

| # | 事项 | 依据文档 | 前置 |
|---|------|---------|------|
| T1 | ~~tasks/ 目录重组（25+ 扁平 → 4 主题）~~ ✅ | journal/20260811 | 无 |
| T2 | ~~scripts/ 归档（第一代脚本，已被 tasks/ 取代）~~ ✅ | journal/20260811 §scripts | 无 |
| T3 | ~~results/ runs/ 搬移到 .cache/ 和 task 本地（572MB + 21GB）~~ ✅ | journal/20260810 §5 | 无 |
| T4 | ~~`_` 前缀脚本审查（benchmark_dmc, hd200）~~ ✅ | 本文档 §3b | T1 后更好 |
| T5 | ~~分析类内容本地 commit~~ ✅ | 本文档 §1 | T1 后路径稳定 |
| T6 | ~~旧分支清理（feat/*, pr/*）~~ ✅ | git branch -v | 无 |

---

## 7. 变更记录

| 日期 | 改动 |
|------|------|
| 2026-08-13 | 合并 theory 分支并迁入 CH08_survey_p0p1/，T4/T5/T6 全部完成，§6 清零 |
| 2026-08-12 | 消除 scripts/ results/ runs/ 顶层目录；活跃缓存→.cache/，结果→task 本地，T3 完成 |
| 2026-08-11 | 更新：分支 → main，docs/ 精简（参考材料移至 notes/），增加§6 待办事项 |
| 2026-08-10 | 初建：仓库边界、目录架构、产出物流向、命名规则 |
