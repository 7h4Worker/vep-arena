# 提交整理与 PR 组织规划

> 分支: `feat/jbhi-bessvep-evaluation` | 时间跨度: 2026-07-07 → 2026-08-09

## 背景

自 PR #5（`experiment/dnn-window-sweep`，2026-06-20 合并）以来，`feat/jbhi-bessvep-evaluation`
分支上积累了约两个月的工作，包含 8 个已修改文件和 ~50 个 untracked 文件/目录。
改动覆盖数据加载器、算法模块、信号处理、13 个数据集的 baseline 任务、
决策信道容量分析扩展、跨范式分析链以及码本结构分析。
全部改动从未推送到 remote，也未创建 PR。

此前的 PR #6（`pr/arena-baseline-contracts`）和 #7（`pr/binocular-methods-datasets`）
已关闭未合并，对应 remote 分支已删除，但本地 worktree 仍存在。

## 关键决策

- **不在当前分支直接提交所有改动**：改动跨度太大，需要拆分为多个主题 PR 以保持可审阅性。
- **按依赖链顺序拆分 7 个 commit / PR**：保证每步可独立编译测试。
- **目录重组在提交整理之后**：先把当前改动入库，再从干净状态做 `tasks/` 重组。
- **建立 `docs/journal/` 工作日志**：弥补两个月无结构化记录的空白。

## 提交拆分计划

按依赖顺序，每个 commit 对应一个 PR 分支：

| # | 主题 | 分支名 | 核心内容 |
|---|------|--------|---------|
| C1 | 新数据集加载器 | `data/new-dataset-loaders` | 4 个 data 模块 + local_paths 配置 + embc_jbhi 测试 |
| C2 | 双目接收器方法 | `feature/binocular-receivers` | bPRCA, bTRCA, FusionCA + eCCA 优化 + filters 模块 |
| C3 | Baseline 任务批量 | `experiment/baseline-tasks-batch` | 13 个数据集 baseline + 共享 runner + 已有任务扩展 |
| C4 | EMBC/JBHI 码本分析 | `experiment/embc-jbhi-codebook` | 码本结构 + FFT 特征 + 接收器对比矩阵 |
| C5 | Benchmark DMC 扩展 | `experiment/benchmark-dmc-extended` | 扩展时间轴 + 多通道扫描 + analyze/plot 参数化 |
| C6 | 跨范式容量分析链 | `experiment/crossparadigm-capacity` | DMC envelope + 信息积累 + 效率分解 + MIMO 设计 |
| C7 | 文档与约定更新 | `docs/conventions-update` | tasks README + CONTEXT.md + data_audit + unification/ + journal/ |

## PR 工作流

每个 PR 遵循已有 `docs/git_pr_workflow_zh.md` 模板：

1. 从 `master` 创建分支（或从前一个已合并的 PR 分支创建）
2. 提交 + 推送
3. `gh pr create` 使用中文标题和结构化 body（背景/改动/验证/影响范围/后续）
4. body 末尾引用本日志条目：`📓 Journal: docs/journal/20260809_commit-consolidation-plan.md`
5. 合并后在日志 README 索引中补充一行

## 现有散落文档归属

| 文档 | 当前位置 | 处理 |
|------|---------|------|
| lookback | `docs/lookback_2026_06_17.md` | 保留原位，日志索引中引用为"前史" |
| session handoff | `docs/session_handoff_20260709.md` | 保留原位，同上 |
| dnn handoff | `docs/dnn_handoff_2026_06_18.md` | 保留原位 |
| experiment plan | `docs/experiment_task_plan_2026_06_17.md` | 保留原位 |
| data audit | `docs/data_audit_embc_jbhi_20260727.md` | 随 C7 入库 |
| unification/ | `docs/unification/` | 随 C7 入库 |
| 任务 NOTES | `tasks/*/NOTES_*.md` | 各随所属 commit 入库 |

## 待清理的残留

- `tmp/` — 临时 PDF，不入库，确认 `.gitignore` 覆盖
- 2 个 orphan worktree（`pr/arena-baseline-contracts`, `pr/binocular-methods-datasets`）— remote 已删除，需 `git worktree remove`
- `dnn-integration` 分支 ahead 2 — 需确认是否还需要

## 遗留与后续

- [ ] 执行 C1-C7 提交和 PR 创建
- [ ] 每个 PR 合并后在日志中补充一行索引记录
- [ ] 所有 PR 合并后，从干净 master 开始 `tasks/` 目录重组
- [ ] 目录重组完成后写独立日志条目

## 相关文档

- `docs/git_pr_workflow_zh.md` — PR 模板和分支命名规范
- `docs/session_handoff_20260709.md` — 最近一次环境/结果状态快照
- `docs/lookback_2026_06_17.md` — 项目早期回顾
