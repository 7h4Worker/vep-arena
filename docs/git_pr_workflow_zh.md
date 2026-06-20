# Git 与 PR 工作流程

## 原则

VEP Arena 的新增工作通过 Git 记录，通过 GitHub 公开仓库协作。默认语言为中文：issue、PR 标题、PR 描述、过程记录、合并说明都使用中文。

工作原则：

- `master` 保持可读、可运行、可回溯。
- 每个主题使用独立分支。
- 每个 PR 只解决一个清晰范围。
- PR 描述记录背景、改动、验证、未完成事项。
- 不把生成结果、数据集、模型权重提交进仓库。
- 不把无关实验改动混进范围/文档 PR。

## 分支命名

推荐格式：

```text
docs/<topic>
data/<dataset>
feature/<capability>
experiment/<method-or-dataset>
fix/<issue>
```

示例：

```text
docs/project-scope
data/beta-adapter
feature/mne-qa-views
experiment/dnn-window-sweep
fix/preprocess-ledger
```

## PR 模板

每个 PR 描述至少包含：

```markdown
## 背景

说明为什么要做这个改动。

## 改动

- ...

## 验证

- [ ] 命令或人工检查

## 影响范围

说明会影响哪些数据集、算法、报告或脚本。

## 未完成/后续

- ...
```

## 当前首批 PR 拆分

### PR 1：项目范围与工作流

分支：`docs/project-scope`

内容：

- `docs/project_scope_zh.md`
- `docs/git_pr_workflow_zh.md`

目标：先把当前项目范围和协作规则写清楚。

### PR 2：预处理契约

分支：`docs/preprocessing-contracts`

内容：

- `docs/preprocessing_contracts_zh.md`

目标：把 `raw_epoch`、`ssvep_toolbox_fb5`、`dnn_cheby3_fb`、`erp_mne_basic` 等契约固定下来。

### PR 3：MNE 环境检查与 QA 图骨架

分支：`feature/mne-qa-views`

内容：

- 环境检查脚本。
- MNE bridge 或 QA figure 脚本。
- 示例输出说明。

目标：低侵入接入 MNE，不影响现有 benchmark runner。

### PR 4：BETA 数据事实与 loader

分支：`data/beta-adapter`

内容：

- `docs/datasets/beta_notes_zh.md`
- `vep_arena/data/beta.py`
- BETA smoke 脚本。

目标：先完成 BETA 单 subject smoke，再扩展到完整 benchmark。

## 合并策略

默认使用普通 PR 合并。每个 PR 合并前至少完成：

- 工作树只包含该 PR 范围内文件。
- 本地验证命令已记录。
- PR 描述中的验证项已勾选或说明原因。

如果一个实验分支很长，应定期把过程记录写到 PR 评论中，避免只在最终结果里出现一大坨不可追溯变化。

