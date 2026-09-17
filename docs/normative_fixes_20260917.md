# 规范性修复记录与执行机验收（2026-09-17）

基线：`bda5a44a7f15644355401a8dbc9c4480823df0dd`，分支 `sync/public-main-202609`。
本记录落实 [三层职责](architecture.md) 中的一部分公共正确性约束，配合 [固定接入回环](integration_workflow_zh.md) 使用。
**代码提交完成不等于执行机验收完成；本轮不宣布真实数据、算法等价性或论文复现通过。**

## 1. 范围与修改归属

不搬目录、不改算法模块、不改原始数据、不迁移专题分析。不引入新依赖或新的框架服务。
保留三个已有脚本入口，将它们重复的参数解析、汇总、报告、写出和续跑逻辑接到公共实现；脚本中的方法创建与数据/模型调用保留原有流程，仍须真实小样本对照验证。

| 编号 | 问题与修改 | 归属 / 证据 |
| --- | --- | --- |
| N01 | 恢复 `RESULT_ROOT`、`RUN_ROOT` 兼容别名，保留新 `CACHE_ROOT`；避免旧 DNN/公开入口因 import 断裂而被误判需要重新迁移 | `vep_arena/config.py`；别名回归测试 |
| N02 | 范围/窗口参数拒绝重复、空输入、倒序、非有限值、零/负步长和过大扫描 | `evaluation.py`；正例与反例测试 |
| N03 | 完成度按被试/block 的实际唯一键检查；重复行报错，其他被试/其他 block 的同等行数不能算完成 | `evaluation.py`；重复、错范围、混合方法测试 |
| N04 | 样本数来自实际记录；保持历史等 block、等被试均值；校验预测数量、标签范围、重建准确率 | `evaluation.py`、`run_contract.py`；预测与聚合测试 |
| N05 | 报告读取当前 manifest 的数据集、协议、预处理及分方法输入；不再固定写 Benchmark；默认描述性结果而非隐含公平排名 | `evaluation.py`；BETA 元数据回归测试 |
| N06 | 续跑检查有效配置、代码版本、来源清单和产物指纹；保留 run ID 与上次执行记录；拒绝不完整/被修改/无来源证据的 checkpoint | `run_contract.py`、`evaluation.py`；参数变更、损坏、legacy、dirty 反例 |
| N07 | 混淆矩阵按 method + window 保存并回读校验，避免混合多个窗口；manifest 最后写入文件哈希清单 | `evaluation.py`；窗口、重算及篡改测试 |
| N08 | 提供显式源文件清单生成与检查；已核对来源的 epoch 缓存按来源和代码身份隔离，不直接信任旧缓存 | `record_sources.py`、`prepare_run`；内容变化/缓存隔离测试 |
| N09 | 提供只读产物检查入口；纯分析不强制输出分类预测，比较仅检查来源/兼容性声明存在，不替代科学比较判断 | `validate_artifacts.py`；analysis、classification 与无效 manifest 测试 |
| N10 | 三个旧入口调用同一套共享实现，并记录有效参数、环境包版本、标签 base、训练/测试 block 划分 | 三个 `scripts/run_*.py`；静态调用检查和模拟 worker 测试 |

## 2. 本轮实现的具体边界

当前共享校验器支持**一个 method/window/subject/block 对应一个评测单元**的旧式 block 评估。
训练组合、重复 seed、在线决策时刻等更多粒度需要保留 split/instance 身份，不能为了通过校验删除这些维度。它们暂不套用这个受限的公共写出器。

新增结果具有 `run_id`、`resolved_config`、`code`、`splits`、`method_inputs`、`artifacts` 和配置指纹。
现有一次 block 内每个 target 只有一条样本的格式，可以暂以 true label 区分该 block 内的试次，但会明确记录原始 trial 来源尚未核验。重复 target 的数据必须提供 trial ID/index，不能靠补编号冒充原始身份。

本轮审计会重算准确率、样本数、已有 trial 行的聚合及每窗口混淆矩阵；**不独立验证 ITR 时间口径、任务数学推导、预处理真实性、score 列语义或算法论文一致性**。analysis/comparison 的 `pass` 只说明所列文件与必要声明通过结构检查。返回值始终把真实数据、论文、task 数学与科学比较验收保留为 `not_run`。

## 3. 有意收紧的行为（不是静默兼容）

- 新运行拒绝写入非空结果目录；换 task name/output directory，不清空旧产物。
- legacy CSV/manifest 保持可读，但不能无条件 `--resume`。没有有效配置、已核对来源清单、同一干净代码版本和产物哈希的旧目录，应保留并开始新 run。旧调用方只传 `load_existing_rows(path)` 会收到明确错误。
- 本轮续跑只接受完整 method/window checkpoint；不自动修复中途写了一部分的窗口，不扩展 subject/window，不混入旧结果。线程数等有效参数变化目前同样保守拒绝。
- `--source-manifest` 核对的是调用者列出的文件，**不是自动证明已经列全所有数据依赖**。本地执行者应包含本次读取的全部原始文件和外部 metadata；读取期间不能修改源文件。未提供清单仍可新跑，但来源状态为 `not_run`，不能续跑。
- 已核对来源的 Benchmark epoch cache 会增加 `verified-...` 命名空间。未带清单的新运行仍沿用旧缓存行为；不能把它标成来源验收已通过。
- Toolbox runner 的 target 子集暂明确报错，直到有完整标签/分数列重映射；本轮没有宣称修好全部子集算法。
- 权威混淆矩阵在结果目录 `confusions/`，窗口对应关系在 manifest。旧 `runs/` 镜像仅为兼容；不再给多窗口新运行生成一个不带窗口的混合矩阵。旧消费者应读取 manifest；历史文件不会被删除。
- manifest 最后替换可以发现撕裂的 checkpoint，**不是多文件事务或自动回滚**；不支持并发进程写同一个 run。
- TDCA `--prebuild-only` 仅生成缓存状态，不是分类运行完成；正式运行使用另一个 task name。

## 4. 已执行的检查与未执行的检查

实现环境为 Linux / Python 3.13.5；这不是项目声明的 Python 3.11/3.12 环境。
容器不能直接克隆 GitHub；代码通过连接器读取，并在隔离目录重建相关文件进行测试。辅助的指标/混淆函数以及 12 项既有 evaluation 测试行为在隔离目录复建，它们不作为修改提交。

本轮提交 **67 项新测试用例**（参数化后），加上隔离复建的 12 项既有测试行为，共 **79 项通过**。范围包括参数/唯一键/配置/来源/产物反例、图表生成、审计 CLI，以及三个 runner worker 的模拟数据/模拟模型调用。模拟 worker 测试核对非连续 block 的训练/测试映射，不运行真实算法。

检查结果不外推：完整仓库测试、所有入口 import、Windows/Python 3.11/3.12、真实 EEG 小样本和旧/新算法输出对照均须执行机验证。不要把旧 PR 中的 `63 passed` 加上这个数字后直接宣称全量通过。

## 5. 执行机固定验证回环

### A. 拉取与快速回归

先确认工作区没有未保存修改；有本地修改则先自行保存，不使用 reset/clean/force。以下命令在仓库目录运行，每个外部命令失败都应停止，而不是继续标通过。

```powershell
git status --short
git fetch origin
git switch sync/public-main-202609
git pull --ff-only origin sync/public-main-202609
git rev-parse HEAD

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$evidence = "results/_verification/$stamp"
New-Item -ItemType Directory -Force $evidence | Out-Null
uv run --with pytest python -m pytest tests/ -q --junitxml "$evidence/pytest.xml"
if ($LASTEXITCODE -ne 0) { throw 'Full tests failed' }

foreach ($entry in @('scripts/run_traditional_benchmark.py', 'scripts/run_tdca.py', 'scripts/run_toolbox_ssvep.py', 'scripts/record_sources.py', 'scripts/validate_artifacts.py')) {
    uv run python $entry --help
    if ($LASTEXITCODE -ne 0) { throw "Entry failed: $entry" }
}
Get-ChildItem tasks/methods/M*/run.py | ForEach-Object {
    uv run python $_.FullName --help
    if ($LASTEXITCODE -ne 0) { throw "Entry failed: $($_.FullName)" }
}
```

记录实际 commit、Python/包版本、命令和日志。`--help` 仅验入口，不算真实方法执行。DNN 入口还应在已配置的 torch 环境检查，不能用未安装 torch 的 import 失败推断模型未迁移。

### B. 原始来源与一次真实执行

在本地选已知可用的 Benchmark 被试，显式列出该次读取的全部文件（包括实际使用的外部 metadata），用新命令记录；文件列表依真实 loader 为准，不猜文件名：

```text
uv run python scripts/record_sources.py <data_root> <relative_file_1> <relative_file_2> ... --output <new_local_sources.json>
```

真实小样本建议先使用 1 名被试、3 个 block、1 个 0.5 秒窗口和 TRCA；3 个 block 留出一个后有两个训练 block。以下 `<...>` 需替换成本机实际路径/新名称：

```text
uv run python scripts/run_traditional_benchmark.py --data-root <data_root> --subjects 1 --blocks 1-3 --windows 0.5 --methods TRCA --workers 1 --task-name <new_verification_run> --source-manifest <new_local_sources.json>
uv run python scripts/validate_artifacts.py results/<new_verification_run> --output <new_audit_evidence.json>
```

保存预测和 manifest；用完全相同的计算参数追加 `--resume` 重跑，核对原预测行数和内容哈希不变、run ID 保留、执行记录追加。不要把校验日志写进同一新运行目录后再启动新 run；证据放 `results/_verification/`。

随后在独立新目录执行 TDCA 和 BETA/Wearable 中本机已有的一条路径，各自检查报告数据集、样本数与预测对应。用不改变参数的旧代码参考结果作逐预测对照；因源码基线曾缺路径别名，旧入口先按已有可工作的环境/版本运行，不能直接把 import 失败当作数值不一致。

### C. 把证据交回而不是只说“通过”

记录每项 `pass/fail/not_run`、代码 commit、来源清单指纹、配置、测试报告/产物路径和限制。真实原始数据、私有映射、源清单与模型留本地；不因本次公共代码提交而上传。

## 6. 明确保留的后续工作

M01–M09 各自的训练组合与自有 `_write()` 未在本轮改写；它们的逐试次产物接入要按各自 split 语义做，不能把 aggregate-only 当 classification 通过。
EEGNetMini 与 TRCA 的统一完整方法接入、其他 DNN 管线接入、数据 adapter 的真实 metadata/单位/时间核对、任务专有数学验证、通用科学比较判断及在线模拟仍不在本轮完成范围。

下一项应当是已有约定上的一个可用能力，而不是再次重排项目树。发现本轮规范实现问题时按 N01–N10 对应层修复，并重跑受影响回归，不重写无关算法。
