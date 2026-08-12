# Ke et al. 2025 Binocular AR SSVEP 原文对齐记录

日期：2026-07-08

来源：`G:/Ke 等 - 2025 - Dataset of binocularly coded steady-state visual evoked potentials recorded with an augmented realit.pdf`

## 任务定位

这个文件用于把 binocular AR 数据集的正式论文协议先沉淀到 task 内，避免后续直接按 Arena 默认 SSVEP 配置跑 TRCA 时偏离原文。当前只做协议对齐和后续执行规划，不声称已经复现实验结果。

## 数据与采集

- 数据集包含 24 名受试者，原始数据按 BIDS-like 结构组织，单受试者 zip 内包含 `events.tsv`、`channels.tsv`、`electrodes.tsv`、`eeg.json`、`eeg.set`、`eeg.fdt`。
- EEG 使用 32 通道系统采集，采样率 1024 Hz，参考电极为 Cz，地电极位于 Fz 与 FPz 中间。
- 论文后续 EEG 响应与分类分析重点使用 10 个 parieto-occipital 电极：`PO7, PO5, PO3, POz, PO4, PO6, PO8, O1, Oz, O2`。
- 本地 smoke 之前的 13ch preset 是探索用通道；代码默认已调整为论文 10ch，13ch 作为扩展 preset 保留。

## 范式与任务

论文包含三个实验：

- Experiment 1：binocularly congruent SSVEP，低频 LF 为 8-15 Hz，中频 MF 为 23-30 Hz，频率步长 1 Hz，相位为 0 到 1.75π、步长 0.25π。
- Experiment 2：四种双目编码条件：`SFSP`、`SFDP`、`DFSP`、`DFDP`，即同频同相、同频异相、异频同相、异频异相。
- Experiment 3：`DFDP-1`、`DFDP-3`、`DFDP-5`，考察双眼频率差为 1/3/5 Hz 且相位相反时的情况。

每个 condition 有两个 sub-session，中间休息约 5 分钟；每个 sub-session 包含 10 个 block，每个 block 8 个 trial，每个 trial 对应一个目标。trial 时序为 1 s cue 加 3 s flicker/gaze 阶段。

## 预处理

论文使用 EEGLAB 完成预处理：

- 49-51 Hz notch filter；
- 5-95 Hz band-pass filter；
- 按事件 trigger 提取 epoch；
- 因视觉延迟，epoch 时间窗设为 `[-0.5, 3.14] s`，其中 0 表示 flickering 开始。

官方代码已在 2026-07-08 下载到本地 metadata cache。`LoadData.m` 将数据 epoch 到 `[-0.5, 3.14] s`；分类脚本中 `len_delay_s = 0.5 + 0.14`，并使用 `segment_data = len_delay_smpl+1 : len_delay_smpl+len_gaze_smpl`，因此分类段等价于从 flicker onset 后 0.14 s 开始裁剪。

## 分类与指标

论文分类比较了四个算法：

- FBCCA
- SSCOR
- eTRCA
- TDCA

正式分类协议：

- leave-one-block-out cross-validation；
- 数据长度从 0.1 s 到 3.0 s，步长 0.1 s；
- 所有算法均使用 filter bank；
- filter bank 数量：除 MF 使用 3 个子带外，其余条件使用 5 个子带；
- filter bank 权重沿用 Chen FBCCA 系列设置；
- 1 s accuracy 用于条件间柱状/矩阵对比；
- accuracy 与 ITR 随时间窗变化用于主结果曲线。

ITR 脚本使用 `len_sel_s = len_gaze_s + len_shift_s`，其中 `len_shift_s = 1`。因此官方分类代码的 ITR 分母为 `window + 1.0 s`，没有额外加入 0.14 s 视觉延迟。

## 对 Arena TRCA 任务的实现要求

第一版正式 TRCA task 建议按下列顺序推进：

1. 解析每个 subject/session/task 的 `events.tsv`，用 `trial` 或事件顺序重建 block。当前 Arena task 默认合并两个 session，因此每个 condition 为 20 blocks。
2. 用 `events.tsv` 的 onset 从连续 `.fdt` 中切窗，并生成 `target x channel x sample x block` 形式的中间数组。这个数组可以作为 TRCA/eTRCA/TDCA 的共同输入，但不要长期保存巨大原始副本。
3. 预处理使用论文滤波：49-51 Hz notch 与 5-95 Hz band-pass。滤波和裁剪顺序要在报告中固定。
4. 使用 10ch paper preset 作为主结果；可选扩展 13ch 或全 32ch，但必须作为附录，不混入论文复现主曲线。
5. 跑 eTRCA/TRCA 第一轮 smoke：单 subject、`LF,DFDP,DFDP1`、窗口 `0.1,0.5,1.0,3.0`。
6. smoke 通过后跑全窗口 `0.1:0.1:3.0`，输出 accuracy/ITR 曲线、1s 条件对比、混淆矩阵和 task 级中文报告。

## 当前本地状态

- 数据入口和 smoke 已在 `vep_arena.data.binocular` 与本 task 内建立。
- 2026-07-08 已通过 `download_figshare_epoch.py` 补齐 epoch zip，当前本地 `sub-001` 到 `sub-024` 均完整。
- 2026-07-08 已下载官方代码 `official_code_26768287_v8.zip`，并据此确认分类 crop 起点与 ITR 分母。
- `.fdt` 读入方式当前采用样点交错 reshape，时序连续性检查通过；正式复现前仍建议用 MNE/EEGLAB 读取同一文件做一次数值对照。
- 当前 smoke 只验证数据结构、事件、频谱和通道，不运行分类器。
- 后续需要继续核对的是 SciPy 预处理与 EEGLAB `pop_eegfiltnew` 的数值差异；当前 Arena task 已按官方分类代码对齐 crop、block 和 ITR 口径。
