# Visual BCI / VEP Foundation Model 对话记录

记录日期：2026-06-23
用途：保留当前关于“面向视觉 BCI / VEP 响应态的 foundation model”的算法设想对话记录。
说明：本文档以对话记录为主，不整理成正式 proposal，不额外删改对话中的核心表述。

## 当前打开的参考页面

- [Towards Neural Foundation Models for Vision: Aligning EEG, MEG and fMRI Representations for Decoding, Encoding and Modality Conversion](https://arxiv.org/abs/2411.09723)

## 相关参考链接

- [LaBraM: A Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://arxiv.org/abs/2405.18765)
- [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://openreview.net/forum?id=lvS2b8CjG5)
- [EEGPT GitHub](https://github.com/BINE022/EEGPT)
- [BIOT GitHub](https://github.com/ycq091044/BIOT)
- [CBraMod OpenReview](https://openreview.net/forum?id=NPNUHgHF2w)
- [CBraMod GitHub](https://github.com/wjq-learning/cbramod)
- [ST-EEGFormer GitHub](https://github.com/LiuyinYang1101/STEEGFormer)
- [EEG-FM-Bench GitHub](https://github.com/xw1216/EEG-FM-Bench)
- [EEG-FM-Compass GitHub](https://github.com/Dingkun0817/EEG-FM-Benchmark)
- [TRIBE v2 paper page](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- [TRIBE v2 GitHub](https://github.com/facebookresearch/tribev2)
- [THINGS Initiative](https://things-initiative.org/)
- [THINGS-EEG paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748587/)
- [Tsinghua Benchmark SSVEP dataset paper](https://pubmed.ncbi.nlm.nih.gov/27849543/)
- [BETA SSVEP benchmark dataset](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00627/full)
- [RSVP / P300 BCI dataset collection](https://springernature.figshare.com/collections/EEG_Dataset_for_RSVP_and_P300_Speller_Brain-Computer_Interfaces/5769449)
- [Keras SSVEPFormer example](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/eeg_bci_ssvepformer.py)

## 对话记录

### 用户

当前项目正有一个在跑的dnn的实验，我正在做比如vep arena的迁移验证。但是我发现一个问题啊，这个dnn用于ssvep的算法，这个cnn的设计基本上对于ssvep这种有不同时间长度窗口考评要求的实验得一直重新跑预训练模型，这对吗，实际上这个比较大的问题吧，我理解实际上比如是按照2秒窗口训练，我可以测试0.2s 0.4s等等长度的。但是这个网络的结构是本身就不是这样设计的是吗。如果可以的话你也可以找一下eegnet，这个应该有很多开源参考，如果有和ssvep实验对应的部分也可以帮我搜寻和总结一下信息

### 用户

那这个本身就是一个基础性的问题。近年也有不少transformer或者宣称zero shot的一些nn基础的ssvep算法模型，我印象里比如cca net os-ssvep，然后transformer这里st eegformer这个应该是比较通用的一个，你可以看看他们有没有开源信息吗，针对ssvep分类头的部分帮我调研一下

### 用户

实际上有没有这种NN为基础的算法呢，就是能做到这个变窗口识别的。就是这个就比较拉啊，因为实际上ssvep的动态系统或者异步系统难免有这种情况，而且存在就是需要考评对ssvep的不同时长这个是和vep的动态响应有关的，实际上每个受试者的最优响应可能不同，如果算法本身非常依赖固定窗口，也不能通过长窗口适应短窗口，这个方向对于实际bci系统缺陷很大，cca或者trca系列是没有这种问题的，nn这种类型就只能全都训练一边？实际成本等等是很高的吧

### 用户

是的，但是我之所以跟你提这个zero shot也是这里有一个关键问题，之所要用NN也是他的大规模数据的训练输入，跨域融合，即使对于单人，我们实际上需要NN带来的模型化学习，这个模型的概念在这里不仅仅是神经网络，空间滤波器方法本身也是目的于得到对于这个表征的校准，但是这个子空间的校准是很容易便宜的，我们一定程度上不知道什么东西是不变的什么是一直改变的，哪一部分大量贡献信息哪一部分其实只是观测扰动（从分类算法角度，当然直觉上我们知道ssvep的时序信号结构，也有频谱分析手段），所以我关注到DW这个问题就是在于，原来的仅仅能适应定长窗口的ssvep识别实际上有很大的问题，对于特定数据集的拟合仍然无法克服比如trca的长时间应用校准便宜问题，或者bci盲的问题，或者个别场景ssvep效果失效的问题，如果只是发挥了神经网络的拟合作用，对于vep这种结构鲜明，相应原理明确的信号来说意义不够。我大概是这种想法的

### 用户

是的，所以为什么现在foundation model也做不到这个呢，如果st eegformer的vit形式不行，labram的codebook或者你可以找一下eegpt，他们的结构做法能够适应吗，可能他们现在对vep的分类都是线性头没研究这个问题？

### 记录要点

- 固定窗口 CNN / EEGNet / DNN-SSVEP 往往使用 `Flatten(time) -> Linear` 分类头，checkpoint 与输入样本长度绑定。
- SSVEP / VEP 的实际 BCI 系统需要变窗口识别、动态停止、异步判断、短窗效率评估。
- CCA / TRCA 系列天然能在不同窗口上重算或累积证据，而很多 NN 方法只是定长分类器。
- zero-shot / few-shot / foundation model 如果只改线性分类头，无法自动解决 VEP 动态窗口和刺激条件化问题。
- 更合理的模型形式应从 `EEG window -> class` 改成 `EEG prefix + stimulus candidate -> evidence score`。

### 用户

ssvepformer的设计是如何的呢，好像现在新的工作直接transformer结构的ssvep的也不多，有一些dual branch之类的东西

### Codex

SSVEPFormer 的设计其实有点“名不完全副实”：它不是标准 ViT/Transformer 那种 `patch token + positional embedding + multi-head self-attention`，更像是一个 **频域输入的 Transformer-style / MLP-Mixer-style 网络**。

它的主流程大概是：

```text
EEG 时域窗口
  -> FFT 到固定频带的复频谱 real/imag
  -> 1x1 Conv 做 channel combination
  -> 两层 Encoder: Conv1D spectral mixing + Channel-wise MLP + residual
  -> Flatten + MLP 分类头
```

关键点是：

1. **输入不是原始时序为主，而是复频谱**
   原论文说它用 SSVEP 的 frequency spectrum 作为输入，同时建模 spectral 和 spatial 信息，并扩展了 filter-bank 版本 FB-SSVEPFormer。见 [arXiv 摘要](https://arxiv.org/abs/2210.04172)。

2. **FFT 是固定频率网格**
   本地实现里是 `nfft = fs / resolution`，比如 `250 / 0.25 = 1000`，然后截取 8-64 Hz，拼接 real 和 imag。这样 raw window 可以是不同长度，但 FFT 输出维度固定。这和普通 DNN/CNN 的 `Flatten(time)` 不一样，后者 checkpoint 直接被 sample length 绑死。

3. **它没有真正的 multi-head attention**
   Keras 的复现说明得很直白：SSVEPFormer 没有 positional embedding，也没有 multi-head attention，而是用 `Conv1D(kernel=31)` 和 MLP 残差块代替 Transformer encoder 的 attention/FFN 结构。见 [Keras SSVEPFormer 示例](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/eeg_bci_ssvepformer.py)。

4. **分类头仍然是普通闭集分类头**
   最后还是 `Flatten -> Dense -> Dense(num_classes)`。所以它不是参考信号条件化模型，也不是 CCA/TRCA 那种“给任意候选频率/相位算匹配分数”的结构。它能做跨被试闭集分类，但不能自然扩展到新频率、新 phase code，除非重新定义分类头或训练任务。

所以对你关心的 DW/变窗口问题，我会这么判断：**SSVEPFormer 比原始 DNN-SSVEP 更适合做单 checkpoint 变窗口测试，因为它的输入维度由 FFT 网格决定，不直接由原始窗口样本数决定；但它并没有真正解决动态窗口识别问题。** 短窗频谱分布、泄漏、SNR、相位稳定性都会变，训练时如果只喂 1s/2s，它未必能可靠泛化到 0.2s/0.4s。

你说“新的直接 Transformer 的 SSVEP 不多”这个判断也对。SSVEP 里很多所谓 Transformer 工作其实是 Conformer、PoolFormer、CNN-Transformer hybrid 或 dual-branch attention。原因很现实：SSVEP 物理结构太强，频率/相位/谐波/空间滤波这些 inductive bias 非常有效；纯 self-attention 在小数据、强跨被试扰动、短窗口条件下不一定占便宜。

dual-branch 这类方向大概有几种：

- **real/imag 双分支**：比如 dual-branch frequency-domain attention，把复频谱的实部和虚部分开建模，再约束或融合注意力。这比纯时域 Transformer 更贴近 SSVEP 的相位/频率结构。
- **time/frequency 双分支**：一个分支看时域瞬态/相位，一个分支看频谱/谐波，再融合。
- **temporal/spatial 双分支**：类似 DBConformer 这种，一个分支建 temporal，一个分支建 spatial/channel；更通用 EEG，不一定是 SSVEP 专用。代码见 [DBConformer repo](https://github.com/wzwvv/DBConformer)。
- **signal/reference 双分支**：ConvCA 这类其实我觉得对 SSVEP 更“正宗”，因为它保留了 CCA 思路：EEG 分支和参考模板分支分别编码，再算相关/相似度。这类结构天然更接近变窗口和 zero-shot 目标。

我的直觉是：SSVEPFormer 可以作为“频域 NN baseline”放进 VEP Arena，但如果目标是实际动态系统，真正有研究价值的不是再做一个更大的 Transformer，而是做：

```text
EEG prefix encoder + stimulus/reference encoder + similarity/evidence head
```

也就是让模型输出 `score(f, phase, t)`，而不是固定 `class_id`。这个方向才真正继承 CCA/TRCA 的变窗口优势，同时引入 NN 的跨域表征学习能力。

### 用户

所以如果深入思考这个模型设计的问题，如何设计一个能够适应vep的这种nn模型呢

### Codex

我会把设计目标定成：

> **不是 `EEG window -> class`，而是 `EEG prefix + stimulus candidate -> evidence score`。**

这样 VEP/SSVEP 模型才会从结构上适应变窗口、异步判断、跨被试校准和新刺激条件。

**核心设计**
模型输入应该是：

```text
x[0:t]        当前 EEG 前缀
r_k[0:t]      第 k 个候选刺激的参考序列
subject_ctx   受试者/会话/少量校准信息
```

输出不是直接 `40-class logits`，而是：

```text
score_k(t) = 当前 EEG 前缀支持候选刺激 k 的证据
```

也就是：

```text
for each candidate k:
    score_k(t) = NeuralMatch(EEGEncoder(x[0:t]), StimulusEncoder(r_k[0:t]), subject_ctx)
```

这比普通 CNN/Transformer 分类头重要得多。普通 NN 的问题是学了一个定长分类映射；这个设计学的是“脑电响应和刺激结构之间的匹配关系”。

**模型骨架**
我觉得比较合理的是一个 **Neural CCA/TRCA hybrid**：

```text
EEG x[0:t]
  -> filterbank / complex spectral / temporal conv
  -> learned spatial filters
  -> variable-length encoder
  -> EEG representation

Stimulus candidate r_k[0:t]
  -> sin/cos harmonics, phase, code sequence, luminance trace
  -> reference encoder
  -> stimulus representation

EEG rep + stimulus rep
  -> correlation / cross-spectrum / bilinear similarity / cross-attention
  -> score_k(t)
```

这里的关键是 stimulus branch。SSVEP 不是自然图像分类，候选类别本身有物理结构：频率、相位、谐波、码序列、屏幕刷新、视觉延迟。模型应该把这些作为条件输入，而不是让分类头死记 40 个 label。

**变窗口能力**
结构上要避免这些东西：

```text
Flatten(time) -> Linear
fixed Samples
fixed positional embedding length
only trained at one window
```

要用这些：

```text
temporal conv / relative-position Transformer
adaptive pooling / attention pooling
complex demodulation / STFT / fixed frequency grid
prefix training
multi-window supervision
```

训练时每个 trial 不只取一个窗口，而是随机采样多个前缀：

```text
t = 0.2, 0.3, ..., 2.0
loss = CE(score(t), label) + consistency + calibration loss
```

但 consistency 不能粗暴要求 0.2s 一定像 2s 一样自信。更合理是让模型学：

```text
短窗：可以不确定，但不能乱自信
长窗：证据逐渐收敛
```

所以还需要 uncertainty / rejection / stopping head：

```text
p_k(t), confidence(t), stop(t)
```

在线系统就可以：

```text
如果 max p_k(t) - second p_k(t) > threshold:
    输出
否则继续积累 EEG
```

**受试者适应**
真正实用的 VEP NN 还应该有一个很轻的 subject adapter。不是每个窗口重训一个模型，而是：

```text
大模型共享参数
少量校准 trial -> subject embedding / prototype / adapter
```

可选实现：

```text
subject embedding
low-rank adapter
conditional batch/layer norm
prototype memory
learned subject-specific spatial filter
```

这样它和 TRCA 的“便宜校准”靠近：主体模型学不变结构，少量参数吸收个体差异。

**分类头怎么做**
我最不建议继续做：

```text
representation -> Linear(40)
```

更建议做：

```text
representation x reference_k -> score_k
```

候选数可以是 40，也可以是新频率、新相位、新 code。这个头才有 zero-shot / open-set / dynamic-window 的可能。

SSVEP 下可以先做三种 score：

```text
1. learned CCA score
2. cross-spectrum phase-locking score
3. neural bilinear similarity score
```

最后融合：

```text
score_k = a * cca_like + b * spectral_match + c * neural_match
```

这样不会完全丢掉 CCA/TRCA 的优势，也能让 NN 学跨域扰动。

**最小可行版本**
如果要在 VEP Arena 里做一个务实原型，我会分三步：

1. **DNN-AdaptivePool baseline**
   先把 fixed flatten 去掉，证明“结构不绑窗口”本身能不能改善。

2. **Frequency-grid SSVEPFormer baseline**
   用固定 FFT/STFT grid，加随机窗口训练，看单 checkpoint 对 0.2-2.0s 的曲线。

3. **Reference-conditioned Neural CCA model**
   EEG branch + stimulus branch + correlation head。这个才是我认为真正有研究价值的主线。

一句话总结：
**适合 VEP 的 NN 不应该只是更大的 EEG 分类器，而应该是一个刺激条件化、变窗口证据累积、可轻校准的神经匹配模型。**这才对应你说的实际 BCI 系统需求。

### 用户

我感觉这个想法主要是先做算法的设计。你觉得呢，或者是不是全新架构配合nn比较好呢，或者是先以labram这样的模型进行baseline调整呢，总的来说我认为是做一个适配VEP的foundation model。现在也有面向MI的这种工作，你可以先调研一下，以及几个目前的eeg foundation model角度的benchmark文章，就是各种考评对比的

### Codex

我觉得不是二选一。更稳的路线是：

**先用 LaBraM/EEGPT/CBraMod/ST-EEGFormer 做强 baseline，但研究主线要是全新 VEP 适配架构。**
也就是：基础模型当参照物和初始化，不把“改一个 linear head”当最终方案。

我查了一圈，近年的结论其实挺支持这个判断。

**已有 EEG FM 的共识**
现在 EEG foundation model 大致分几类：

- **LaBraM**：channel patch + VQ neural tokenizer + masked code prediction，约 2500 小时、20 个数据集预训练。[论文/代码](https://arxiv.org/abs/2405.18765)
- **EEGPT**：masked dual self-supervised，强调 spatio-temporal representation alignment，官方说是 10M+ 参数、通用 EEG feature extractor。[OpenReview](https://openreview.net/forum?id=lvS2b8CjG5) / [GitHub](https://github.com/BINE022/EEGPT)
- **BIOT**：把不同通道、长度、缺失值的 biosignal 转成一致的 sentence-like structure，偏“异构格式统一”。[GitHub](https://github.com/ycq091044/BIOT)
- **CBraMod**：criss-cross transformer，空间和时间依赖分开建模。[OpenReview](https://openreview.net/forum?id=NPNUHgHF2w) / [GitHub](https://github.com/wjq-learning/cbramod)
- **ST-EEGFormer**：ViT/MAE 风格，重点是公平 benchmark；官方框架评估 6 种 decoding protocols。[GitHub](https://github.com/LiuyinYang1101/STEEGFormer)
- **REVE**：2025 NeurIPS，60,000+ 小时、92 datasets、25,000 subjects，4D positional encoding，目标是任意长度和电极配置。[项目页](https://brain-bzh.github.io/reve/)

但 benchmark 的信号也很明确：**FM 不等于自动胜利**。
ICLR 2026 的 “Are EEG Foundation Models Worth It?” 结论很关键：fine-tune 后 FM 在数据丰富 population setting 里有优势，但 linear probing 很弱，数据稀缺场景经常不能显著超过 compact NN 或传统非神经方法。[OpenReview](https://openreview.net/forum?id=5Xwm8e6vbh)

EEG-FM-Bench/Compass 也类似：评估 BENDR、BIOT、CBraMod、EEGPT、LaBraM 等，结论包括 **linear probing 不够、专用模型仍然很强、模型越大不一定越好**。[EEG-FM-Bench](https://github.com/xw1216/EEG-FM-Bench) / [EEG-FM-Compass](https://github.com/Dingkun0817/EEG-FM-Benchmark)

**MI 方向的启发**
你提到 MI，很对。MIRepNet 这个路线值得关注，因为它不是“通用 EEG FM 直接套 MI”，而是明确做 **paradigm-specific foundation model**：统一 MI preprocessing、通道模板、masked token reconstruction + supervised MI classification，目标是少量 trial 快速适配。[MIRepNet GitHub](https://github.com/staraink/MIRepNet)

这对 VEP 很有启发：
**VEP 也应该做 paradigm-specific foundation model，而不是只借一个通用 EEG backbone。**

**VEP-FM 应该不一样在哪里**
通用 EEG FM 多数只看 EEG 本身：

```text
EEG -> representation -> linear/classification head
```

但 VEP/SSVEP/cVEP 的核心不是孤立脑电，而是：

```text
stimulus structure -> visual response -> EEG evidence
```

所以 VEP foundation model 应该至少多一个 stimulus branch：

```text
EEG prefix encoder
Stimulus/reference encoder
Subject/session adapter
Evidence matching head
Dynamic stopping head
```

也就是：

```text
score_k(t) = f(EEG[0:t], stimulus_k[0:t], subject_ctx)
```

这和 LaBraM/EEGPT 的普通分类头不同。它能自然支持：

- 变窗口识别
- 动态停止
- 新频率/新相位/新 code 的 zero-shot 或 few-shot
- cVEP/SSVEP/P300 统一建模
- 跨被试、跨设备、少校准

**我建议的研究路线**
第一阶段别急着训大模型，先做三个 baseline 层：

1. **FM backbone baseline**
   LaBraM / EEGPT / CBraMod / ST-EEGFormer frozen + linear head、full fine-tune、LoRA/adapters。
   目的：证明现有 FM 在 VEP Arena 上到底强不强。

2. **Multi-window baseline**
   同一个 checkpoint 训练多窗口，测试 0.2-2.0s。
   目的：直接验证你关心的 DW 问题。

3. **Reference-conditioned head baseline**
   不改大 backbone，只替换分类头为 `EEG representation + stimulus reference -> score`。
   这个是最关键的中间层，成本低，但概念已经从“分类器”变成“VEP evidence model”。

第二阶段再做真正的 VEP-FM：

```text
raw EEG / spectral tokens
+ stimulus trace / frequency / phase / code tokens
+ channel coordinate embedding
+ subject/session adapter
+ contrastive EEG-stimulus pretraining
+ prefix/evidence accumulation objective
```

预训练任务可以是：

- masked EEG reconstruction
- spectral/harmonic prediction
- EEG-stimulus contrastive alignment
- correct stimulus vs shifted/incorrect stimulus discrimination
- prefix consistency
- latency-shift robustness
- subject/session invariant learning

**我的判断**
如果目标是发一个有价值的方向，单纯“调 LaBraM 做 VEP baseline”不够；它适合作为对照和初始化。真正值得做的是：

> **VEP-specific foundation model：通用 EEG 表征 + 刺激条件化匹配 + 变窗口证据累积。**

这个方向比“再做一个 SSVEP Transformer 分类器”更站得住，也正好回应你最开始发现的那个基础问题：固定窗口 NN 不是工程小缺陷，而是模型设定错了。

### 用户

我确实是想做这个的，而且其实我觉得image方面的也可以纳入，就是包含视觉specific的，虽然现在的数据相对来说比什么癫痫 情绪少，但是这部分的数据比如说更加匹配视觉通路和皮层，目的非常明确的然后可以融合meg或者fmri，比现在有些直接做什么omni模型的工作我感觉要好不少吧，就已经有一些什么全模态工作了，但是我感觉挺烂的。我想做的这个我希望能够更好的模拟计算神经科学的一些成果出来，计算特性，以及特定subject的这个模型的差异，有点世界模型那个意思，或者PINN，其实我是感觉单纯的拟合好像差一点。当然我的主思路还是以transformer为主的，还是希望挑战一下scale law的

### Codex

我觉得你这个判断是对的，而且比“EEG omni model”更有抓手。

更准确地说，你想做的不是普通 EEG foundation model，而是：

> **Visual-pathway-specific neural foundation model**
> 学习视觉刺激、视觉皮层响应、个体差异和下游 BCI 行为之间的可计算映射。

这比“把 EEG、MEG、fMRI、文字、图像全都塞进一个 embedding 空间”更像科学模型。因为视觉系统本身有很强结构：视网膜/皮层层级、retinotopy、latency、frequency following、phase locking、semantic hierarchy、个体差异。这里有东西可以建模，不只是做 retrieval。

**我会这样定位**
不是：

```text
EEG -> label
EEG -> image embedding
all modalities -> shared embedding
```

而是：

```text
visual stimulus -> latent visual process -> EEG/MEG/fMRI response -> task/evidence/behavior
```

也就是一个 forward + inverse 都能做的模型：

```text
Encoding:  image/video/flicker/code -> predicted EEG/MEG/fMRI
Decoding:  EEG/MEG/fMRI -> stimulus/evidence/state
BCI:       EEG prefix + candidate stimulus -> dynamic evidence score
Subject:   few-shot calibration -> subject-specific visual response model
```

你说“世界模型 / PINN”其实很贴切。VEP 方向可以放入一些硬约束或软约束：

- latency：P1/N1/P300/SSVEP 相位延迟不是任意的
- frequency locking：SSVEP 对基频和谐波有明确结构
- visual hierarchy：低级视觉区偏局部边缘/运动，高级区偏对象/语义
- modality transfer：fMRI 空间准、EEG/MEG 时间准
- subject adapter：个体差异应被显式建模，而不是当噪声平均掉
- stimulus geometry：图像位置、闪烁频率、phase/code、刷新率都应该进模型

**为什么纳入 image/video 是好主意**
视觉刺激本身是天然“大数据接口”。EEG-VEP 数据少，但图像/视频 foundation model 已经学了大量视觉结构。你真正要学的是：

```text
视觉表征 -> 人脑视觉响应映射
```

而不是从 EEG 小数据里重新学视觉世界。

这和 TRIBE v2 这类工作很接近。Meta 的 [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) 是视频/音频/语言到 fMRI 响应的 brain encoding model，号称用 1000+ 小时 fMRI、720 subjects，能做 in-silico neuroscience；代码也开源了：[facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)。这说明“用 foundation model 做脑响应预测”已经是严肃路线，不是幻想。

但它和你想做的仍然不同：TRIBE v2 主要是 fMRI、自然刺激、宏观脑区预测；你关心的是 **视觉诱发、电生理、短窗动态、BCI 可用性、subject-specific 校准**。这正好是一个空位。

**我建议的架构方向**
可以叫暂定名：`VEP-FM` 或 `Visual Neural Response Model`。

```text
Stimulus Encoder
  image/video encoder: CLIP / DINOv2 / V-JEPA / SigLIP
  flicker/code encoder: frequency, phase, harmonics, cVEP sequence, refresh timing

Neural Encoder / Decoder
  EEG/MEG tokenizer: channel geometry + temporal patches + spectral tokens
  fMRI tokenizer: ROI/voxel/surface tokens + HRF/time tokens

Subject Model
  subject embedding
  few-shot adapter / LoRA / hypernetwork
  subject-specific latency and spatial filter parameters

Shared Latent Visual-Neural Space
  transformer backbone
  cross-attention between stimulus tokens and neural tokens

Heads
  EEG/MEG response prediction
  fMRI response prediction
  stimulus contrastive retrieval
  SSVEP/cVEP evidence scoring
  dynamic stopping / uncertainty
```

最关键的是不要只有 decoding head。要有 **response generation head**：

```text
given stimulus + subject -> predict neural response
```

这样它才像计算神经科学模型；否则又变成“拟合分类器”。

**训练任务**
我会混合这些 objective：

```text
1. masked neural modeling
2. stimulus-neural contrastive alignment
3. EEG/MEG/fMRI response prediction
4. modality conversion: EEG <-> MEG <-> fMRI latent
5. candidate stimulus evidence scoring
6. prefix consistency / dynamic window training
7. subject few-shot adaptation loss
```

SSVEP/VEP 这边尤其要加：

```text
EEG prefix + candidate stimulus -> score
```

这会把它和 CCA/TRCA 的物理逻辑接上。

**数据路线**
可以分三类数据：

- **视觉神经科学数据**：NSD、THINGS EEG/MEG/fMRI、BOLD5000、BOLD Moments、CNeuroMod/Algonauts。NSD 是 7T fMRI、8 个被试、大量自然图像，[官网](https://naturalscenesdataset.org/)；THINGS 有图像、行为和多种神经记录，[项目页](https://things-initiative.org/)。
- **VEP/BCI 数据**：Tsinghua Benchmark、BETA、Nakanishi、cVEP、P300 speller、你现在的 VEP Arena。
- **辅助视觉模型数据**：不直接需要脑数据，用 image/video FM 提供视觉先验。

已有一篇非常贴近你想法的工作是 [Towards Neural Foundation Models for Vision](https://arxiv.org/abs/2411.09723)，它尝试对齐 EEG、MEG、fMRI 和视觉刺激，用于 decoding、encoding、modality conversion。它可以当相关工作，但我觉得它还偏“共享表征/对齐”，离你说的 VEP 动态系统、subject model、BCI evidence 还有距离。

**scale law 怎么挑战**
你可以把 scale law 设计成严肃实验，而不是口号：

```text
scale axis:
  model size
  visual stimulus amount
  neural hours
  subject count
  modality count
  repeated-trial count

evaluate:
  neural response prediction
  cross-subject transfer
  few-shot subject adaptation
  SSVEP/cVEP dynamic-window accuracy
  zero-shot stimulus generalization
  known visual hierarchy recovery
```

一个很漂亮的问题是：

> 增加视觉刺激多样性、subject 数量、模型规模，是否会系统性改善 VEP 短窗证据估计和 subject-specific response prediction？

如果能画出 scaling curve，这个方向就会比普通 EEG benchmark 更有分量。

**我的判断**
你这个方向值得做，而且应该坚持 visual-specific。不要一开始就做“大而全 omni brain model”。更好的路线是：

```text
视觉通路 foundation model
+ 多模态神经响应预测
+ subject-specific adaptation
+ VEP/BCI 动态证据头
```

这条线既能和 TRIBE v2、SION、LaBraM/EEGPT 这些工作对话，又有你自己的核心问题：**能否建立一个可迁移、可校准、可解释、能做动态 VEP 决策的视觉脑响应模型。**

### 用户

所以我关注的就是直接和bci相关的，因为ssvep mvep p300以及rsvp，也有一定的高级皮层激活，但是很大一部分是低级层面的，那么低级层面其实bci的效率还是很高但是解码模型 解码方式很不统一或者还没找到很高的效率来提升，虽然已经有很多跨受试者跨session的工作（其实我觉得做的也一般，包括一些宣称zero shot的模型）。针对的就是这种情况，我们要设计一种精简的，面向这种响应态的模型，可以利用流形几何或者PINN的理念，就是我们应该认为人的视觉响应层是有共性又有差异的，这种差异应该又一种foundation model可以解决，这个是可预见的，人和人或者场景和场景的差别不会超出这种模式区域，应该能够高效的提供解码。尤其是有些ssvep的工作其实已经体现出来，0.4-0.6s即达到最大效率，这个时候基本上ssvep是度过前期稳定期之后没几个周期，我是隐隐觉得实际上算法能力的体现就是需要的周期很少就能完成估计。这种模态可以迁移到复杂一点的比如things eeg那种任务，很多生图模型之类的宣称确实有颜色 纹理等等特征被利用，但是他们又说不清楚，我们应该可以某种程度上辨别或者分层级的设计model来适应这种情况，高底层的组合，感受人所感受和关注的，并且结合某种特殊的vep，比如仅图像的红色被增强，仅图像的对比纹理被增强，这样的过程来带特定空间的激发。我说了很多可能有点乱，你可以整理一下

### Codex

你说的其实不乱，我觉得可以整理成一个很清楚的研究命题：

> **面向 BCI 的视觉响应态 foundation model：学习人类视觉系统在可控刺激下的共性响应流形，以及 subject/session 对这个流形的低维变形，从而高效完成 SSVEP、mVEP、P300、RSVP、cVEP 等视觉 BCI 解码。**

这和普通 EEG omni model 的区别非常大。你不是想“所有脑电任务都做一点”，而是抓住一个结构很强的子系统：**视觉刺激 -> 视觉通路 -> 可观测脑响应 -> BCI 决策**。

**核心思想**
可以把人的视觉 BCI 响应写成：

```text
stimulus program
  -> latent visual response dynamics
  -> subject/session transform
  -> EEG/MEG/fMRI observation
  -> evidence / decision
```

其中 `latent visual response dynamics` 是共性的，`subject/session transform` 是个体差异。你真正想学的是这两个东西，而不是只学 `EEG -> label`。

这就解释了你说的几个点：

- SSVEP 高效，是因为低级视觉通路的频率/相位锁定很强，算法如果足够好，确实应该用很少周期就完成估计。
- P300/RSVP 不是纯低级视觉，但它们仍然有视觉输入、注意、目标检测、语义加工的层级结构。
- THINGS-EEG 这类自然图像任务可以作为高级视觉响应补充，THINGS-EEG 本身有 50 subjects、1,854 concepts、22,248 images，适合补语义/对象层面的响应结构：[THINGS-EEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748587/)。
- EEG/MEG/fMRI 对齐不是为了“全模态炫技”，而是因为 fMRI 给空间，EEG/MEG 给时间，这对视觉响应模型很自然。你当前浏览的这篇 [Towards Neural Foundation Models for Vision](https://arxiv.org/abs/2411.09723) 就是用 EEG、MEG、fMRI 做视觉表征对齐、解码、编码和模态转换。

**模型应该是什么**
我会暂时叫它：

```text
VEP-RFM: Visual Evoked/Response Foundation Model
```

它不是普通 Transformer 分类器，而是一个轻量的刺激条件化响应模型：

```text
Stimulus Encoder
  image / video / flicker / code / event / color / texture / contrast

Neural Encoder
  EEG / MEG / fMRI tokens
  channel geometry
  time-frequency tokens
  latency tokens

Latent Visual Response Transformer
  low-level visual response
  mid-level feature response
  high-level semantic/attention response

Subject Adapter
  subject embedding
  latency shift
  spatial transform
  gain/phase adapter

Heads
  BCI evidence score
  dynamic stopping
  response prediction
  feature attribution
  modality conversion
```

最关键的输出不是单一分类：

```text
EEG -> class
```

而是：

```text
EEG prefix + candidate stimulus -> evidence score
stimulus + subject -> predicted neural response
neural response -> inferred stimulus feature / decision state
```

这样才同时服务 BCI 和计算神经科学。

**低层和高层怎么统一**
可以分层建模：

```text
Low-level:
  luminance, contrast, color, spatial frequency, flicker frequency, phase, motion

Mid-level:
  texture, contour, shape, object parts

High-level:
  object category, semantic relevance, targetness, attention, oddball
```

SSVEP/cVEP/mVEP 更多吃 low-level 和 temporal locking。P300/RSVP 则多了 targetness、attention、semantic recognition。THINGS-EEG 可以补高层视觉和对象语义。

你说“红色增强、纹理增强、对比增强”非常关键。这个可以变成模型训练里的 **controlled stimulus perturbation**：

```text
image
image + red boost
image + contrast boost
image + texture boost
image + flicker overlay
image + local region modulation
```

然后要求模型预测：

```text
神经响应哪里变了？
哪个 latency band 变了？
哪个 channel/source 变了？
BCI evidence 是否更强？
```

这样可解释性就不是事后 saliency map，而是实验设计里带进去的。

**PINN / 流形几何可以怎么放**
这里不一定要真的做传统 PINN，但可以有 PINN 风格约束：

```text
latency constraint:
  视觉响应不能任意早出现

frequency constraint:
  SSVEP evidence 应该锁定基频/谐波/相位

smooth manifold constraint:
  相近刺激、相近 subject adapter 的响应应连续变化

subject transform constraint:
  个体差异主要是低维变换，而不是完全换一个系统

causal perturbation constraint:
  红色增强应主要影响颜色相关响应，纹理增强应影响纹理/空间频率响应
```

这比“纯拟合”更像你说的世界模型。

**直接 BCI 的评价方式**
这个模型的 benchmark 不能只看分类准确率。要看：

```text
0.2-2.0s dynamic window curve
0.4-0.6s ITR peak
few-shot subject calibration
cross-session transfer
new frequency / new phase / new code zero-shot
adaptive stopping
failure rejection
subject-specific response prediction
feature perturbation consistency
```

SSVEP 数据可以先用 Tsinghua Benchmark 和 BETA。Tsinghua Benchmark 是 35 subjects、40 targets 的 SSVEP speller 数据：[PubMed](https://pubmed.ncbi.nlm.nih.gov/27849543/)。BETA 是 70 subjects、面向真实应用的大型 SSVEP benchmark：[Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00627/full)。P300/RSVP 可以接类似 55 subjects 的公开数据：[Figshare](https://springernature.figshare.com/collections/EEG_Dataset_for_RSVP_and_P300_Speller_Brain-Computer_Interfaces/5769449)。

**我会把你的想法压缩成一句主张**
> 视觉 BCI 不是一组分散任务，而是一族可控视觉刺激诱发的响应态。它们共享视觉通路的低维动力学结构，又通过 subject/session/stimulus condition 发生可校准变形。因此应该训练一个视觉响应 foundation model，用刺激条件化、流形约束和动态证据累积统一 SSVEP、mVEP、P300、RSVP 与自然图像 EEG。

这个方向是很有辨识度的。它不只是“做一个更大 EEG Transformer”，而是在说：**BCI 解码效率的瓶颈来自我们没有建模视觉响应生成机制。**
