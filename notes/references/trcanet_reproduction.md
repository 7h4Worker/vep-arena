# TRCA-Net Reproduction Notes

Reference:

- Paper: `TRCA-Net: Using TRCA filters to boost the SSVEP classification with convolutional neural network`
- Journal: Journal of Neural Engineering, 2023
- DOI: `10.1088/1741-2552/ace380`
- Official code: `D:/ProjData/_reference/TRCA-Net`

## What TRCA-Net Does

TRCA-Net is a hybrid method:

1. For each subject and each train/test split, train TRCA/eTRCA spatial filters
   from the subject's training blocks.
2. Project every raw SSVEP trial through the learned TRCA filters.
3. Convert each trial into an image-like tensor:

```text
class_filters x samples x subbands
```

For Benchmark, this is typically:

```text
40 x samples x 3
```

4. Train a CNN close to the DNN-SSVEP architecture on these transformed trials.
5. Optionally use two-stage training:

```text
global training across subjects -> subject-specific fine tuning
```

## Official MATLAB Settings

For `Bench` in the official `main_our.m`:

- subjects: 35
- blocks: 6
- classes: 40
- sampling rate: 250 Hz
- channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
- subbands: 3
- filter low cutoffs: 8, 16, 24 Hz for the DNN filters
- TRCA filterbank passbands: 6, 14, 22 Hz to 90 Hz
- first-stage epochs: 500
- transfer-global epochs: 1000 in some branches
- subject fine-tune epochs: 1000 if transfer is enabled
- learning rate: 1e-4
- weight decay/L2: 1e-3
- batch size: 100 for global training
- subject fine-tune batch size: `classes * train_blocks`
- dropout first stage: 0.1
- dropout second stage: 0.6
- final dropout: 0.95

## Arena Protocol

The first Arena reproduction should be:

```text
dataset: Tsinghua Benchmark SSVEP
channels: 9 classical occipital/parietal channels
protocol: subject-specific leave-one-block-out
windows: 0.2 to 1.0 s
comparison: DNN, SSVEPFormer, FBTRCA, TDCA, TRCA-Net
```

For practical verification, use this staged plan:

1. `smoke`: subject 1, block 1, 0.4 s, few epochs.
2. `window_0.4`: all subjects, all blocks, 0.4 s.
3. `full_windows`: all windows 0.2 to 1.0 s.

## Implementation Notes

TRCA-Net should not reuse the normal DNN input directly. The adapter must
produce TRCA-projected inputs:

```text
raw EEG -> subject-specific TRCA filters -> projected tensor -> CNN
```

The CNN can reuse the DNN module shape with:

```text
channels = 40 TRCA filters
subbands = 3
samples = window_samples
```

The important difference is that the "channel" axis is no longer EEG channels;
it is the bank of TRCA spatial filters.

## Caveats

The official repository is MATLAB-only. The PyTorch version should be validated
against the MATLAB tensor shapes and against TRCA-only accuracy before long
training runs.

The official code's default `dataset='data_our'`; Benchmark reproduction
requires setting `dataset='Bench'` behavior explicitly.
