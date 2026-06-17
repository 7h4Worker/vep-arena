# CCA-Net Reproduction Notes

Date: 2026-05-22

Primary sources:

- Paper: `C:\Users\Admin\Desktop\papers-unread\CCA-Net_Zero-Shot_SSVEP_Classification_via_an_Integration_of_Canonical_Correlation_Analysis_and_Deep_Neural_Network.pdf`
- Official CCA-Net code: `D:\ProjData\_reference\Zero-shot-SSVEP-classification`
- Ensemble DNN baseline code/results: `D:\ProjData\_reference\Ensemble-of-DNNs`

## Feasibility

CCA-Net is reproducible in Python because the authors provide an official PyTorch implementation. It should not be mixed directly with the existing subject-specific DNN/TRCA-Net/TDCA Benchmark reports, because its main protocol is zero-shot cross-subject evaluation, not within-subject leave-one-block-out.

The right comparison group is:

- CCA-Net
- CSDuDoFN only
- modified FBtt-CCA only
- CCA / FBCCA / FBtt-CCA baselines
- Ensemble DNN zero-shot baseline
- optionally zero-shot SSVEPFormer

The existing DNN fine-tune results in VEP Arena remain useful, but they answer a different question: calibrated or subject-specific decoding.

## Benchmark Protocol From Paper And Code

Dataset: Tsinghua Benchmark SSVEP.

Channels:

- Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2
- MATLAB 1-based indices: 48, 54, 55, 56, 57, 58, 61, 62, 63
- Python 0-based indices in the official code: 47, 53, 54, 55, 56, 57, 60, 61, 62

Timing:

- Sampling rate: 250 Hz
- Cue offset: 0.5 s
- Visual latency: 0.14 s
- Crop start: 0.64 s after trial start
- Paper figure windows: 0.5 to 1.0 s in 0.1 s steps
- ITR selection time: window + 0.5 s gaze shift

Split:

- Leave-one-subject-out zero-shot.
- For target subject S, all other 34 subjects are source subjects.
- Train CSDuDoFN using all source-subject blocks.
- Test on all target-subject blocks.
- Report per target subject, per block, window-level mean, std, and SEM.

This differs from the earlier DNN/TRCA-Net/TDCA leave-one-block-out protocol.

## Preprocessing Choices To Record

CCA-Net contains two filter-bank conventions.

For the neural CSDuDoFN input branch:

- 3 subbands
- Passbands: 6-90, 14-90, 22-90 Hz
- Stopbands: 4-100, 10-100, 16-100 Hz
- Chebyshev type I, `cheb1ord`, passband ripple 0.5

For FBCCA / modified FBtt-CCA:

- 5 subbands
- Passbands: 5-90, 14-90, 22-90, 30-90, 38-90 Hz
- Stopbands: 3-92, 12-92, 20-92, 28-92, 36-92 Hz
- Chebyshev type I SOS filter, order 15, ripple 0.5
- Filter weights: `(idx + 1)^(-1.25) + 0.25`
- CCA harmonics: 5

For LST/GMLST:

- LST harmonics in the official config: 4
- GMLST computes one global LST coefficient per class from source-subject class averages.
- The same GMLST coefficients are applied to source data and target data.

## Model And Training

The official CSDuDoFN implementation is in `model.py`.

Key settings:

- PyTorch
- Adam learning rate: 0.0002
- Epochs: 100
- Batch size: 128
- Dropout: 0.1, 0.1, 0.95
- No user-specific fine-tuning on the target subject

Important code detail:

- `model.py` currently returns the DNN/raw branch score from `forward`.
- CCA-Net is assembled in `main.py` by normalizing and summing the neural score with CCA-family scores.
- Therefore reproducing only `model.py` is not enough; the score-fusion section in `main.py` is part of the method definition.

## Score Variants In Official Code

The official `main.py` produces multiple variants:

- `CNN`: neural branch only
- `FBCCA`: sine/cosine reference template
- `FBCCA_new`: subject-specific template from source data
- `FBTTCCA`: transfer-template CCA with sine/cosine reference
- `FBTTCCA_new`: modified transfer-template CCA with source-derived template
- `CNN+FBCCA`
- `CNN+FBTTCCA_new`
- `CNN+FBCCA_new`
- `CNN+FBTTCCA_new+FBCCA_new`
- `CNN+FBTTCCA_new+FBCCA`
- `CNN+FBCCA+FBTTCCA_new+FBCCA_new`, the final combined CCA-Net score in the script

The paper emphasizes CSDuDoFN, modified FBtt-CCA, and their integration, so the report should keep these variants visible rather than hiding them behind one method name.

## Ensemble DNN Status

Our earlier DNN result is not the zero-shot Ensemble DNN baseline.

The official Ensemble-of-DNNs repository includes saved Benchmark results in:

`D:\ProjData\_reference\Ensemble-of-DNNs\Results\Benchmark_ensemble_results.mat`

I exported those saved results into VEP Arena:

- `D:\ProjData\proj_python\vep_arena\results\external_baselines\ensemble_dnn_benchmark_official_summary.csv`
- `D:\ProjData\proj_python\vep_arena\results\external_baselines\ensemble_dnn_benchmark_official_subject_block.csv`

These are external official baseline results, not a local retraining run.

## Recommended VEP Arena Integration

Add a separate protocol:

```text
zero_shot_leave_one_subject_out
```

Keep it separate from:

```text
subject_leave_one_block_out
```

The evaluator can still share the same output schema:

```text
method, dataset, protocol, subject, block, window, accuracy, itr, samples
```

For CCA-Net, the initial implementation should wrap the official code but replace hardcoded paths and globals with a clean config object. The first runnable target should be:

```text
dataset: benchmark_9ch
window: 0.5
target_subjects: 1-3 smoke test, then 1-35 full sweep
methods: CNN, modified FBtt-CCA, final CCA-Net
```

After the smoke test matches the official code behavior, run the full paper sweep for windows 0.5 to 1.0.

