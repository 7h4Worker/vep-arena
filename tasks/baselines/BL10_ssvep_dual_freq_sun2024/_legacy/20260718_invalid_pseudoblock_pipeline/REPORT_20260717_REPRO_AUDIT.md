# Sun2024 Reproduction Audit 2026-07-17

## Status

The current Arena run in `results/formal_full_20260717_occipital9` is a
preliminary implementation check, not a paper-level reproduction. The direction
is plausible because ebTRCA is consistently better than eTRCA/TRCA, but the
magnitude is clearly lower than the paper's reported offline/online results.

## Paper Facts Extracted From The PDF

- Acquisition: 64-lead Neuroscan EEG system, 10-10 layout, 1000 Hz original
  sampling rate.
- Display: circularly polarized 3D monitor with polarized glasses.
- 40-target offline protocol: 1 s cue, 2 s stimulation, 1 s dark interval, 5
  trials per target, 200 trials total, randomized order, leave-one-out
  cross-validation.
- 40-target online protocol: 0.6 s stimulation, 7 training trials per target,
  then 2 validation trials per target.
- ITR: data window plus 0.5 s target-searching time.
- Preprocessing: all data were downsampled from 1000 Hz to 250 Hz. For
  single-target data, ICA plus comb filtering was used. For 40-target
  offline/online data, the paper states that only comb filtering was used for
  computational efficiency.
- Algorithm: bTRCA separates 20 optimized stimulus combinations as group O and
  their 20 swapped counterparts as group S. The equations define filter
  matrices `W_O,i`, `W_S,i`, templates `X_O,i`, `X_S,i`, and target scoring via
  correlation after applying the target's filter matrix. ebTRCA stacks filters
  in an ensemble strategy analogous to eTRCA.
- Reported results: offline bTRCA peak individual ITR 338.32 bits/min and mean
  peak ITR 224.96 bits/min. Online average accuracy/ITR were 89.63% and 234.13
  bits/min over 10 participants.

## Current Arena Run

Command:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_formal.py --subjects all --windows 0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 --methods TRCA,ETRCA,BTRCA,EBTRCA --channels occipital9 --n-bands 5 --output tasks\ssvep_efficient_dual_frequency_sun2024\results\formal_full_20260717_occipital9
```

Observed summary:

- Completed 448 summary rows: 14 subjects x 8 windows x 4 methods.
- Mean accuracy across all windows: TRCA 0.5531, ETRCA 0.5687, BTRCA 0.5589,
  EBTRCA 0.7489.
- Mean 0.6 s accuracy: TRCA 0.3465, ETRCA 0.3309, BTRCA 0.4170, EBTRCA 0.5644.
- Mean 2.0 s accuracy: TRCA 0.6981, ETRCA 0.7414, BTRCA 0.6500, EBTRCA 0.8478.
- Mean per-subject peak accuracy: TRCA 0.6985, ETRCA 0.7418, BTRCA 0.6533,
  EBTRCA 0.8493.
- Mean per-subject peak ITR: TRCA 86.43, ETRCA 92.12, BTRCA 85.63, EBTRCA
  133.74 bits/min.

## Data Structure Mismatch

The paper's offline description says 5 trials per target and 200 trials total.
The local files are mixed:

- Subjects 1-9: 320 usable 40-target trials in 8 blocks.
- Subjects 10-14: 200 usable 40-target trials in 5 blocks.

The current runner uses all available blocks with leave-one-block-out
cross-validation. This is not the same as a strict reproduction of the paper's
stated 200-trial offline protocol.

## Important Gaps

1. Channel count mismatch. The paper used all 64 leads; the full Arena run used
   `occipital9`. bTRCA is explicitly motivated by spatial differences between
   interocular resources, so reducing to occipital channels can suppress part
   of the intended signal.
2. Filtering mismatch. The first completed full run used notch filtering plus
   Butterworth filter-bank bands `(8-90), (16-90), (24-90), (32-90), (40-90)`.
   A follow-up `--preprocess comb_filterbank` path has been added, but the
   exact author comb-filter parameters are still unknown from the public PDF.
3. Latency crop uncertainty. The current runner defaults to `--onset-shift
   0.14`. The reviewed paper text does not explicitly document this 140 ms
   crop, so it is an Arena assumption until author code or supplementary notes
   confirm it.
4. bTRCA matrix mismatch. The paper writes filter matrices and scores after
   applying `W * Y * W^T`. The current `BTRCA` implementation stores one
   leading spatial vector per target, plus an eTRCA-like ensemble option. This
   may be a useful approximation, but it should not yet be called exact
   author-code reproduction.
5. O/S group assignment uncertainty. The public Excel codebook has trigger and
   left/right frequency columns, but no explicit optimized/swapped group color
   label. The current implementation assigns the lower trigger in each swapped
   frequency pair to group O.

## Immediate Diagnostics

- All-channel subject 1, 0.6 s, 5-band, two-fold smoke improved EBTRCA to
  0.8500, compared with 0.7500 in the earlier comparable occipital9 smoke.
- All-channel subject 1 with `--onset-shift 0` reduced EBTRCA to 0.7250, so the
  current 0.14 s crop helps this smoke case, but still needs paper/source-code
  confirmation.

## Recommended Next Sun2024 Pass

1. Treat `formal_full_20260717_occipital9` only as a diagnostic baseline.
2. Add a paper-audit run using all EEG channels, with both 5-block-only and
   all-block variants where applicable.
3. Resolve the exact O/S grouping from the article figure, supplementary
   material, or author code.
4. Compare the new comb-filter preprocessing path against the original
   notch-plus-filter-bank approximation.
5. Implement or validate a matrix/multi-filter bTRCA scoring path against the
   paper equations.
6. Re-run a focused diagnostic grid before launching another full overnight
   run: channels `all` vs `occipital9`, onset `0` vs `0.14`, comb filter vs
   current filter bank, vector bTRCA vs matrix bTRCA.

## Follow-Up Implementation

`run_formal.py` now supports `--preprocess comb_filterbank`, `--comb-f0`,
`--comb-q`, and `--max-blocks`. The intended paper-audit rerun is all channels,
comb-filter preprocessing, `--max-blocks 5`, and the paper's Fig. 6 window
grid from `0.2` to `2.0` seconds so the local 8-block subjects match the
paper's stated 5 trials per target before cross-validation.
