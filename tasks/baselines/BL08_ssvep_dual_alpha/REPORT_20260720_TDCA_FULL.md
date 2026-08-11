# Dual-Alpha Arena TDCA Full Report (2026-07-20)

## Status

The Arena-native Dual-Alpha TDCA extension is complete for all 35 subjects,
three paradigms, and ten analysis windows. It is an Arena method comparison,
not an author-provided TDCA reproduction.

Result directory:

```text
tasks/ssvep_dual_alpha_baselines/results/tdca_full_occipital9_20260720
```

## Configuration And Completion

| Item | Value |
| --- | --- |
| Subjects | 1-35 |
| Paradigms | Checkerboard Arrangement, Binocular Vision, Binocular-Swap Vision |
| Channels | Common occipital 9 for every paradigm |
| Windows | 0.2-2.0 s, 0.2 s step |
| CV | Five-block leave-one-block-out |
| Methods | TDCA and same-condition ETRCA |
| Filter bank | First five public Dual-Alpha TRCA subbands |
| TDCA | 5 harmonics, `n_delay=5`, `n_components=8` |
| Reference phase | Zero; public codebook contains frequencies only |
| Workers | 14, BLAS threads capped at one |
| Main run time | About 18.3 minutes |

Completion checks:

| Artifact | Observed | Expected |
| --- | ---: | ---: |
| Latest complete units | 105 | 105 |
| `summary.csv` rows | 2,100 | `35 x 3 x 10 x 2` |
| `trials.csv` rows | 420,000 | `35 x 3 x 10 x 2 x 5 x 40` |
| `aggregate_summary.csv` rows | 60 | `3 x 10 x 2` |
| Manifest | `complete` | `complete` |

Every source epoch is exactly 500 samples. At the 2.0 s boundary, TDCA uses
five zero-padded tail samples for training delay augmentation, matching the
existing test-side delay-padding behavior. No shorter window is padded.

## Accuracy Results

| Paradigm | Method | 0.2 s | 0.4 s | 0.6 s | 1.0 s | 2.0 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Checkerboard | TDCA | 54.40% | 71.74% | 78.93% | 86.89% | 93.07% |
| Checkerboard | ETRCA | 54.49% | 73.59% | 81.17% | 87.70% | 93.59% |
| Binocular Vision | TDCA | 47.00% | 70.23% | 78.84% | 85.19% | 90.30% |
| Binocular Vision | ETRCA | 46.70% | 67.70% | 75.07% | 82.16% | 88.03% |
| Binocular-Swap | TDCA | 40.44% | 62.24% | 69.76% | 77.34% | 85.26% |
| Binocular-Swap | ETRCA | 40.63% | 58.94% | 65.89% | 73.57% | 81.76% |

![Accuracy curves](results/tdca_full_occipital9_20260720/figures/tdca_etrca_paradigm_curves.png)

At 0.6 s, TDCA is 2.24 percentage points below ETRCA for Checkerboard, 3.77
points above for Binocular Vision, and 3.87 points above for Binocular-Swap.
The corresponding TDCA subject wins/ties/losses are 8/5/22, 26/4/5, and
28/2/5. Exploratory paired t-tests at this window give p=0.0179, p=0.000114,
and p=0.0000061 respectively; these are descriptive checks without a
multiple-comparison correction.

## ITR And Window Choice

All six paradigm-method curves reach their maximum mean ITR at 0.4 s:

| Paradigm | TDCA ITR | ETRCA ITR |
| --- | ---: | ---: |
| Checkerboard | 197.97 bits/min | 206.19 bits/min |
| Binocular Vision | 191.33 bits/min | 180.48 bits/min |
| Binocular-Swap | 158.00 bits/min | 145.01 bits/min |

Accuracy continues increasing after 0.4 s, but not enough to offset the longer
selection time in the standard `window + 0.5 s` ITR calculation.

## Paradigm And Subject Effects

The three-subject smoke correctly predicted that TDCA would help the two
binocular paradigms, but its BsV mean was optimistic: 85.83% at 0.6 s versus
69.76% in the full cohort.

At 0.6 s, TDCA subject accuracy correlations across paradigms are weak:

| Pair | Spearman rho | p |
| --- | ---: | ---: |
| CA vs BV | -0.095 | 0.587 |
| CA vs BsV | -0.223 | 0.198 |
| BV vs BsV | -0.076 | 0.664 |

![Subject-method delta](results/tdca_full_occipital9_20260720/figures/tdca_minus_etrca_delta_heatmap.png)

These results show a paradigm-by-subject interaction rather than a common
good-subject/bad-subject ordering. Cross-paradigm reports should retain
subject-level values and should not use one paradigm as a subject-quality proxy.

## Binocular-Swap References

BsV contains 40 ordered targets but only 20 unordered dual-frequency reference
subspaces. Arena TDCA still trains 40 separate EEG templates, so the duplicate
reference projection does not force paired targets into the same class.

At 0.6 s, TDCA made 2,117 errors and 219 were the corresponding swap partner
(10.3%). ETRCA made 2,388 errors and 220 were swap-partner errors (9.2%). Most
errors therefore occur between different frequency pairs rather than within a
swap pair.

## Interpretation Boundaries

1. This is not author TDCA; the public implementation contains ETRCA/FBDCCA.
2. The public stimulus codebook contains no phases, so references use zero phase.
3. The controlled comparison uses 9 channels for all paradigms. It must remain
   separate from the official BsV 64-channel baseline.
4. ETRCA here uses the same five subbands as TDCA, not the official seven-band
   configuration. This isolates method behavior but is not the official ETRCA
   result table.
