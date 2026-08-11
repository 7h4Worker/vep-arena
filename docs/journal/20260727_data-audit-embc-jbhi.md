# EMBC/JBHI 9/16/35-target data audit - 2026-07-27

## Scope and privacy

The external transfer package was audited in place and was not modified. Repository code and generated outputs use only stable anonymous IDs (`S01`, `S02`, ...). Historical participant codes, source filenames, and identity mappings embedded in source MAT files are not propagated.

## Integrity and MAT structure

- Updated `SHA256SUMS.csv`: all 46 listed files matched; no listed file was missing and no extra file existed outside the checksum list.
- EMBC 9-target: 8 files, each `data [13, 4000, 9, 20]`, numeric finite values, `Fs=1000`. The paper channels are the final 9 stored channels.
- JBHI 16-target: 13 files, each `eegdata [9, 4001, 96]`, numeric finite values, `sample_rate=1000`. Every label vector is exactly six repetitions of `1..16`; stored channel order is `Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2`.
- JBHI 35-target labeled source: 6 files, each `eegdata [9, 2000/2001, 210]`, numeric finite values, `sample_rate=1000`, explicit `label_list`, and file-verified channel order. Each consecutive 35-trial block contains labels `1..35` exactly once. Grouping the first 2000 samples by label reproduces the six older canonical 4D arrays element-for-element (`max_abs_diff=0`).
- JBHI 35-target mapping: `target_mapping_35.csv` contains 35 ordered target labels and the authoritative left/right frequency pair for each target. It is read only from the configured private-data directory and is not copied into repository outputs.

## Protocols

### EMBC 9-target

- Published VR binocular encoding with 8.5 and 9.5 Hz plus a no-stimulation unit, producing 9 targets.
- Eight participants, 20 blocks, 9 trials per block; target presentation order was randomized within block.
- Each trial used 2 s cue plus 4 s stimulation.
- Epoch `[0.13, 4.13]` s; Cz reference, FPz ground, 48-52 Hz notch, EEGLAB basic FIR 5-100 Hz. The recovered paper-result code verifies direct 4x decimation (`1:4:end`) to 250 Hz.
- Paper baseline: subject-specific TRCA. The paper says leave-one-out; the recovered MATLAB code resolves the held-out unit as one complete 9-trial block in each of 20 folds.
- The recovered receiver uses three dynamic Chebyshev subbands with lower passbands 6/14/22 Hz. Its training function accidentally cascades the bands while testing applies them independently. Arena retains the standard independent-subband definition and keeps the asymmetry only as an audit comparator.
- The final plotting script excludes anonymous `S02` from the reported seven-participant mean. At 2 s, the recovered MAT result is 79.0476%; corrected Arena three-band TRCA is 79.8413%, versus 71.5079% under the previous single-band setup.

### JBHI 16-target

- Final manuscript VR binocular encoding uses independent left/right units from `{void, 11, 12, 13 Hz}` to form 16 targets.
- Thirteen participants, 6 blocks, 16 trials per block; fixed column-wise target order from top-to-bottom then left-to-right.
- Each trial used 2 s cue plus 4 s stimulation, with 20 s between blocks.
- Epoch `[0.13, 4.13]` s; Cz reference, FPz ground, 48-52 and 98-102 Hz notch, EEGLAB basic FIR 4-100 Hz, 1000 Hz analysis.
- Paper CV is explicitly six-fold leave-one-block-out. ITR uses `window + 0.5 s`.
- Receiver filter bank is separate from EEGLAB preprocessing. The manuscript specifies sixth-order Chebyshev Type I passbands `[8b-2, 90]` Hz and weights `b^-1.25+0.25`, but omits `B` and ripple; the smoke records `B=5`, `0.5 dB` as assumptions.
- A single-window diagnostic selected anonymous `S04,S06` as representative smoke examples; their mean 0.4 s five-band ETRCA accuracy is 72.9% (about 142 bits/min).

### JBHI 35-target

- Unpublished JBHI extension, 6 participants, 35 targets, 6 blocks, 2 s stored analysis window.
- Uses the operator-confirmed JBHI manual EEGLAB profile. No per-file EEGLAB histories are supplied, but the updated transfer now includes explicit event labels, channel names, and the authoritative target codebook.
- Arena treats this as a diagnostic six-fold leave-one-block-out extension, not a paper reproduction.
- The initial smoke was near chance. Its follow-up Hungarian target-matching heuristic is now retracted: maximizing assignment over noisy single trials can overfit chance correlations and cannot diagnose label permutation.
- The authoritative `label_list` fixes the supplied block labels, and reordering by it reproduces the old 4D values exactly. The labeled source is now the preferred loader input. This excludes an Arena reshape/order bug, but it does not independently prove every historical ordered left/right stimulus annotation.
- A follow-up 2 s audit over all six anonymous subjects compared TRCA/eTRCA with one versus five receiver bands. Five bands improved mean TRCA from 21.4% to 27.2% and mean ETRCA from 28.2% to 32.6%; it did not cause the low initial result.
- A subject-level signal audit separates the low cases. `S02` has near-zero expected-frequency effect and chance-level unordered frequency-set agreement, consistent with low signal quality or a source annotation problem. `S01,S03` have clear expected-frequency effects and above-chance frequency-set agreement, but expected-frequency Oz ITPC near the six-block random-phase level; their spectra broadly match the codebook while their phase-locked temporal templates do not.
- The historical MATLAB `train_trca.m` cascades filter-bank outputs during training while `test_trca.m` filters each test band independently. Reproducing that asymmetry changes the six-subject 2 s mean from 32.62% to 33.65% and does not rescue `S01-S03`. Arena keeps the standard independent-subband definition and treats the historical MAT values only as implementation-provenance anchors.
- The audit identified `S04,S06` as quality-normal smoke examples: five-band 2 s ETRCA reached 87.1% and 63.8%. `S01-S03` were near chance and `S05` was weaker, so the initial `S01,S03` smoke was not representative.
- The corrected `S04,S06` smoke completed 8/8 units and 1680/1680 predictions with no errors. Mean ETRCA accuracy increased from 39.0% at 0.2 s to 75.5% at 2.0 s; mean ITR peaked at 115.7 bits/min at 0.4 s.
- A later read-only audit of the legacy paper workspace found that its 35-target plotting script expected five sessions, not the current canonical six. Only three sessions overlap (`S04-S06`); the other two are alternate sessions for `S01` and `S03`, while current `S02` is absent. Under the same standard Arena ETRCA, the legacy five-session mean is 64.86% at 2 s versus 32.62% for the current six-session set.
- The alternate `S01` and `S03` sessions reach 78.10% and 67.14% at 2 s, compared with 6.67% and 2.86% for their current sessions. Their expected-frequency Oz ITPC is 0.773 and 0.876 versus 0.339 and 0.334 currently. This identifies a session-level phase/timing-stability difference; it does not support treating the low result as a stable participant trait or a simple global label permutation.

## Preprocessing boundary

The supplied MAT arrays are already manually preprocessed. Arena does not repeat the upstream EEGLAB notch or bandpass filters. EMBC performs the legacy-code-verified direct decimation and algorithm-level three-band receiver filtering; JBHI classification methods likewise apply only their algorithm-level receiver filter bank. Runtime windows start at sample zero because the stored epochs already include the 130 ms latency correction.

## bPRCA and FusionCA verification

- The transfer package itself does not contain the original bPRCA/FusionCA MATLAB implementation or result tables/figure assets. A separate read-only legacy workspace copy was subsequently located on the external backup and contains the implementation plus final JBHI16 result MAT files. The transfer provenance and the legacy-code provenance remain separate.
- The more complete published snapshot contains 236 JBHI result MAT files across final Bfusion, TDCA, gamma sweeps, and intermediate correction folders. Only the exact-source, protocol-resolved final anchors are used for reproduction claims; inventory counts do not turn parameter sweeps or intermediate files into paper evidence.
- The Arena implementation consumes the existing receiver filter-bank epochs, performs period segmentation, reproduces the MATLAB cross-period covariance and full-trial normalization, builds repeated period templates, and combines sub-band scores with `b^-1.25+0.25`.
- A MATLAB-definition unit test compares the vectorized covariance solution with an independent explicit period-pair loop. The complete repository test suite passes (`22 passed`).
- EMBC 9-target full run covered all 8 anonymous subjects, 5 windows, and 6 methods. It completed 240/240 units and 43200/43200 predictions with an empty error log and BLAS threads fixed to one. At 4 s, TRCA reached 78.89%, ETRCA 74.58%, and eFusionCA 71.11%. On the paper-included seven subjects, TRCA reached 79.84% at 2 s and 81.98% at 4 s, versus recovered MATLAB values of 79.05% and 81.35%. The remaining difference is implementation-level, not a source-array or CV-unit mismatch.
- JBHI 16-target full run covered all 13 anonymous subjects, ten 0.2-2.0 s windows, and 6 methods. It completed 780/780 units and 74880/74880 predictions with no errors. After correcting the PRCA denominator to the period-segment autocovariance sum in the manuscript, eFusionCA reached 71.394% and 138.70 bits/min at 0.4 s versus the paper's 71.39% and 138.50 bits/min. At 2.0 s it reached 82.21% versus the active manuscript value of 82.69%. The nearby 82.29% sentence is commented out in the supplied TeX and is not treated as a paper anchor. The paper comparison is recorded in the task-owned `paper_comparison.csv` and figure.
- All 13 JBHI16 transfer arrays match one legacy source array exactly. The legacy workspace supplies 390 anonymous subject-method-window anchors for eTRCA, ebPRCA, and eFusionCA; Arena hard decisions agree on average in 93.66% of trials. Group eFusionCA accuracy matches exactly at 0.4 s (71.3942%) and differs by -0.4808 percentage points at 2.0 s (82.2115% versus 82.6923%), showing close but not bit-identical implementation behavior.
- JBHI 35-target full diagnostic run covered all 6 anonymous subjects, ten 0.2-2.0 s windows, and 6 methods. It completed 360/360 units and 75600/75600 predictions with no errors. At 2.0 s, ETRCA reached 32.62%, ebPRCA reached 32.70%, and eFusionCA reached 35.95%; the group mean remains limited by the three near-chance subjects identified during the receiver audit. This dataset has no paper result anchor and is therefore a diagnostic extension, not a paper reproduction.
- Across the three full runs, all 1380 expected units and all 193680 expected predictions completed with zero logged errors.

## Remaining evidence gaps

- EMBC's paper does not name the leave-one-out unit in prose, but the recovered result code verifies block-level folds; the paper text and code provenance should remain separately labeled.
- EMBC's stored `result_label_best` vectors correspond to different or ambiguous curve windows across subjects and are not a uniform-window hard-decision anchor.
- EMBC MAT files do not embed their complete channel/codebook metadata.
- JBHI 35-target preprocessing is operator-confirmed rather than per-file history-verified.
- The older JBHI 35-target canonical 4D MAT files omit labels and the target codebook; retain them only as numeric references and use the labeled source for Arena loading.
- The legacy 35-target plotting script references five result MAT files, but only two of those exact five files remain available; a third MAT is an alternate scoring variant for one overlapping session. The anonymous legacy-cohort receiver comparison is therefore recomputed with the declared Arena implementation rather than presented as a recovered five-subject MATLAB result.
- The JBHI manuscript does not specify receiver filter-bank count or Chebyshev ripple.
