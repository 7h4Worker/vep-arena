# EMBC/JBHI Oz multi-trial FFT label checks

This task creates one paper-interface FFT figure for each private EMBC/JBHI dataset using the Oz channel and all available samples in every trial. It computes a Hann-tapered spectrum for each trial separately and then averages power across trials, avoiding phase cancellation from a time-domain trial average. Every interface cell contains the spectrum of its corresponding target.

- EMBC 9-target uses the paper's `3 x 3` interface and the full 4-second, 1000 Hz stored epoch. Its complete codebook is cross-checked against the paper interface and original experiment code.
- JBHI 16-target uses Figure 1's `4 x 4` interface and column-major trial order with the full 4-second epoch. Its complete codebook is cross-checked against the paper and original experiment code.
- JBHI 35-target uses the file-verified `5 x 7` target grid, full 2-second epoch, and all transferred label-to-left/right frequency mappings.
- No extra algorithm filter bank is applied to these descriptive spectra because the stored epochs already contain the paper preprocessing.
- `--subject auto` ranks anonymous subjects independently for each dataset by median target-frequency Oz SNR and selects the clearest example. The default 16384-point FFT zero-pads without claiming resolution beyond the original 4-second or 2-second epoch.
- The plots and manifest contain anonymous IDs only. External MAT and CSV files are read in place and are never copied.

Run:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\plot_embc_jbhi_fft_features.py --subject auto --n-fft 16384
```
