# Environment Strategy

## Canonical local environment

VEP Arena uses the repository `.venv` as the canonical local execution
environment for agents and task scripts:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe
```

Keep it synchronized from `pyproject.toml` and `uv.lock`:

```powershell
cd D:\ProjData\proj_python\vep_arena
uv sync --extra all
.venv\Scripts\python.exe scripts\check_neuro_env.py
```

Do not use bare `python` for official runs. Commands should either call the
`.venv` interpreter explicitly or go through a documented wrapper.

## Dependency groups

The project uses optional dependency groups instead of per-model virtual
environments. All code lives in one repo; extras only control which dependency
families are installed.

### Layer 1 — core (default)

Traditional algorithms, data loading, plotting, artifact audit.

```
uv sync              # numpy/scipy/sklearn/pandas/matplotlib/h5py/…
```

Runs: CCA, FBCCA, TRCA, TDCA, MVMD, SA-MVMD, all data loaders, and plotting
scripts.

### Layer 2 — torch

All DNN training and evaluation.  One shared CUDA environment replaces the
five separate sibling-repo venvs (~29 GB saved).

```
uv sync --extra torch   # adds torch>=2.7 with CUDA 12.8 wheels
```

Runs: `evaluate_dnn_checkpoints.py`, `run_dnn_window_sweep.py`, and any
script that imports from `vep_arena.nn.*`.

### Layer 2 — neuro (optional)

MNE/MOABB for neuroscience QA, topomaps, and dataset downloads.

```
uv sync --extra neuro   # adds mne>=1.8, moabb>=1.1
```

Runs: `make_benchmark_mne_qa.py`, `check_neuro_env.py`, and scripts under
`vep_arena.neuroviz`.

### Layer 2 — reports (optional)

PowerPoint/report helper dependencies.

```
uv sync --extra reports   # adds python-pptx and its dependencies
```

Runs: `generate_hd200_ppt.py` and other local report-generation helpers.

### Full install

```
uv sync --extra all     # core + torch + neuro + reports
```

## Migration from per-repo venvs

The sibling repos (`dnn_ssvep_pytorch`, `ssvepformer_benchmark_pytorch`,
`trcanet_benchmark_pytorch`, `eegnet_minimal`, `ssvepformer_minimal`) each
had their own `.venv` with `torch==2.7.1+cu128`.  Their model code has been
copied into `vep_arena/vep_arena/nn/`:

| Sibling repo                    | Arena module                      |
|---------------------------------|-----------------------------------|
| `dnn_ssvep_pytorch`             | `vep_arena.nn.dnn_ssvep.DNNSsvep` |
| `ssvepformer_benchmark_pytorch` | `vep_arena.nn.ssvepformer.*`      |
| `trcanet_benchmark_pytorch`     | `vep_arena.nn.trcanet.TRCANet`    |
| `eegnet_minimal`                | `vep_arena.nn.eegnet.EEGNetMini`  |

The sibling repos are **frozen** — they stay on disk as historical baselines
and provenance for imported results, but no new development happens there.
Scripts like `import_dnn_results.py` and `build_report.py` still reference
sibling result CSVs for historical comparison; this is intentional.

## Legacy conda environment

`D:\ProjData\envs\erp_ssvep_lab` (Python 3.10, conda) has MNE 1.12.1 and
PsychoPy. MNE is now available in the Arena `.venv`, so this conda environment
is no longer the default for Arena task execution. Keep it as a legacy
PsychoPy/online-experiment fallback unless a task explicitly requires it.

## Arena-native training scripts

All DNN models can now be trained and evaluated entirely within Arena:

| Script                           | Model          | Preprocessing                       |
|----------------------------------|----------------|--------------------------------------|
| `scripts/evaluate_dnn_checkpoints.py` | DNNSsvep       | Chebyshev filterbank (3 subbands)    |
| `scripts/evaluate_ssvepformer.py`     | SSVEPFormerTH  | Butterworth bandpass 8–64 Hz + FFT   |
| `scripts/evaluate_trcanet.py`         | TRCANet        | TRCA spatial filter projection       |

All scripts output Arena-format artifacts: `trials.csv`, `summary.csv`,
`block.csv`, `predictions.csv`, `runtime.csv`, `epoch_history.csv`,
`manifest.json`, and `confusion_*.npy`.

Smoke tasks under `tasks/benchmark_*_smoke/` validate each pipeline with
2 subjects, 1 block, 10 epochs.

## Cleaning up sibling venvs

Once Arena is confirmed working (`uv sync --extra all` + smoke tests pass),
the sibling repo `.venv` directories can be deleted to reclaim ~29 GB:

```bash
# From D:/ProjData/proj_python/
rm -rf dnn_ssvep_pytorch/.venv
rm -rf ssvepformer_benchmark_pytorch/.venv
rm -rf trcanet_benchmark_pytorch/.venv
rm -rf eegnet_minimal/.venv
rm -rf ssvepformer_minimal/.venv
```

The sibling repos themselves should be kept — they contain historical results
under `outputs/` (~1.6 GB of .pt checkpoints) and serve as provenance for
imported Arena baselines.

## CUDA index

The PyTorch CUDA 12.8 wheel index is configured in `pyproject.toml` under
`[tool.uv.sources]` and `[[tool.uv.index]]`.  This avoids the need for
`--index-url` flags on the command line.
