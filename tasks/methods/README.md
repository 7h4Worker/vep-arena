# Public method runners

This directory contains the runnable benchmark entry points for the public
method implementations under `vep_arena/methods/`.

| ID | Method | Entry point |
|---|---|---|
| M01 | RESS | `M01_ress/run.py` |
| M02 | PRCA | `M02_prca/run.py` |
| M03 | sTRCA | `M03_strca/run.py` |
| M04 | LA-TRCA | `M04_latrca/run.py` |
| M05 | MOHP filter | `M05_mohp/run.py` |
| M06 | Sinusoidal-Referenced TRCA | `M06_sinref_trca/run.py` |
| M07 | xTRCA | `M07_xtrca/run.py` |
| M08 | gTRCA | `M08_gtrca/run.py` |
| M09 | scTRCA | `M09_sctrca/run.py` |

Use `uv run python <entry-point> --help` to inspect a runner. Datasets remain
outside the repository, and generated caches and results are ignored.
