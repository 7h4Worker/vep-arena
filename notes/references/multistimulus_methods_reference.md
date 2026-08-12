# Multi-Stimulus CCA/TRCA Notes

Date: 2026-06-17

## Scope

TDCA should stay as the standard TDCA adapter unless a source-backed ensemble
variant is explicitly needed. Multi-stimulus CCA/TRCA methods are separate
algorithms and should be named separately, even when they reuse CCA, TRCA,
template, and correlation helpers.

## Source Trail

- Chi Man Wong et al., "Learning across multi-stimulus enhances target
  recognition methods in SSVEP-based BCIs", Journal of Neural Engineering,
  2020. DOI: `10.1088/1741-2552/ab2373`.
- PubMed entry: `https://pubmed.ncbi.nlm.nih.gov/31112937/`.
- Public demo repository: `https://github.com/edwin465/SSVEP-MSCCA-MSTRCA`.
- SSVEPAnalysisToolbox also lists multi-stimulus CCA and multi-stimulus TRCA
  as source-backed algorithms: `https://pypi.org/project/SSVEPAnalysisToolbox/`.

## Implementation Direction

- Treat `ms-eCCA` and `ms-eTRCA` as dedicated method adapters.
- Reuse Arena utilities for CCA references, subject-specific templates, TRCA
  spatial filters, and correlation scoring.
- Keep task manifests explicit about neighboring or auxiliary stimuli, because
  the learning unit is no longer one target in isolation.
- Add a smoke comparison against ordinary eCCA/eTRCA before putting
  multi-stimulus variants into shared figures.
