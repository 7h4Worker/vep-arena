"""Sum-of-squared-correlations spatial filtering for SSVEP.

Reference
---------
Kumar GRK, Reddy MR. Designing a Sum of Squared Correlations Framework for
Enhancing SSVEP-Based BCIs. IEEE TNSRE. 2019.
doi:10.1109/TNSRE.2019.2941349.

Implementation identity
-----------------------
This is an Arena numerical adaptation of the public SSCOR training/testing
code, not a bitwise MATLAB port. It uses symmetric eigendecomposition,
scale-relative diagonal regularization, and an Arena-style filter-bank
classifier interface. Those choices make rank-deficient short-window data
well-defined and must be reported when comparing against paper tables.
"""
from __future__ import annotations

import numpy as np


def filterbank_weights(n_fbs: int) -> np.ndarray:
    """Default SSCOR/FBSSCOR subband weights."""

    return np.asarray([(idx + 1) ** (-1.25) + 0.25 for idx in range(n_fbs)], dtype=np.float64)


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _as_3d_trials(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be shaped trials x channels x samples.")
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 2:
        raise ValueError(f"{name} must contain at least one trial, one channel, and two samples.")
    return arr


def _as_4d_epochs(x: np.ndarray, *, name: str) -> np.ndarray:
    """Return epochs as trials x subbands x channels x samples.

    Native Arena callers should pass 4D filter-bank epochs. A 3D
    trials x channels x samples array is accepted and treated as one subband.
    """

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]
    elif arr.ndim != 4:
        raise ValueError(
            f"{name} must be shaped trials x subbands x channels x samples "
            "or trials x channels x samples."
        )
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1 or arr.shape[3] < 2:
        raise ValueError(
            f"{name} must contain at least one trial, one subband, one channel, and two samples."
        )
    return arr


def _symmetrize(x: np.ndarray) -> np.ndarray:
    return (x + x.T) * 0.5


def _regularized_cov(x: np.ndarray, reg: float) -> np.ndarray:
    cov = _symmetrize(x @ x.T)
    scale = float(np.trace(cov) / cov.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return cov + np.eye(cov.shape[0], dtype=np.float64) * (reg * scale)


def _inverse_sqrt_psd(cov: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_symmetrize(cov))
    vals = np.maximum(vals, np.finfo(np.float64).eps)
    return (vecs / np.sqrt(vals)) @ vecs.T


def _normalize_columns(w: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(w, axis=0, keepdims=True)
    out = np.array(w, dtype=np.float64, copy=True)
    keep = norms[0] > 1e-12
    out[:, keep] /= norms[:, keep]
    return out


def sscor_filter(trials: np.ndarray, *, reg: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Fit regularized Arena SSCOR filters for one class and one subband.

    Parameters
    ----------
    trials:
        Training epochs shaped trials x channels x samples.
    reg:
        Relative diagonal regularization used when whitening class templates
        and individual trials.

    Returns
    -------
    filters, eigenvalues:
        Spatial filters shaped channels x filters and their descending
        eigenvalues. Callers usually keep the first ``n_components`` columns.
    """

    x = _center_rows(_as_3d_trials(trials, name="trials"))
    if reg < 0.0:
        raise ValueError("reg must be non-negative.")

    template = np.mean(x, axis=0)
    template_cov = _regularized_cov(template, reg)
    template_white = _inverse_sqrt_psd(template_cov)

    channels = x.shape[1]
    objective = np.zeros((channels, channels), dtype=np.float64)
    for trial in x:
        trial_cov = _regularized_cov(trial, reg)
        trial_white = _inverse_sqrt_psd(trial_cov)
        cross_cov = template @ trial.T
        g = template_white @ cross_cov @ trial_white
        objective += g.T @ g

    vals, vecs = np.linalg.eigh(_symmetrize(objective))
    order = np.argsort(vals)[::-1]
    vals = np.real(vals[order])
    filters = _normalize_columns(template_white @ np.real(vecs[:, order]))
    return filters, vals


def sscor_feature(filters: np.ndarray, x: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Project epochs through SSCOR filters.

    Parameters
    ----------
    filters:
        Spatial filters shaped channels x filters.
    x:
        Epochs shaped trials x channels x samples.
    n_components:
        Number of leading filters to apply.

    Returns
    -------
    np.ndarray
        Projected features shaped trials x n_components x samples.
    """

    w = np.asarray(filters, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("filters must be shaped channels x filters.")
    if n_components < 1 or n_components > w.shape[1]:
        raise ValueError("n_components must be between 1 and the number of filters.")

    epochs = _center_rows(_as_3d_trials(x, name="x"))
    if epochs.shape[1] != w.shape[0]:
        raise ValueError("x and filters have incompatible channel counts.")
    return np.matmul(w[:, :n_components].T, epochs)


def _corr_to_template(features: np.ndarray, template_feature: np.ndarray) -> np.ndarray:
    a = np.reshape(features, (features.shape[0], -1))
    b = np.ravel(template_feature)
    a = a - np.mean(a, axis=1, keepdims=True)
    b = b - np.mean(b)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
    out = np.zeros(a.shape[0], dtype=np.float64)
    keep = denom > 1e-12
    out[keep] = a[keep] @ b / denom[keep]
    return out


def sscor_scores(
    x: np.ndarray,
    templates: np.ndarray,
    filters: np.ndarray,
    weights: np.ndarray,
    *,
    n_components: int = 1,
    ensemble: bool = False,
) -> np.ndarray:
    """Score SSCOR or ensemble SSCOR trials.

    Shapes
    ------
    x:
        trials x subbands x channels x samples.
    templates:
        classes x subbands x channels x samples.
    filters:
        subbands x classes x channels x fitted_filters.
    weights:
        One scalar per subband.

    With ``ensemble=False``, each class is scored through its own filters.
    With ``ensemble=True``, filters from all classes are concatenated per
    subband and reused for every class template, matching the ETRCA-style
    ensemble scoring semantics.
    """

    epochs = _as_4d_epochs(x, name="x")
    templates_arr = np.asarray(templates, dtype=np.float64)
    filters_arr = np.asarray(filters, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)

    if templates_arr.ndim != 4:
        raise ValueError("templates must be shaped classes x subbands x channels x samples.")
    if filters_arr.ndim != 4:
        raise ValueError("filters must be shaped subbands x classes x channels x fitted_filters.")
    if epochs.shape[1:] != templates_arr.shape[1:]:
        raise ValueError("x and templates have incompatible subband/channel/sample dimensions.")
    if filters_arr.shape[:3] != (epochs.shape[1], templates_arr.shape[0], epochs.shape[2]):
        raise ValueError("filters have incompatible subband/class/channel dimensions.")
    if n_components < 1 or n_components > filters_arr.shape[-1]:
        raise ValueError("n_components must be between 1 and the fitted filter count.")
    if weights_arr.shape[0] < epochs.shape[1]:
        raise ValueError("weights must contain at least one value per subband.")

    trials, n_fbs, _, _ = epochs.shape
    classes = templates_arr.shape[0]
    band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)

    for fb_idx in range(n_fbs):
        if ensemble:
            w = np.concatenate(
                [filters_arr[fb_idx, cls, :, :n_components] for cls in range(classes)],
                axis=1,
            )
            projected_trials = sscor_feature(w, epochs[:, fb_idx], n_components=w.shape[1])
            for cls in range(classes):
                projected_template = sscor_feature(
                    w,
                    templates_arr[cls : cls + 1, fb_idx],
                    n_components=w.shape[1],
                )[0]
                band_scores[:, fb_idx, cls] = _corr_to_template(projected_trials, projected_template)
        else:
            for cls in range(classes):
                w = filters_arr[fb_idx, cls, :, :n_components]
                projected_trials = sscor_feature(w, epochs[:, fb_idx], n_components=n_components)
                projected_template = sscor_feature(
                    w,
                    templates_arr[cls : cls + 1, fb_idx],
                    n_components=n_components,
                )[0]
                band_scores[:, fb_idx, cls] = _corr_to_template(projected_trials, projected_template)

    return np.einsum("f,tfc->tc", weights_arr[:n_fbs], band_scores)


class SSCOR:
    """Arena-native Sum of Squared Correlations classifier.

    ``fit`` and ``predict`` use Arena's traditional-method convention:
    epochs are shaped trials x subbands x channels x samples. For convenience,
    3D trials x channels x samples input is also accepted and treated as a
    single-subband filter-bank input.
    """

    name = "SSCOR"

    def __init__(
        self,
        n_fbs: int | None = 5,
        n_components: int = 1,
        ensemble: bool = False,
        filterweights: np.ndarray | None = None,
        reg: float = 1e-8,
    ) -> None:
        if n_fbs is not None and n_fbs < 1:
            raise ValueError("n_fbs must be positive when provided.")
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        if reg < 0.0:
            raise ValueError("reg must be non-negative.")

        self.n_fbs = n_fbs
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = None if filterweights is None else np.asarray(filterweights, dtype=np.float64)
        self.reg = reg

        self.weights: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "SSCOR":
        x = _as_4d_epochs(train_x, name="train_x")
        y = np.asarray(train_y)
        if y.ndim != 1:
            raise ValueError("train_y must be one-dimensional.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("train_x and train_y contain different numbers of trials.")
        if self.n_components > x.shape[2]:
            raise ValueError("n_components cannot exceed the number of channels.")

        classes = np.unique(y)
        if classes.size < 1:
            raise ValueError("train_y must contain at least one class.")

        _, n_fbs, channels, samples = x.shape
        templates = np.zeros((classes.size, n_fbs, channels, samples), dtype=np.float64)
        filters = np.zeros((n_fbs, classes.size, channels, self.n_components), dtype=np.float64)
        eigenvalues = np.zeros((n_fbs, classes.size, self.n_components), dtype=np.float64)

        for class_idx, label in enumerate(classes):
            class_trials = x[y == label]
            templates[class_idx] = np.mean(class_trials, axis=0)
            for fb_idx in range(n_fbs):
                class_filters, class_eigenvalues = sscor_filter(class_trials[:, fb_idx], reg=self.reg)
                filters[fb_idx, class_idx] = class_filters[:, : self.n_components]
                eigenvalues[fb_idx, class_idx] = class_eigenvalues[: self.n_components]

        if self.filterweights is None:
            weights = filterbank_weights(n_fbs if self.n_fbs is None else max(self.n_fbs, n_fbs))[:n_fbs]
        else:
            if self.filterweights.shape[0] < n_fbs:
                raise ValueError("filterweights must contain at least one value per fitted subband.")
            weights = self.filterweights[:n_fbs].astype(np.float64, copy=True)

        self.weights = weights
        self.classes_ = classes
        self.templates = templates
        self.filters = filters
        self.eigenvalues = eigenvalues
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.classes_ is None or self.templates is None or self.filters is None or self.weights is None:
            raise RuntimeError("SSCOR model is not fitted.")

        scores = sscor_scores(
            x,
            self.templates,
            self.filters,
            self.weights,
            n_components=self.n_components,
            ensemble=self.ensemble,
        )
        return self.classes_[np.argmax(scores, axis=1)], scores
