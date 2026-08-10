# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-07-04
# Last updated: 2026-07-04
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, pinv, qr

from vep_arena.config import BenchmarkSpec
from vep_arena.data.benchmark import reference_signals


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _as_3d_epochs(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[:, 0]
    elif arr.ndim != 3:
        raise ValueError(f"{name} must be trials x subbands x channels x samples or trials x channels x samples.")
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 2:
        raise ValueError(f"{name} must contain at least one trial, one channel, and two samples.")
    return arr


def _as_4d_epochs(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim != 4:
        raise ValueError(f"{name} must be trials x subbands x channels x samples or trials x channels x samples.")
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1 or arr.shape[3] < 2:
        raise ValueError(f"{name} must contain trials, subbands, channels, and at least two samples.")
    return arr


def _filterbank_weights(n_fbs: int) -> np.ndarray:
    return np.asarray([(idx + 1) ** (-1.25) + 0.25 for idx in range(n_fbs)], dtype=np.float64)


def _orth_basis_rows(x: np.ndarray) -> np.ndarray:
    z = _center_rows(np.asarray(x, dtype=np.float64)).T
    if np.linalg.norm(z) <= 1e-12:
        return np.zeros((z.shape[0], 1), dtype=np.float64)
    q, r = np.linalg.qr(z, mode="reduced")
    if r.ndim != 2:
        return q[:, :1]
    diag = np.abs(np.diag(r))
    keep = diag > 1e-10
    if not np.any(keep):
        return q[:, :1] * 0.0
    return q[:, keep]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.ravel(a).astype(np.float64, copy=False)
    bv = np.ravel(b).astype(np.float64, copy=False)
    av = av - np.mean(av)
    bv = bv - np.mean(bv)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((av @ bv) / denom, -1.0, 1.0))


def _ged_filters(z: np.ndarray, basis: np.ndarray, n_components: int) -> np.ndarray:
    # For the orthogonal projector P = QQ.T, (Pz).T(Pz) is exactly
    # (Q.Tz).T(Q.Tz).  Keeping Q avoids an O(samples^2) projector.
    projected = basis.T @ z
    a = projected.T @ projected
    b = z.T @ z
    reg = 1e-8 * max(float(np.trace(b)) / max(b.shape[0], 1), 1.0)
    b = b + np.eye(b.shape[0], dtype=np.float64) * reg
    vals, vecs = eigh(a, b)
    order = np.argsort(vals)[::-1]
    return np.real(vecs[:, order[:n_components]])


def cca_filters(x: np.ndarray, y: np.ndarray, n_components: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return CCA spatial filters for row-wise signals.

    `x` and `y` are shaped variables x samples. The returned filters are
    variables x n_components and can be applied as `filters.T @ signal`.
    """

    if n_components < 1:
        raise ValueError("n_components must be positive.")
    x_c = _center_rows(np.asarray(x, dtype=np.float64))
    y_c = _center_rows(np.asarray(y, dtype=np.float64))
    if x_c.ndim != 2 or y_c.ndim != 2:
        raise ValueError("x and y must be two-dimensional.")
    if x_c.shape[1] != y_c.shape[1]:
        raise ValueError("x and y must have the same number of samples.")

    components = min(x_c.shape[0], y_c.shape[0])
    if n_components > components:
        raise ValueError("n_components exceeds the available CCA rank.")
    q, r = qr(y_c.T, mode="economic")
    wx = _ged_filters(x_c.T, q, components)
    wy = pinv(r) @ q.T @ x_c.T @ wx
    return wx[:, :n_components], wy[:, :n_components]


def _project(filters: np.ndarray, x: np.ndarray, n_components: int) -> np.ndarray:
    return filters[:, :n_components].T @ _center_rows(np.asarray(x, dtype=np.float64))


def ecca_scores(
    x: np.ndarray,
    templates: np.ndarray,
    refs: np.ndarray,
    template_ref_filters: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """Compute extended CCA scores.

    ECCA combines four correlations for each candidate class: standard CCA
    against sine-cosine references, individual-template CCA, and two template
    correlations using the standard/reference-derived spatial filters.
    """

    epochs = _as_3d_epochs(x, name="x")
    templates_arr = np.asarray(templates, dtype=np.float64)
    refs_arr = np.asarray(refs, dtype=np.float64)
    u3_arr = np.asarray(template_ref_filters, dtype=np.float64)
    if templates_arr.ndim != 3:
        raise ValueError("templates must be classes x channels x samples.")
    if refs_arr.ndim != 3:
        raise ValueError("refs must be classes x reference_rows x samples.")
    if u3_arr.ndim != 3:
        raise ValueError("template_ref_filters must be classes x channels x components.")
    if epochs.shape[1:] != templates_arr.shape[1:]:
        raise ValueError("x and templates have incompatible channel/sample dimensions.")
    if templates_arr.shape[0] != refs_arr.shape[0] or templates_arr.shape[0] != u3_arr.shape[0]:
        raise ValueError("templates, refs, and template_ref_filters must have the same class count.")

    scores = np.zeros((epochs.shape[0], templates_arr.shape[0]), dtype=np.float64)
    for trial_idx, trial in enumerate(epochs):
        for class_idx, (template, ref, u3) in enumerate(zip(templates_arr, refs_arr, u3_arr)):
            u1, v1 = cca_filters(trial, ref, n_components=n_components)
            rho1 = _safe_corr(_project(u1, trial, n_components), _project(v1, ref, n_components))
            rho2 = _safe_corr(_project(u1, trial, n_components), _project(u1, template, n_components))

            u2, _ = cca_filters(trial, template, n_components=n_components)
            rho3 = _safe_corr(_project(u2, trial, n_components), _project(u2, template, n_components))

            rho4 = _safe_corr(_project(u3, trial, n_components), _project(u3, template, n_components))
            rhos = np.asarray([rho1, rho2, rho3, rho4], dtype=np.float64)
            scores[trial_idx, class_idx] = float(np.sum(np.sign(rhos) * np.square(rhos)))
    return scores


class ECCA:
    """Filter-bank Extended CCA with subject-specific SSVEP templates.

    The Arena interface accepts either 3D trials x channels x samples input or
    4D trials x subbands x channels x samples input. For 4D input, ECCA follows
    the common SSVEP toolbox convention: compute eCCA features per subband and
    combine them with FBCCA-style subband weights.
    """

    name = "ECCA"

    def __init__(
        self,
        window: float,
        harmonics: int = 5,
        n_components: int = 1,
        n_fbs: int | None = None,
        spec: BenchmarkSpec | None = None,
        frequencies: tuple[float, ...] | list[float] | None = None,
        phases_pi: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        self.window = float(window)
        self.harmonics = int(harmonics)
        self.n_components = int(n_components)
        self.n_fbs = n_fbs
        self.spec = spec or BenchmarkSpec()
        self.refs_all = np.stack(
            reference_signals(
                self.window,
                self.harmonics,
                self.spec,
                frequencies=frequencies,
                phases_pi=phases_pi,
            )
        )
        self.classes_: np.ndarray | None = None
        self.templates: np.ndarray | None = None
        self.refs: np.ndarray | None = None
        self.template_ref_filters: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "ECCA":
        x = _as_4d_epochs(train_x, name="train_x")
        y = np.asarray(train_y)
        if y.ndim != 1:
            raise ValueError("train_y must be one-dimensional.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("train_x and train_y contain different numbers of trials.")
        if self.n_components > x.shape[2]:
            raise ValueError("n_components cannot exceed the number of channels.")

        classes = np.unique(y)
        templates = np.stack([np.mean(x[y == label], axis=0) for label in classes])
        refs = self.refs_all[classes.astype(np.int64)]
        filters = []
        for fb_idx in range(x.shape[1]):
            band_filters = []
            for template, ref in zip(templates[:, fb_idx], refs):
                u3, _ = cca_filters(template, ref, n_components=self.n_components)
                band_filters.append(u3)
            filters.append(np.stack(band_filters))

        self.classes_ = classes
        self.templates = templates
        self.refs = refs
        self.template_ref_filters = np.stack(filters)
        weight_count = x.shape[1] if self.n_fbs is None else max(int(self.n_fbs), x.shape[1])
        self.weights = _filterbank_weights(weight_count)[: x.shape[1]]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.classes_ is None
            or self.templates is None
            or self.refs is None
            or self.template_ref_filters is None
            or self.weights is None
        ):
            raise RuntimeError("ECCA model is not fitted.")
        epochs = _as_4d_epochs(x, name="x")
        if epochs.shape[1] != self.templates.shape[1]:
            raise ValueError("x has a different number of subbands from the fitted ECCA model.")
        scores = np.zeros((epochs.shape[0], self.templates.shape[0]), dtype=np.float64)
        for fb_idx in range(epochs.shape[1]):
            scores += self.weights[fb_idx] * ecca_scores(
                epochs[:, fb_idx],
                self.templates[:, fb_idx],
                self.refs,
                self.template_ref_filters[fb_idx],
                n_components=self.n_components,
            )
        return self.classes_[np.argmax(scores, axis=1)], scores
