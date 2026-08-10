import numpy as np
from scipy.linalg import qr

from vep_arena.methods.ecca import ECCA, _center_rows, _ged_filters, cca_filters, ecca_scores


def _toy_epochs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = 3
    trials_per_class = 7
    channels = 4
    samples = 250
    time = np.arange(samples) / 250.0
    patterns = rng.normal(size=(classes, channels, 2))
    x = []
    y = []
    for cls in range(classes):
        freq = 8.0 + cls
        source = np.stack(
            [
                np.sin(2.0 * np.pi * freq * time),
                np.cos(2.0 * np.pi * freq * time),
            ]
        )
        template = patterns[cls] @ source
        for _ in range(trials_per_class):
            x.append(template + 0.03 * rng.normal(size=template.shape))
            y.append(cls)
    order = rng.permutation(len(y))
    return np.asarray(x, dtype=np.float64)[order], np.asarray(y)[order]


def test_cca_filters_project_correlated_signals() -> None:
    x, _ = _toy_epochs()
    wx, wy = cca_filters(x[0], x[1], n_components=1)
    assert wx.shape == (4, 1)
    assert wy.shape == (4, 1)
    assert np.all(np.isfinite(wx))
    assert np.all(np.isfinite(wy))


def test_ged_basis_matches_explicit_projector() -> None:
    rng = np.random.default_rng(17)
    x = _center_rows(rng.normal(size=(9, 125)))
    y = _center_rows(rng.normal(size=(6, 125)))
    q, _ = qr(y.T, mode="economic")
    z = x.T
    projector = q @ q.T
    projected = projector.T @ z
    legacy_a = projected.T @ projected
    fast_projected = q.T @ z
    fast_a = fast_projected.T @ fast_projected

    np.testing.assert_allclose(fast_a, legacy_a, rtol=1e-11, atol=1e-11)
    filters = _ged_filters(z, q, n_components=2)
    assert filters.shape == (x.shape[0], 2)


def test_ecca_fit_predict_shape_and_3d_input() -> None:
    x, y = _toy_epochs()
    model = ECCA(window=1.0, harmonics=2, n_components=1).fit(x[:, None], y)
    pred, scores = model.predict(x[:5, None])
    assert pred.shape == (5,)
    assert scores.shape == (5, 3)

    model_3d = ECCA(window=1.0, harmonics=2, n_components=1).fit(x, y)
    pred_3d, scores_3d = model_3d.predict(x[:4])
    assert pred_3d.shape == (4,)
    assert scores_3d.shape == (4, 3)


def test_ecca_separates_toy_data() -> None:
    x, y = _toy_epochs()
    model = ECCA(window=1.0, harmonics=2, n_components=1).fit(x, y)
    pred, scores = model.predict(x)
    assert np.mean(pred == y) >= 0.95
    assert scores.shape == (x.shape[0], 3)


def test_ecca_score_helper_matches_model_shape() -> None:
    x, y = _toy_epochs()
    model = ECCA(window=1.0, harmonics=2, n_components=1).fit(x, y)
    scores = ecca_scores(
        x[:3],
        model.templates[:, 0],
        model.refs,
        model.template_ref_filters[0],
        n_components=model.n_components,
    )
    assert scores.shape == (3, 3)


def test_ecca_uses_filterbank_input() -> None:
    x, y = _toy_epochs()
    fb_x = np.stack([x, 0.5 * x], axis=1)
    model = ECCA(window=1.0, harmonics=2, n_components=1, n_fbs=2).fit(fb_x, y)
    pred, scores = model.predict(fb_x[:5])
    assert pred.shape == (5,)
    assert scores.shape == (5, 3)
    assert model.templates.shape[:2] == (3, 2)
    assert model.template_ref_filters.shape[:2] == (2, 3)
