import numpy as np

from vep_arena.methods.sscor import SSCOR, sscor_filter, sscor_scores


def _toy_epochs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = 3
    trials_per_class = 8
    n_fbs = 2
    channels = 4
    samples = 160
    time = np.linspace(0.0, 1.0, samples, endpoint=False)
    spatial_patterns = rng.normal(size=(classes, channels, 2))

    x = []
    y = []
    for cls in range(classes):
        freq = 8.0 + 3.0 * cls
        source = np.stack(
            [
                np.sin(2.0 * np.pi * freq * time),
                np.cos(2.0 * np.pi * freq * time),
            ]
        )
        template = spatial_patterns[cls] @ source
        for _ in range(trials_per_class):
            bands = []
            for fb_idx in range(n_fbs):
                noise = 0.08 * rng.normal(size=(channels, samples))
                bands.append(template / float(fb_idx + 1) + noise)
            x.append(bands)
            y.append(cls)

    order = rng.permutation(len(y))
    return np.asarray(x, dtype=np.float64)[order], np.asarray(y)[order]


def test_sscor_fit_predict_shape_and_3d_single_subband() -> None:
    x, y = _toy_epochs()
    model = SSCOR(n_components=1).fit(x, y)

    pred, scores = model.predict(x[:5])
    assert pred.shape == (5,)
    assert scores.shape == (5, 3)
    assert model.templates is not None
    assert model.filters is not None
    assert model.templates.shape == (3, 2, 4, 160)
    assert model.filters.shape == (2, 3, 4, 1)

    single_band = SSCOR(n_components=1).fit(x[:, 0], y)
    pred_3d, scores_3d = single_band.predict(x[:5, 0])
    assert pred_3d.shape == (5,)
    assert scores_3d.shape == (5, 3)


def test_sscor_separates_toy_data() -> None:
    x, y = _toy_epochs()
    model = SSCOR(n_components=1, ensemble=False).fit(x, y)

    pred, scores = model.predict(x)
    assert scores.shape == (x.shape[0], 3)
    assert np.mean(pred == y) >= 0.95


def test_sscor_ensemble_runs_and_scores() -> None:
    x, y = _toy_epochs()
    model = SSCOR(n_components=1, ensemble=True).fit(x, y)

    pred, scores = model.predict(x[:6])
    assert pred.shape == (6,)
    assert scores.shape == (6, 3)
    assert np.all(np.isfinite(scores))


def test_sscor_kernel_and_score_helpers() -> None:
    x, y = _toy_epochs()
    filters, eigenvalues = sscor_filter(x[y == 0, 0])

    assert filters.shape == (4, 4)
    assert eigenvalues.shape == (4,)
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    model = SSCOR(n_components=1).fit(x, y)
    scores = sscor_scores(
        x[:4],
        model.templates,
        model.filters,
        model.weights,
        n_components=model.n_components,
        ensemble=False,
    )
    assert scores.shape == (4, 3)
