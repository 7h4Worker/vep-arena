import numpy as np

from vep_arena.methods.gtrca import gTRCA, group_trca_filters


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    samples = 100
    time = np.arange(samples) / 250
    subjects = []
    labels = np.tile(np.repeat(np.arange(2), 2), (3, 1))
    for subject in range(3):
        trials = []
        mixing = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
        for cls, frequency in enumerate((10.0, 12.0)):
            for _ in range(2):
                source = np.vstack(
                    [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time), np.zeros(samples)]
                )
                trials.append((mixing @ source + 0.08 * rng.standard_normal(source.shape))[None])
        subjects.append(trials)
    return np.asarray(subjects), labels


def test_group_filter_shapes() -> None:
    x, labels = _epochs()
    filters, templates = group_trca_filters(x[:, labels[0] == 0, 0])
    assert filters.shape == (3, 3)
    assert templates.shape == (3, 3, 100)
    assert np.allclose(np.linalg.norm(filters, axis=1), 1.0)


def test_gtrca_multisubject_fit_predict_contract() -> None:
    x, labels = _epochs()
    model = gTRCA(n_fbs=1).fit(x, labels)
    predictions, scores = model.predict(x[:, :, 0:1])
    assert predictions.shape == (3, 4)
    assert scores.shape == (3, 4, 2)
    assert np.mean(predictions == labels) >= 2 / 3
