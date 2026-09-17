import numpy as np

from vep_arena.methods.la_trca import LATRCA, align_channels, estimate_wave_delays


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    sampling_rate = 250
    samples = 200
    time = np.arange(samples) / sampling_rate
    epochs = []
    labels = []
    channel_delays = np.asarray([0.008, 0.006, 0.004, 0.0, 0.003, 0.006, 0.005, 0.004, 0.005])
    for cls, frequency in enumerate((10.0, 12.0)):
        for _ in range(2):
            trial = np.asarray(
                [np.sin(2 * np.pi * frequency * (time - delay)) for delay in channel_delays]
            )
            epochs.append((trial + 0.08 * rng.standard_normal(trial.shape))[None])
            labels.append(cls)
    return np.asarray(epochs), np.asarray(labels)


def test_alignment_advances_channels() -> None:
    time = np.arange(250) / 250
    source = np.sin(2 * np.pi * 10 * time)
    delayed = np.sin(2 * np.pi * 10 * (time - 0.008))
    aligned = align_channels(np.vstack([source, delayed]), np.asarray([0.0, 0.008]), 250)
    assert np.corrcoef(aligned[0, :-4], aligned[1, :-4])[0, 1] > 0.99


def test_la_trca_fit_predict_contract() -> None:
    x, y = _epochs()
    delays = estimate_wave_delays(x[y == 0, 0], 10.0, 250)
    assert delays.shape == (9,)
    assert np.all(delays >= 0)
    assert np.max(np.abs(delays)) <= 0.05
    model = LATRCA(n_fbs=1, frequencies=(10.0, 12.0)).estimate_delays(x, y).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (4,)
    assert scores.shape == (4, 2)
    assert np.mean(predictions == y) >= 0.75
