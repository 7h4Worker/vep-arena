import numpy as np
import pytest

from vep_arena.methods.fbdcca import FBDCCA


def _references(samples: int = 250) -> list[np.ndarray]:
    time = np.arange(samples) / 250.0
    return [
        np.stack(
            [
                np.sin(2.0 * np.pi * frequency * time),
                np.cos(2.0 * np.pi * frequency * time),
            ]
        )
        for frequency in (8.0, 10.0, 12.0)
    ]


def test_fbdcca_separates_reference_aligned_toy_trials() -> None:
    refs = _references()
    trials = np.stack(
        [
            np.stack([ref + 0.01, 0.5 * ref], axis=0)
            for ref in refs
        ]
    )
    model = FBDCCA(refs, weights=[1.0, 0.5])

    pred, scores = model.predict(trials)

    np.testing.assert_array_equal(pred, np.arange(3))
    assert scores.shape == (3, 3)
    assert np.all(np.isfinite(scores))


def test_fbdcca_validates_input_contract() -> None:
    refs = _references()
    model = FBDCCA(refs, weights=[1.0])

    with pytest.raises(ValueError, match="subband"):
        model.predict(np.zeros((1, 2, 2, 250)))
    with pytest.raises(ValueError, match="samples"):
        model.predict(np.zeros((1, 1, 2, 200)))
    with pytest.raises(ValueError, match="references"):
        FBDCCA([])
