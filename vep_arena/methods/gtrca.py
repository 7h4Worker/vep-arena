from __future__ import annotations

import numpy as np

from vep_arena.methods.trca_core import _solve_trca, corr_flat


def _normalize(x: np.ndarray) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64)
    data = data - np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return np.divide(data, std, out=np.zeros_like(data), where=std > 1e-12)


def group_trca_filters(subject_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit subject-specific gTRCA filters for subjects x trials x channels x samples."""

    x = _normalize(subject_trials)
    subjects, trials, channels, samples = x.shape
    if trials < 2:
        raise ValueError("gTRCA requires at least two trials per training subject")
    means = np.mean(x, axis=1)
    second_moments = np.mean(x @ x.transpose(0, 1, 3, 2), axis=1)
    size = subjects * channels
    objective = np.zeros((size, size), dtype=np.float64)
    constraint = np.zeros_like(objective)
    for subject_a in range(subjects):
        row = slice(subject_a * channels, (subject_a + 1) * channels)
        diagonal = trials / ((trials - 1) * samples) * (
            means[subject_a] @ means[subject_a].T - second_moments[subject_a] / trials
        )
        objective[row, row] = 2.0 * diagonal
        continuous = x[subject_a].transpose(1, 0, 2).reshape(channels, -1)
        constraint[row, row] = continuous @ continuous.T / continuous.shape[1]
        for subject_b in range(subjects):
            if subject_a == subject_b:
                continue
            column = slice(subject_b * channels, (subject_b + 1) * channels)
            objective[row, column] = means[subject_a] @ means[subject_b].T / samples
    solution = _solve_trca(objective, constraint).reshape(subjects, channels)
    for subject in range(subjects):
        norm = np.linalg.norm(solution[subject])
        if norm > 1e-12:
            solution[subject] /= norm
    return solution, means


class gTRCA:
    """Group TRCA with Eq. (21) predictive filters for new subjects."""

    name = "GTRCA"

    def __init__(self, n_fbs: int = 1) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.subject_filters: np.ndarray | None = None
        self.subject_templates: np.ndarray | None = None
        self.group_components: np.ndarray | None = None
        self.source_sums: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "gTRCA":
        x = np.asarray(train_x, dtype=np.float64)
        if x.ndim == 4:
            x = x[None, ...]
        if x.ndim != 5:
            raise ValueError(
                "train_x must have shape subjects x trials x subbands x channels x samples"
            )
        subjects, trials, n_fbs, channels, samples = x.shape
        labels = np.asarray(train_y, dtype=np.int64)
        if labels.ndim == 1:
            if labels.shape[0] != trials:
                raise ValueError("one-dimensional train_y must label the shared trial axis")
            labels = np.broadcast_to(labels[None, :], (subjects, trials))
        if labels.shape != (subjects, trials):
            raise ValueError("train_y must have shape trials or subjects x trials")
        classes = int(np.max(labels)) + 1
        normalized = _normalize(x)
        self.subject_filters = np.zeros(
            (n_fbs, classes, subjects, channels), dtype=np.float64
        )
        self.subject_templates = np.zeros(
            (n_fbs, classes, subjects, channels, samples), dtype=np.float64
        )
        self.group_components = np.zeros((n_fbs, classes, samples), dtype=np.float64)
        self.source_sums = np.zeros_like(self.group_components)
        for cls in range(classes):
            counts = [int(np.sum(labels[subject] == cls)) for subject in range(subjects)]
            if len(set(counts)) != 1 or counts[0] < 2:
                raise ValueError("every subject must provide the same >=2 trials per class")
            for fb_idx in range(n_fbs):
                class_data = np.asarray(
                    [normalized[subject, labels[subject] == cls, fb_idx] for subject in range(subjects)]
                )
                filters, templates = group_trca_filters(class_data)
                self.subject_filters[fb_idx, cls] = filters
                self.subject_templates[fb_idx, cls] = templates
                components = np.einsum("sc,sct->st", filters, templates)
                self.group_components[fb_idx, cls] = np.mean(components, axis=0)
                self.source_sums[fb_idx, cls] = np.sum(components, axis=0)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.group_components is None or self.source_sums is None:
            raise RuntimeError("gTRCA model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        leading_shape = epochs.shape[:-3]
        flat = _normalize(epochs.reshape(-1, *epochs.shape[-3:]))
        scores = np.zeros((flat.shape[0], self.group_components.shape[1]), dtype=np.float64)
        for trial_idx, trial in enumerate(flat):
            for cls in range(self.group_components.shape[1]):
                score = 0.0
                for fb_idx in range(trial.shape[0]):
                    band = trial[fb_idx]
                    covariance = band @ band.T / band.shape[-1]
                    predictive_filter = np.linalg.pinv(
                        covariance + np.eye(covariance.shape[0]) * 1e-8
                    ) @ band @ self.source_sums[fb_idx, cls]
                    norm = np.linalg.norm(predictive_filter)
                    if norm > 1e-12:
                        predictive_filter /= norm
                    component = predictive_filter @ band
                    score += self.weights[fb_idx] * corr_flat(
                        component, self.group_components[fb_idx, cls]
                    )
                scores[trial_idx, cls] = score
        predictions = np.argmax(scores, axis=1)
        return predictions.reshape(leading_shape), scores.reshape(*leading_shape, scores.shape[-1])
