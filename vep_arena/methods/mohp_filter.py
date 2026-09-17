from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from vep_arena.methods.trca_core import _solve_trca, corr_rows


def _center(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _correlation_matrices(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aa = _center(a)
    bb = _center(b)
    cross = (aa @ bb.T + bb @ aa.T) * 0.5
    return cross, aa @ aa.T, bb @ bb.T


def _quadratic_correlation(
    spatial_filter: np.ndarray, matrices: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    cross, auto_a, auto_b = matrices
    numerator = float(spatial_filter @ cross @ spatial_filter)
    denominator = np.sqrt(
        max(float(spatial_filter @ auto_a @ spatial_filter), 0.0)
        * max(float(spatial_filter @ auto_b @ spatial_filter), 0.0)
    )
    return numerator / denominator if denominator > 1e-12 else 0.0


def _weighted_sum_filter(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    non_targets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    channels = target[0].shape[0]
    projector = np.eye(channels) - np.ones((channels, channels)) / channels
    non_cross = np.mean([matrices[0] for matrices in non_targets], axis=0)
    non_auto = np.mean([matrices[1] for matrices in non_targets], axis=0)
    objective = projector @ (target[0] - non_cross) @ projector
    constraint = projector @ (target[1] + target[2] + non_auto) @ projector
    spatial_filter = projector @ _solve_trca(objective, constraint)
    norm = np.linalg.norm(spatial_filter)
    return spatial_filter / norm if norm > 1e-12 else spatial_filter


def mohp_filter(
    target_trials: np.ndarray,
    non_target_trials: list[np.ndarray],
    *,
    solver: str = "goal_attainment",
    max_iter: int = 100,
) -> np.ndarray:
    target = np.asarray(target_trials, dtype=np.float64)
    template = np.mean(target, axis=0)
    target_continuous = np.transpose(target, (1, 0, 2)).reshape(target.shape[1], -1)
    template_continuous = np.tile(template, (1, target.shape[0]))
    target_matrices = _correlation_matrices(target_continuous, template_continuous)
    non_matrices = []
    for trials in non_target_trials:
        continuous = np.transpose(trials, (1, 0, 2)).reshape(target.shape[1], -1)
        non_matrices.append(_correlation_matrices(continuous, template_continuous))
    initial = _weighted_sum_filter(target_matrices, non_matrices)
    if solver == "weighted_sum":
        return initial
    if solver != "goal_attainment":
        raise ValueError("solver must be 'goal_attainment' or 'weighted_sum'")

    def objective(value: np.ndarray) -> float:
        return float(value[-1])

    def objective_values(spatial_filter: np.ndarray) -> np.ndarray:
        target_value = -_quadratic_correlation(spatial_filter, target_matrices)
        non_values = [_quadratic_correlation(spatial_filter, item) for item in non_matrices]
        return np.asarray([target_value, *non_values])

    result = minimize(
        objective,
        np.concatenate([initial, [float(np.max(objective_values(initial)))]]),
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * len(initial) + [(-2.0, 2.0)],
        constraints=[
            {"type": "eq", "fun": lambda value: float(np.sum(value[:-1]))},
            {"type": "eq", "fun": lambda value: float(np.dot(value[:-1], value[:-1]) - 1.0)},
            {
                "type": "ineq",
                "fun": lambda value: value[-1] - objective_values(value[:-1]),
            },
        ],
        options={"maxiter": int(max_iter), "ftol": 1e-7, "disp": False},
    )
    spatial_filter = result.x[:-1] if result.success else initial
    spatial_filter = spatial_filter - np.mean(spatial_filter)
    norm = np.linalg.norm(spatial_filter)
    return spatial_filter / norm if norm > 1e-12 else initial


class MultiObjectiveHighPassFilter:
    """Multi-objective high-pass spatial filtering SSVEP classifier."""

    name = "MOHP"

    def __init__(
        self,
        n_fbs: int = 1,
        *,
        solver: str = "goal_attainment",
        max_iter: int = 100,
        ensemble: bool = False,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.solver = solver
        self.max_iter = int(max_iter)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "MultiObjectiveHighPassFilter":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        _, n_fbs, channels, samples = x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        class_trials = [x[y == cls] for cls in range(classes)]
        for cls in range(classes):
            self.templates[cls] = np.mean(class_trials[cls], axis=0)
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = mohp_filter(
                    class_trials[cls][:, fb_idx],
                    [class_trials[other][:, fb_idx] for other in range(classes) if other != cls],
                    solver=self.solver,
                    max_iter=self.max_iter,
                )
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("MOHP model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        band_scores = np.zeros(
            (epochs.shape[0], epochs.shape[1], self.templates.shape[0]), dtype=np.float64
        )
        for fb_idx in range(epochs.shape[1]):
            if self.ensemble:
                spatial_filter = self.filters[fb_idx].T
                projected_trials = (epochs[:, fb_idx].transpose(0, 2, 1) @ spatial_filter).reshape(
                    epochs.shape[0], -1
                )
                for cls in range(self.templates.shape[0]):
                    projected_template = (self.templates[cls, fb_idx].T @ spatial_filter).reshape(-1)
                    band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)
            else:
                for cls in range(self.templates.shape[0]):
                    spatial_filter = self.filters[fb_idx, cls]
                    projected_trials = epochs[:, fb_idx].transpose(0, 2, 1) @ spatial_filter
                    projected_template = self.templates[cls, fb_idx].T @ spatial_filter
                    band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)
        scores = np.einsum("f,tfc->tc", self.weights[: epochs.shape[1]], band_scores)
        return np.argmax(scores, axis=1), scores
