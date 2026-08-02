"""Evaluation metrics for clustering experiments.

Provides wrappers around scikit-learn metrics plus custom diagnostics for bootstrap stability and
cluster shape analysis.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from geomexp.clustering.clustering_base import BaseClusterer


def adjusted_rand_index(y_true: NDArray[np.intp], y_pred: NDArray[np.intp]) -> float:
    """Adjusted Rand Index between two partitions.

    Args:
        y_true: Ground-truth cluster labels.
        y_pred: Predicted cluster labels.

    Returns:
        ARI in :math:`[-1, 1]` (1 = perfect agreement).
    """
    return float(adjusted_rand_score(y_true, y_pred))


def normalized_mutual_info(y_true: NDArray[np.intp], y_pred: NDArray[np.intp]) -> float:
    """Normalised Mutual Information between two partitions.

    Args:
        y_true: Ground-truth cluster labels.
        y_pred: Predicted cluster labels.

    Returns:
        NMI in :math:`[0, 1]` (1 = perfect agreement).
    """
    return float(normalized_mutual_info_score(y_true, y_pred))


def silhouette(X: NDArray[np.float64], labels: NDArray[np.intp]) -> float:
    """Mean silhouette coefficient.

    Args:
        X: Data array of shape ``(n_samples, n_features)``.
        labels: Cluster labels.

    Returns:
        Silhouette score in :math:`[-1, 1]`.
    """
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= len(X):
        return 0
    return float(silhouette_score(X, labels))


def davies_bouldin(X: NDArray[np.float64], labels: NDArray[np.intp]) -> float:
    """Davies--Bouldin index (lower is better).

    Args:
        X: Data array of shape ``(n_samples, n_features)``.
        labels: Cluster labels.

    Returns:
        Davies--Bouldin score (non-negative; 0 is ideal).
    """
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= len(X):
        return float("inf")
    return float(davies_bouldin_score(X, labels))


def stability_score(
    X: NDArray[np.float64],
    clusterer_factory: Callable[[], BaseClusterer],
    n_resamples: int = 20,
    subsample_frac: float = 0.8,
    rng: np.random.Generator | None = None,
) -> float:
    """Bootstrap stability of a clustering method.

    Repeatedly draws sub-samples, fits the clusterer on each, and measures the mean pairwise ARI
    on the intersection of each pair of sub-samples.

    Args:
        X: Data array of shape ``(n_samples, n_features)``.
        clusterer_factory: Zero-argument callable returning a fresh clusterer instance.
        n_resamples: Number of bootstrap resamples.
        subsample_frac: Fraction of data to draw per resample.
        rng: Optional random generator.

    Returns:
        Mean pairwise ARI across resamples (higher = more stable).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(X)
    runs: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=max(2, int(subsample_frac * n)), replace=False)
        idx.sort()
        result = clusterer_factory().fit(X[idx])
        runs.append((idx, np.asarray(result.assignments, dtype=np.intp)))

    aris: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            idx_i, lab_i = runs[i]
            idx_j, lab_j = runs[j]
            shared = np.intersect1d(idx_i, idx_j)
            if len(shared) < 2:
                continue
            ari = adjusted_rand_score(lab_i[np.isin(idx_i, shared)], lab_j[np.isin(idx_j, shared)])
            aris.append(float(ari))

    return float(np.mean(aris)) if aris else 0


def radius_ratio(
    X: NDArray[np.float64],
    centers: NDArray[np.float64],
    assignments: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Per-cluster radius ratio: max distance / median distance to centroid.

    A large ratio indicates elongated or "tendril"-like cluster capture regions.

    Args:
        X: Data array of shape ``(n_samples, n_features)``.
        centers: Cluster centres of shape ``(n_clusters, n_features)``.
        assignments: Cluster labels of shape ``(n_samples,)``.

    Returns:
        Array of shape ``(n_clusters,)`` with the radius ratio for each cluster.
    """
    ratios = np.ones(len(centers))
    for k in range(len(centers)):
        mask = assignments == k
        if np.sum(mask) < 2:
            continue
        dists = np.linalg.norm(X[mask] - centers[k], axis=1)
        med = np.median(dists)
        if med > 1e-12:
            ratios[k] = float(np.max(dists)) / med
    return ratios


def variation_of_information(y_true: NDArray[np.intp], y_pred: NDArray[np.intp]) -> float:
    """Variation of Information between two partitions.

    Defined as :math:`\\mathrm{VI}(U, V) = H(U) + H(V) - 2\\,I(U, V)`, using natural logs.

    Args:
        y_true: Ground-truth cluster labels.
        y_pred: Predicted cluster labels.

    Returns:
        Non-negative VI, with 0 indicating identical partitions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty")

    n = y_true.shape[0]
    labels_true, inverse_true = np.unique(y_true, return_inverse=True)
    labels_pred, inverse_pred = np.unique(y_pred, return_inverse=True)

    contingency = np.zeros((len(labels_true), len(labels_pred)), dtype=np.float64)
    np.add.at(contingency, (inverse_true, inverse_pred), 1)
    contingency /= n

    p_true = contingency.sum(axis=1)
    p_pred = contingency.sum(axis=0)

    H_true = -float(np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0])))
    H_pred = -float(np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])))

    nz = contingency > 0
    mi = float(np.sum(contingency[nz] * np.log(contingency[nz] / np.outer(p_true, p_pred)[nz])))

    return float(max(0, H_true + H_pred - 2 * mi))


def misclassification_error(y_true: NDArray[np.intp], y_pred: NDArray[np.intp]) -> float:
    """Minimum misclassification error under optimal label permutation.

    Uses the Hungarian algorithm to find the permutation of predicted labels that maximises
    agreement with the ground truth.

    Args:
        y_true: Ground-truth cluster labels.
        y_pred: Predicted cluster labels.

    Returns:
        Misclassification rate in :math:`[0, 1]` (0 = perfect agreement).
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)))
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cost[i, j] = -np.sum((y_true == lt) & (y_pred == lp))

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(1 + cost[row_ind, col_ind].sum() / len(y_true))
