"""Tie-breaking and empty cluster policies for clustering algorithms.

This module provides pluggable policy objects that control tie-breaking during cluster assignment
and the handling of empty clusters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from geomexp.clustering.geometry import EuclideanGeometry

if TYPE_CHECKING:
    from geomexp.clustering.geometry import HilbertGeometry


class TieBreakRule(ABC):
    """Abstract base class for tie-breaking rules in cluster assignment.

    When a data point has equal cost to multiple clusters, the tie-break rule determines which
    cluster is chosen.
    """

    @abstractmethod
    def __call__(self, costs: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Resolve a cost matrix to cluster assignments.

        Args:
            costs: Cost matrix of shape ``(n_samples, n_clusters)``.
            rng: Random state for stochastic rules.

        Returns:
            Assignment array of shape ``(n_samples,)`` with integer cluster labels.
        """


class LowestIndexTieBreak(TieBreakRule):
    """Break ties by choosing the lowest cluster index.

    This is the default behavior of ``np.argmin`` and provides deterministic, reproducible
    assignment independent of the random state.
    """

    def __call__(self, costs: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        return np.asarray(np.argmin(costs, axis=1))


class RandomTieBreak(TieBreakRule):
    """Break ties uniformly at random among clusters achieving minimal cost."""

    def __call__(self, costs: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        min_costs = costs.min(axis=1, keepdims=True)
        is_min = np.isclose(costs, min_costs)
        n_samples = len(costs)
        assignments = np.empty(n_samples, dtype=np.intp)
        for i in range(n_samples):
            candidates = np.where(is_min[i])[0]
            assignments[i] = rng.choice(candidates)
        return assignments


class EmptyClusterRule(ABC):
    """Abstract base class for handling empty clusters.

    After the assignment step, some clusters may become empty. This policy determines how to
    reinitialize their centers and index vectors so that the subsequent reassignment can populate
    them.
    """

    @abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reinitialize empty cluster centers and index vectors.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.
            assignments: Current assignments of shape ``(n_samples,)``.
            centers: Cluster centers of shape ``(n_clusters, n_features)``.
            indices: Index vectors of shape ``(n_clusters, n_features)``.
            rng: Random state.

        Returns:
            Tuple ``(updated_centers, updated_indices)``. The caller is responsible for re-running
            the assignment step afterwards.
        """


class RandomReinitRule(EmptyClusterRule):
    """Reinitialize each empty cluster center to a randomly chosen data point.

    Index vectors for reinitialized clusters are reset to zero (symmetric). This is the simplest
    and most common approach.
    """

    def __call__(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        centers = centers.copy()
        indices = indices.copy()
        n_clusters = len(centers)

        for k in range(n_clusters):
            if np.sum(assignments == k) == 0:
                centers[k] = X[rng.choice(len(X))]
                indices[k] = 0

        return centers, indices


class FarthestPointRule(EmptyClusterRule):
    """Reinitialize each empty cluster to the point farthest from its assigned center.

    Distances are measured in the Hilbert geometry provided at construction (Euclidean by default).
    This tends to spread clusters apart and can help escape degenerate local optima more
    aggressively than random reinitialization.

    When several clusters are empty they are seeded with successively farther points, so that no
    two of them receive the same center.
    """

    def __init__(self, geometry: HilbertGeometry | None = None) -> None:
        """Initialize farthest-point rule.

        Args:
            geometry: Hilbert space geometry for distances. ``None`` uses Euclidean.
        """
        self._geometry: HilbertGeometry = geometry or EuclideanGeometry()

    def __call__(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        centers = centers.copy()
        indices = indices.copy()

        empty = [k for k in range(len(centers)) if np.sum(assignments == k) == 0]
        if not empty:
            return centers, indices

        dists = self._geometry.norm(X - centers[assignments])
        for k in empty:
            farthest = int(np.argmax(dists))
            centers[k] = X[farthest]
            indices[k] = 0
            dists[farthest] = -np.inf

        return centers, indices
