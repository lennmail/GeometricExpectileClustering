"""Index vector strategies for geometric expectile clustering.

The index vectors :math:`(u_1, \\ldots, u_K)` control the directional asymmetry of the geometric
expectile loss and hence the cluster geometry. This module provides several strategies for choosing
these vectors, as described in Section 2.1.2 of the thesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from geomexp.clustering.geometry import EuclideanGeometry

if TYPE_CHECKING:
    from geomexp.clustering.geometry import HilbertGeometry


class IndexStrategy(ABC):
    """Abstract base class for index vector selection strategies.

    Index vectors :math:`u_k` with :math:`\\|u_k\\|_H = r` determine the directional asymmetry
    of the geometric expectile loss. Different strategies offer different trade-offs between
    flexibility and the need for prior knowledge.
    """

    @abstractmethod
    def initialize(
        self,
        n_clusters: int,
        n_features: int,
        index_radius: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Create initial index vectors.

        Args:
            n_clusters: Number of clusters.
            n_features: Dimensionality of the data.
            index_radius: Radius constraint :math:`r` with :math:`\\|u_k\\|_H = r`.
            rng: Random state for stochastic initialization.

        Returns:
            Array of shape ``(n_clusters, n_features)``.
        """

    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        index_radius: float,
    ) -> np.ndarray:
        """Update index vectors given the current clustering state.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.
            assignments: Cluster assignments of shape ``(n_samples,)``.
            centers: Cluster centers of shape ``(n_clusters, n_features)``.
            indices: Current index vectors of shape ``(n_clusters, n_features)``.
            index_radius: Radius constraint :math:`r`.

        Returns:
            Updated index vectors of shape ``(n_clusters, n_features)``.
        """


class GlobalIndexStrategy(IndexStrategy):
    """Fixed single global index vector shared by all clusters.

    Sets :math:`u_1 = \\cdots = u_K = u` for a given direction :math:`u`. Appropriate when a
    meaningful asymmetry direction is known a priori; one implicitly assumes a common anisotropy
    mechanism across clusters.

    The provided direction is normalised to have Hilbert norm equal to :math:`r`.
    """

    def __init__(
        self,
        direction: np.ndarray,
        geometry: HilbertGeometry | None = None,
    ) -> None:
        """Initialize with a direction vector.

        Args:
            direction: Direction vector of shape ``(n_features,)``. Will be normalised
                to :math:`\\|u\\|_H = r`.
            geometry: Hilbert space geometry for normalization. ``None`` uses Euclidean.

        Raises:
            ValueError: If direction is the zero vector.
        """
        self._geometry: HilbertGeometry = geometry or EuclideanGeometry()
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(self._geometry.norm(direction))
        if norm < 1e-12:
            raise ValueError("Direction must be non-zero")
        self._unit_direction = direction / norm

    def initialize(
        self,
        n_clusters: int,
        n_features: int,
        index_radius: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if len(self._unit_direction) != n_features:
            raise ValueError(
                f"Direction has {len(self._unit_direction)} components "
                f"but data has {n_features} features"
            )
        u = self._unit_direction * index_radius
        return np.tile(u, (n_clusters, 1))

    def update(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        index_radius: float,
    ) -> np.ndarray:
        return indices.copy()


class ClusterSpecificIndexStrategy(IndexStrategy):
    """Fixed per-cluster index vectors.

    Distinct fixed directions are prescribed for different clusters, each normalised to
    :math:`\\|u_k\\|_H = r`. Appropriate when prior information about cluster-specific
    asymmetry directions is available.

    Zero-norm rows in the provided directions result in :math:`u_k = 0`.
    """

    def __init__(
        self,
        directions: np.ndarray,
        geometry: HilbertGeometry | None = None,
    ) -> None:
        """Initialize with per-cluster direction vectors.

        Args:
            directions: Array of shape ``(n_clusters, n_features)``. Each row is normalised
                to :math:`\\|u_k\\|_H = r`; zero rows yield :math:`u_k = 0`.
            geometry: Hilbert space geometry for normalization. ``None`` uses Euclidean.
        """
        self._geometry: HilbertGeometry = geometry or EuclideanGeometry()
        directions = np.asarray(directions, dtype=np.float64)
        norms = self._geometry.norm(directions)
        self._is_nonzero = norms > 1e-12
        safe_norms = np.where(self._is_nonzero, norms, 1)
        self._unit_directions = directions / safe_norms[:, np.newaxis]

    def initialize(
        self,
        n_clusters: int,
        n_features: int,
        index_radius: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if len(self._unit_directions) != n_clusters:
            raise ValueError(
                f"Expected {n_clusters} direction vectors, got {len(self._unit_directions)}"
            )
        indices = self._unit_directions * index_radius
        indices[~self._is_nonzero] = 0
        return np.asarray(indices)

    def update(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        index_radius: float,
    ) -> np.ndarray:
        return indices.copy()


class BestResponseIndexStrategy(IndexStrategy):
    """Adaptive best-response index vectors (equation 2.2).

    Each index vector is set to minimise the cluster-wise risk for fixed partition and centroids.
    The closed-form solution in a Hilbert space :math:`H` is:

    .. math::
        u_k^* = -r \\frac{s_k}{\\|s_k\\|_H},
        \\quad s_k = \\sum_{i : X_i \\in C_k} \\|X_i - c_k\\|_H \\, (X_i - c_k).

    This captures the length-weighted residual direction (skew) of the cluster. When
    :math:`s_k = 0`, the current index is retained.
    """

    def __init__(self, geometry: HilbertGeometry | None = None) -> None:
        """Initialize best-response strategy.

        Args:
            geometry: Hilbert space geometry for norms. ``None`` uses Euclidean.
        """
        self._geometry: HilbertGeometry = geometry or EuclideanGeometry()

    def initialize(
        self,
        n_clusters: int,
        n_features: int,
        index_radius: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        return np.zeros((n_clusters, n_features))

    def update(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        index_radius: float,
    ) -> np.ndarray:
        n_clusters = len(centers)
        new_indices = indices.copy()

        for k in range(n_clusters):
            mask = assignments == k
            if not np.any(mask):
                continue

            residuals = X[mask] - centers[k]
            dists = self._geometry.norm(residuals)[:, np.newaxis]

            s_k = np.sum(dists * residuals, axis=0)
            norm_s = float(self._geometry.norm(s_k))

            if norm_s > 1e-12:
                new_indices[k] = -index_radius * s_k / norm_s

        return new_indices


class CustomIndexStrategy(IndexStrategy):
    """User-provided callable for custom index vector updates.

    Allows full flexibility by accepting arbitrary callables for initialisation and update of index
    vectors.

    The update callable should have the signature::

        def update_fn(
            X: np.ndarray,            # (n_samples, n_features)
            assignments: np.ndarray,   # (n_samples,)
            centers: np.ndarray,       # (n_clusters, n_features)
            indices: np.ndarray,       # (n_clusters, n_features)
            index_radius: float,
        ) -> np.ndarray:               # (n_clusters, n_features)

    The optional init callable should have the signature::

        def init_fn(
            n_clusters: int,
            n_features: int,
            index_radius: float,
            rng: np.random.RandomState,
        ) -> np.ndarray:               # (n_clusters, n_features)
    """

    def __init__(
        self,
        update_fn: Callable[..., np.ndarray],
        init_fn: Callable[..., np.ndarray] | None = None,
    ) -> None:
        """Initialize with custom callables.

        Args:
            update_fn: Callable that computes updated index vectors.
            init_fn: Optional callable for initialisation. If ``None``, indices are
                initialised to zero.
        """
        self._update_fn = update_fn
        self._init_fn = init_fn

    def initialize(
        self,
        n_clusters: int,
        n_features: int,
        index_radius: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if self._init_fn is not None:
            return self._init_fn(n_clusters, n_features, index_radius, rng)
        return np.zeros((n_clusters, n_features))

    def update(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        indices: np.ndarray,
        index_radius: float,
    ) -> np.ndarray:
        return self._update_fn(X, assignments, centers, indices, index_radius)
