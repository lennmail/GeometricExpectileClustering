"""Base classes and utilities for clustering algorithms.

This module provides abstract base classes, common utilities, and the geometric expectile loss
function for implementing clustering algorithms with a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from geomexp.utils.validation import (
    validate_data_array,
    validate_n_clusters,
    validate_positive_int,
    validate_tolerance,
)

if TYPE_CHECKING:
    from geomexp.clustering.geometry import HilbertGeometry


def expectile_loss(
    residuals: np.ndarray,
    index_vectors: np.ndarray,
    geometry: HilbertGeometry | None = None,
) -> np.ndarray:
    """Compute the geometric expectile loss.

    The loss function is:

    .. math::
        \\ell_u(t) = \\frac{1}{2}\\bigl(\\|t\\|_H^2 + \\|t\\|_H\\,\\langle u, t \\rangle_H\\bigr)

    for :math:`(t, u) \\in \\mathcal{X} \\times \\mathcal{B}`, as introduced by Herrmann et al.
    (2018). When ``geometry`` is ``None`` the standard Euclidean inner product is used.

    Args:
        residuals: Residual vectors :math:`t = x - c`, shape ``(n, d)`` or ``(d,)``.
        index_vectors: Index vector :math:`u`, broadcastable to ``residuals``. Shape ``(d,)``
            for a single shared index, or ``(n, d)`` for per-sample indices.
        geometry: Hilbert space geometry. ``None`` (default) uses Euclidean.

    Returns:
        Loss values of shape ``(n,)`` (or scalar if input was 1-D).
    """
    squeeze = residuals.ndim == 1
    if squeeze:
        residuals = residuals[np.newaxis, :]

    if geometry is None:
        norms = np.linalg.norm(residuals, axis=1)
        inner_products = np.sum(index_vectors * residuals, axis=-1)
    else:
        norms = geometry.norm(residuals)
        inner_products = geometry.inner(index_vectors, residuals)

    losses = 0.5 * (norms**2 + norms * inner_products)
    return np.asarray(losses[0] if squeeze else losses)


@dataclass
class ClusterResult:
    """Container for clustering results.

    Attributes:
        assignments: Array of shape ``(n_samples,)`` containing cluster indices for each point.
        centers: Array of shape ``(n_clusters, n_features)`` containing cluster centers.
        objective: Final objective function value achieved by the algorithm.
        n_iterations: Number of iterations performed until convergence.
        converged: Whether the algorithm converged within ``max_iter`` iterations.
        metadata: Optional dictionary containing algorithm-specific information.
    """

    assignments: np.ndarray
    centers: np.ndarray
    objective: float
    n_iterations: int
    converged: bool = True
    metadata: dict[str, object] | None = None


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms.

    Defines the interface that all clustering algorithms must implement. Subclasses provide
    ``_initialize``, ``_fit_iteration``, ``_compute_objective``, and ``_extract_result``.

    The ``fit`` method contains the outer optimisation loop with convergence checking based on
    objective change and an optional partition stability hook.

    Attributes:
        n_clusters: Number of clusters to form.
        max_iter: Maximum number of iterations to perform.
        tol: Convergence tolerance for objective function change.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-7,
        random_state: int | None = None,
    ) -> None:
        """Initialize the base clusterer.

        Args:
            n_clusters: Number of clusters to form. Must be positive.
            max_iter: Maximum number of iterations. Must be positive.
            tol: Convergence tolerance on objective change. Must be non-negative.
            random_state: Random seed for reproducibility.

        Raises:
            TypeError: If ``n_clusters`` or ``max_iter`` are not integers.
            ValueError: If ``n_clusters < 1``, ``max_iter < 1``, or ``tol < 0``.
        """
        validate_n_clusters(n_clusters)
        validate_positive_int(max_iter, "max_iter")
        validate_tolerance(tol)

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Fit the clustering algorithm to data.

        The outer loop follows Algorithm 1 from the thesis: iterate until the objective decrease
        falls below ``tol``, an additional convergence criterion is met, or ``max_iter`` iterations
        are reached.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            ClusterResult containing assignments, centers, and convergence info.

        Raises:
            ValueError: If ``X`` has fewer samples than ``n_clusters`` or invalid shape.
        """
        X = self._validate_input(X)
        state = self._initialize(X)

        obj_new = obj_old = self._compute_objective(X, state)
        converged = False
        n_iter = 0

        for n_iter in range(self.max_iter):  # noqa: B007
            state = self._fit_iteration(X, state)
            obj_new = self._compute_objective(X, state)

            if abs(obj_old - obj_new) <= self.tol or self._additional_convergence_check(state):
                converged = True
                break
            obj_old = obj_new

        return self._extract_result(state, obj_new, n_iter + 1, converged)

    def _additional_convergence_check(self, state: dict[str, object]) -> bool:
        """Hook for subclass-specific convergence criteria.

        Override in subclasses to add criteria such as partition stability. Called after each
        iteration, after the objective check.

        Args:
            state: Current algorithm state dictionary.

        Returns:
            ``True`` if the algorithm should stop, ``False`` otherwise.
        """
        return False

    @abstractmethod
    def _initialize(self, X: np.ndarray) -> dict[str, object]: ...

    @abstractmethod
    def _fit_iteration(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]: ...

    @abstractmethod
    def _compute_objective(self, X: np.ndarray, state: dict[str, object]) -> float: ...

    @abstractmethod
    def _extract_result(
        self, state: dict[str, object], objective: float, n_iterations: int, converged: bool
    ) -> ClusterResult: ...

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input data.

        Args:
            X: Input data array.

        Returns:
            Validated numpy array of shape ``(n_samples, n_features)``.

        Raises:
            ValueError: If input is invalid.
        """
        X = validate_data_array(X)
        validate_n_clusters(self.n_clusters, X.shape[0])
        return X

    def _initialize_centers_random(self, X: np.ndarray) -> np.ndarray:
        """Initialize centers by randomly selecting distinct data points.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_clusters, n_features)`` containing initial centers.
        """
        return X[self._rng.choice(len(X), size=self.n_clusters, replace=False)].copy()

    def _assign_to_nearest_centers(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each point to its nearest center using Euclidean distance.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.
            centers: Centers array of shape ``(n_clusters, n_features)``.

        Returns:
            Array of shape ``(n_samples,)`` containing cluster assignments.
        """
        return np.asarray(
            np.argmin(
                np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2), axis=1
            )
        )


class IterativeClusterer(BaseClusterer):
    """Base class for iterative clustering algorithms with assignment-update steps.

    Provides a template for Lloyd-type algorithms that alternate between assigning points to
    clusters, handling empty clusters, and updating cluster parameters.
    """

    def _fit_iteration(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Perform assignment, empty-cluster handling, and update steps."""
        state = self._assignment_step(X, state)
        state = self._handle_empty_clusters_in_state(X, state)
        state = self._update_step(X, state)
        return self._assignment_step(X, state)

    @abstractmethod
    def _assignment_step(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]: ...

    @abstractmethod
    def _update_step(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]: ...

    def _handle_empty_clusters_in_state(
        self, X: np.ndarray, state: dict[str, object]
    ) -> dict[str, object]:
        """Handle empty clusters by reinitialising them to random data points."""
        assignments = state["assignments"]
        centers = state["centers"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(centers, np.ndarray)

        if not any(np.sum(assignments == k) == 0 for k in range(self.n_clusters)):
            return state

        for k in range(self.n_clusters):
            if np.sum(assignments == k) == 0:
                centers[k] = X[self._rng.choice(len(X))]

        state["centers"] = centers
        state["assignments"] = self._assign_to_nearest_centers(X, centers)
        return state
