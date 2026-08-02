"""Clustering algorithm implementations.

This module provides K-means and geometric expectile clustering following Algorithm 1 of the
thesis. The expectile algorithm supports pluggable index vector strategies, tie-breaking rules,
and empty-cluster policies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geomexp.clustering.clustering_base import (
    BaseClusterer,
    ClusterResult,
    IterativeClusterer,
    expectile_loss,
)
from geomexp.clustering.geometry import EuclideanGeometry
from geomexp.clustering.index_strategies import BestResponseIndexStrategy, IndexStrategy
from geomexp.clustering.policies import (
    EmptyClusterRule,
    LowestIndexTieBreak,
    RandomReinitRule,
    TieBreakRule,
)
from geomexp.utils.validation import (
    validate_index_radius,
    validate_positive_float,
    validate_positive_int,
)

if TYPE_CHECKING:
    from geomexp.clustering.geometry import HilbertGeometry


class KMeans(IterativeClusterer):
    """K-means clustering algorithm.

    Partitions data into :math:`K` clusters by minimising the within-cluster sum of squared
    distances via Lloyd's alternating minimisation:

    .. math::
        J = \\sum_{k=1}^K \\sum_{i \\in C_k} \\|x_i - m_k\\|^2

    where :math:`C_k` denotes the set of points assigned to cluster :math:`k`, and :math:`m_k`
    is the cluster mean.

    Attributes:
        n_clusters: Number of clusters to form.
        n_init: Number of random restarts (best result is kept).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on objective change.
        random_state: Random seed for reproducibility; ``None`` draws from system entropy.

    Example:
        >>> import numpy as np
        >>> X = np.random.randn(100, 2)
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> result = kmeans.fit(X)
        >>> print(result.assignments.shape)
        (100,)
    """

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize the K-means clusterer.

        Args:
            n_clusters: Number of clusters to form. Must be positive.
            n_init: Number of independent restarts with different random seeds. The run with the
                lowest objective is returned. Defaults to a single run, unlike
                :class:`GeometricExpectileClustering`, whose non-convex objective benefits more
                from restarts.
            max_iter: Maximum number of iterations per restart. Must be positive.
            tol: Convergence tolerance on objective change. Must be non-negative.
            random_state: Random seed for reproducibility; ``None`` draws from system entropy.

        Raises:
            ValueError: If parameters are invalid.
        """
        validate_positive_int(n_init, "n_init")
        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state
        )
        self.n_init = n_init

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Fit K-means, keeping the best of ``n_init`` random restarts.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            ClusterResult from the best restart.
        """
        return self._fit_best_of_restarts(self._validate_input(X), self.n_init)

    def _initialize(self, X: np.ndarray) -> dict[str, object]:
        """Initialize state with random centers and Euclidean assignment."""
        centers = self._initialize_centers_random(X)
        return {"centers": centers, "assignments": self._assign_to_nearest_centers(X, centers)}

    def _assignment_step(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Assign each point to nearest center by Euclidean distance."""
        centers = state["centers"]
        assert isinstance(centers, np.ndarray)
        state["assignments"] = self._assign_to_nearest_centers(X, centers)
        return state

    def _update_step(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Update centers as cluster means.

        .. math::
            m_k = \\frac{1}{|C_k|} \\sum_{i \\in C_k} x_i
        """
        centers = state["centers"]
        assignments = state["assignments"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(assignments, np.ndarray)

        new_centers = centers.copy()
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)

        state["centers"] = new_centers
        return state

    def _compute_objective(self, X: np.ndarray, state: dict[str, object]) -> float:
        """Compute within-cluster sum of squared distances."""
        centers = state["centers"]
        assignments = state["assignments"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(assignments, np.ndarray)

        objective: float = 0
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.any(mask):
                objective += float(np.sum((X[mask] - centers[k]) ** 2))
        return objective

    def _extract_result(
        self, state: dict[str, object], objective: float, n_iterations: int, converged: bool
    ) -> ClusterResult:
        assignments = state["assignments"]
        centers = state["centers"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(centers, np.ndarray)
        return ClusterResult(
            assignments=assignments,
            centers=centers,
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
        )


class GeometricExpectileClustering(BaseClusterer):
    """Geometric expectile clustering with directional asymmetry (Algorithm 1).

    A Lloyd-type block coordinate descent algorithm that minimises the empirical risk based on the
    geometric expectile loss:

    .. math::
        \\hat{\\mathcal{R}}(\\Pi, C, U)
        = \\frac{1}{n} \\sum_{k=1}^K \\sum_{i=1}^n
          \\ell_{u_k}(X_i - c_k)\\,\\mathbb{1}_{\\{X_i \\in C_k\\}},
        \\quad \\|u_k\\| = r \\;\\forall k.

    Each iteration consists of:

    1. **Assignment** -- assign each point to the cluster minimising
       :math:`\\ell_{u_k}(X_i - c_k)`.
    2. **Empty-cluster handling** -- reinitialise empty clusters, then re-assign.
    3. **Centroid update** -- for each cluster, minimise the cluster-wise risk over :math:`c` via
       gradient descent.
    4. **Index update** -- update :math:`U` via the chosen
       :class:`~geomexp.clustering.index_strategies.IndexStrategy`.

    Setting :math:`r = 0` recovers standard K-means.

    Attributes:
        n_clusters: Number of clusters.
        index_radius: Radius constraint :math:`r \\in [0, 1)`.
        index_strategy: Strategy for index vector selection.
        tie_break_rule: Rule for breaking ties in assignment.
        empty_cluster_rule: Policy for handling empty clusters.
        n_init: Number of random restarts (best result is kept).
        max_iter: Maximum number of outer iterations.
        tol: Convergence tolerance on objective change.
        center_lr: Learning rate for gradient descent on centers.
        center_steps: Number of gradient steps per centroid update.
        random_state: Random seed for reproducibility; ``None`` draws from system entropy.

    Example:
        >>> import numpy as np
        >>> X = np.random.randn(100, 2)
        >>> model = GeometricExpectileClustering(n_clusters=3, index_radius=0.5)
        >>> result = model.fit(X)
        >>> print(result.metadata["indices"].shape)
        (3, 2)
    """

    def __init__(
        self,
        n_clusters: int,
        index_radius: float = 0.5,
        index_strategy: IndexStrategy | None = None,
        tie_break_rule: TieBreakRule | None = None,
        empty_cluster_rule: EmptyClusterRule | None = None,
        geometry: HilbertGeometry | None = None,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-7,
        center_lr: float = 0.1,
        center_steps: int = 50,
        random_state: int | None = None,
    ) -> None:
        """Initialize geometric expectile clusterer.

        Args:
            n_clusters: Number of clusters to form. Must be positive.
            index_radius: Radius constraint :math:`r \\in [0, 1)` on index vectors,
                :math:`\\|u_k\\| = r`.
            index_strategy: Strategy for choosing index vectors. Defaults to
                :class:`~geomexp.clustering.index_strategies.BestResponseIndexStrategy`.
            tie_break_rule: Rule for breaking assignment ties. Defaults to
                :class:`~geomexp.clustering.policies.LowestIndexTieBreak`.
            empty_cluster_rule: Policy for handling empty clusters. Defaults to
                :class:`~geomexp.clustering.policies.RandomReinitRule`.
            geometry: Hilbert space geometry for inner products and norms. Defaults to
                :class:`~geomexp.clustering.geometry.EuclideanGeometry`. Pass
                :class:`~geomexp.clustering.geometry.WeightedEuclideanGeometry` for functional
                data with quadrature weights, or :class:`~geomexp.clustering.geometry.GramGeometry`
                for basis coefficient representations.
            n_init: Number of independent restarts with different random seeds. The run with the
                lowest objective is returned.
            max_iter: Maximum number of iterations per restart. Must be positive.
            tol: Convergence tolerance. Must be non-negative.
            center_lr: Learning rate for gradient descent on centers. Must be positive. Because
                the center update uses the Hilbert gradient (natural gradient), this rate is
                independent of grid resolution even for functional data.
            center_steps: Number of gradient steps per centroid update. Must be positive.
            random_state: Random seed for reproducibility; ``None`` draws from system entropy.

        Raises:
            ValueError: If parameters are invalid.
        """
        validate_index_radius(index_radius)
        validate_positive_float(center_lr, "center_lr")
        validate_positive_int(center_steps, "center_steps")
        validate_positive_int(n_init, "n_init")

        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state
        )

        self.index_radius = index_radius
        self.geometry: HilbertGeometry = geometry or EuclideanGeometry()
        self.index_strategy = index_strategy or BestResponseIndexStrategy(geometry=self.geometry)
        self.tie_break_rule = tie_break_rule or LowestIndexTieBreak()
        self.empty_cluster_rule = empty_cluster_rule or RandomReinitRule()
        self.n_init = n_init
        self.center_lr = center_lr
        self.center_steps = center_steps

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Fit geometric expectile clustering with multiple random restarts.

        Runs ``n_init`` independent trials with different random seeds and returns the result
        with the lowest objective.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            ClusterResult from the best restart.
        """
        return self._fit_best_of_restarts(self._validate_input(X), self.n_init)

    def _initialize(self, X: np.ndarray) -> dict[str, object]:
        """Initialize centers, index vectors, and assignments.

        Centers are initialised to random data points; index vectors are initialised by the
        chosen strategy; initial assignment uses Hilbert-space distance.
        """
        centers = self._initialize_centers_random(X)
        indices = self.index_strategy.initialize(
            self.n_clusters, X.shape[1], self.index_radius, self._rng
        )

        distances_sq = np.empty((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances_sq[:, k] = self.geometry.squared_norm(X - centers[k])

        return {
            "centers": centers,
            "indices": indices,
            "assignments": np.argmin(distances_sq, axis=1),
            "prev_assignments": None,
        }

    def _fit_iteration(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """One outer iteration: assign, handle empties, update centers, update indices.

        After updating centers and indices, a final re-assignment is performed so that the
        partition stability convergence check compares assignments under the old and new
        parameters.
        """
        assignments = state["assignments"]
        assert isinstance(assignments, np.ndarray)
        state["prev_assignments"] = assignments.copy()

        state = self._assignment_step(X, state)
        state = self._handle_empty_clusters_in_state(X, state)
        state = self._update_centers(X, state)
        state = self._update_indices(X, state)
        return self._assignment_step(X, state)

    def _assignment_step(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Assign each point to the cluster minimising the expectile loss.

        .. math::
            k^*(i) \\in \\arg\\min_{k} \\ell_{u_k}(X_i - c_k)
        """
        centers = state["centers"]
        indices = state["indices"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(indices, np.ndarray)

        costs = np.empty((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            costs[:, k] = expectile_loss(X - centers[k], indices[k], geometry=self.geometry)

        state["assignments"] = self.tie_break_rule(costs, self._rng)
        return state

    def _handle_empty_clusters_in_state(
        self, X: np.ndarray, state: dict[str, object]
    ) -> dict[str, object]:
        """Apply empty-cluster policy and re-run assignment (Algorithm 1, Step 3)."""
        assignments = state["assignments"]
        centers = state["centers"]
        indices = state["indices"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(centers, np.ndarray)
        assert isinstance(indices, np.ndarray)

        if not any(np.sum(assignments == k) == 0 for k in range(self.n_clusters)):
            return state

        state["centers"], state["indices"] = self.empty_cluster_rule(
            X, assignments, centers, indices, self._rng
        )
        return self._assignment_step(X, state)

    def _update_centers(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Update each center by minimising cluster-wise risk via gradient descent.

        For each cluster, the center is updated by taking gradient steps on the cluster-wise
        empirical risk with respect to :math:`c_k`.
        """
        centers = state["centers"]
        assignments = state["assignments"]
        indices = state["indices"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(assignments, np.ndarray)
        assert isinstance(indices, np.ndarray)

        new_centers = centers.copy()
        for k in range(self.n_clusters):
            mask = assignments == k
            cluster_points = X[mask]

            if len(cluster_points) == 0:
                continue
            if len(cluster_points) == 1:
                new_centers[k] = cluster_points[0]
                continue

            c_k = new_centers[k].copy()
            for _ in range(self.center_steps):
                c_k -= self.center_lr * self._center_gradient(cluster_points, c_k, indices[k])
            new_centers[k] = c_k

        state["centers"] = new_centers
        return state

    def _center_gradient(
        self, cluster_points: np.ndarray, center: np.ndarray, u_k: np.ndarray
    ) -> np.ndarray:
        """Hilbert gradient of cluster-wise risk w.r.t. center.

        The Hilbert gradient of the expectile loss :math:`\\ell_u(x_i - c)` w.r.t. :math:`c` is:

        .. math::
            \\nabla_{H,c}\\, \\ell_u(x_i - c)
            = -\\hat{t}_i \\bigl(\\|t_i\\|_H + \\tfrac{1}{2}\\langle u, t_i \\rangle_H\\bigr)
              - \\tfrac{1}{2}\\|t_i\\|_H\\, u

        where :math:`t_i = x_i - c`, :math:`\\hat{t}_i = t_i / \\|t_i\\|_H`, and all norms /
        inner products are in the Hilbert space defined by ``self.geometry``. This is the Riesz
        representative of the Frechet derivative, which coincides with the Euclidean gradient when
        the geometry is Euclidean. Using the Hilbert gradient makes the learning rate independent
        of the coordinate representation (e.g. grid resolution for functional data).

        Args:
            cluster_points: Points in the cluster, shape ``(n_k, d)``.
            center: Current center, shape ``(d,)``.
            u_k: Index vector, shape ``(d,)``.

        Returns:
            Mean Hilbert gradient vector of shape ``(d,)``.
        """
        residuals = cluster_points - center
        dists = self.geometry.norm(residuals)
        nonzero = dists > 1e-12
        if not np.any(nonzero):
            return np.zeros_like(center)

        res, d = residuals[nonzero], dists[nonzero]
        t_hat = res / d[:, np.newaxis]
        coeff = d + 0.5 * self.geometry.inner(u_k, res)
        return np.asarray(
            -(np.sum(t_hat * coeff[:, np.newaxis], axis=0) + 0.5 * np.sum(d) * u_k)
            / len(cluster_points)
        )

    def _update_indices(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """Update index vectors via the chosen strategy (Algorithm 1, Step 5)."""
        centers = state["centers"]
        assignments = state["assignments"]
        indices = state["indices"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(assignments, np.ndarray)
        assert isinstance(indices, np.ndarray)
        state["indices"] = self.index_strategy.update(
            X, assignments, centers, indices, self.index_radius
        )
        return state

    def _compute_objective(self, X: np.ndarray, state: dict[str, object]) -> float:
        """Compute total empirical risk.

        .. math::
            \\hat{\\mathcal{R}} = \\frac{1}{n} \\sum_{k=1}^K
            \\sum_{i : X_i \\in C_k} \\ell_{u_k}(X_i - c_k)
        """
        centers = state["centers"]
        indices = state["indices"]
        assignments = state["assignments"]
        assert isinstance(centers, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(assignments, np.ndarray)

        objective: float = 0
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.any(mask):
                objective += float(
                    np.sum(expectile_loss(X[mask] - centers[k], indices[k], geometry=self.geometry))
                )
        return objective / len(X)

    def _additional_convergence_check(self, state: dict[str, object]) -> bool:
        """Check partition stability: stop if assignments unchanged."""
        prev = state.get("prev_assignments")
        curr = state.get("assignments")
        if isinstance(prev, np.ndarray) and isinstance(curr, np.ndarray):
            return bool(np.array_equal(prev, curr))
        return False

    def _extract_result(
        self, state: dict[str, object], objective: float, n_iterations: int, converged: bool
    ) -> ClusterResult:
        assignments = state["assignments"]
        centers = state["centers"]
        indices = state["indices"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(centers, np.ndarray)
        assert isinstance(indices, np.ndarray)
        return ClusterResult(
            assignments=assignments,
            centers=centers,
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
            metadata={"indices": indices},
        )
