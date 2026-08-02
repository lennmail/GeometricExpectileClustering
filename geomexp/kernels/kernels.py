"""Kernel functions and kernelised clustering algorithms.

This module provides kernel function abstractions, a fully implemented kernel K-means algorithm,
and the kernelised geometric expectile clustering algorithm. In the dual representation, centroids
and index vectors are expressed as linear combinations of feature-mapped data points, so all
computations reduce to operations on the Gram matrix :math:`K_{ij} = k(X_i, X_j)`.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gamma, kv

from geomexp.clustering.clustering_base import BaseClusterer, ClusterResult
from geomexp.utils.validation import (
    validate_index_radius,
    validate_positive_float,
    validate_positive_int,
)


class Kernel(ABC):
    """Abstract base class for kernel functions.

    A kernel :math:`k : \\mathcal{X} \\times \\mathcal{X} \\to \\mathbb{R}` implicitly defines a
    feature map :math:`\\varphi` into a reproducing kernel Hilbert space :math:`\\mathcal{H}`
    such that :math:`k(x, y) = \\langle \\varphi(x), \\varphi(y) \\rangle_{\\mathcal{H}}`.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluate the kernel on all pairs of rows.

        Args:
            X: Array of shape ``(n_x, d)``.
            Y: Array of shape ``(n_y, d)``.

        Returns:
            Gram sub-matrix of shape ``(n_x, n_y)`` with entry :math:`K_{ij} = k(X_i, Y_j)`.
        """

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the full Gram matrix :math:`K_{ij} = k(X_i, X_j)`.

        Args:
            X: Data array of shape ``(n, d)``.

        Returns:
            Symmetric positive semi-definite matrix of shape ``(n, n)``.
        """
        return self(X, X)


class LinearKernel(Kernel):
    """Linear (inner product) kernel: :math:`k(x, y) = \\langle x, y \\rangle`."""

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.asarray(X @ Y.T)


class RBFKernel(Kernel):
    """Radial basis function (Gaussian) kernel.

    .. math::
        k(x, y) = \\exp\\!\\bigl(-\\gamma \\|x - y\\|^2\\bigr)

    Attributes:
        gamma: Bandwidth parameter :math:`\\gamma > 0`.
    """

    def __init__(self, gamma: float = 1) -> None:
        validate_positive_float(gamma, "gamma")
        self.gamma = gamma

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dist_sq = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * X @ Y.T
            + np.sum(Y**2, axis=1, keepdims=True).T
        )
        np.maximum(dist_sq, 0, out=dist_sq)
        return np.asarray(np.exp(-self.gamma * dist_sq))


class PolynomialKernel(Kernel):
    """Polynomial kernel.

    .. math::
        k(x, y) = (\\langle x, y \\rangle + c)^p

    Attributes:
        degree: Polynomial degree :math:`p`.
        coef0: Additive constant :math:`c`.
    """

    def __init__(self, degree: int = 3, coef0: float = 1) -> None:
        validate_positive_int(degree, "degree")
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.asarray((X @ Y.T + self.coef0) ** self.degree)


class MaternKernel(Kernel):
    """Matern kernel implementing the Matern covariance function.

    .. math::
        k(x, y) = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
        \\left(\\frac{\\|x - y\\|}{\\ell}\\right)^{\\nu}
        K_{\\nu}\\!\\left(\\frac{\\|x - y\\|}{\\ell}\\right)

    where :math:`K_{\\nu}` is the modified Bessel function of the second kind. Special cases:
    :math:`\\nu = 0.5` gives the exponential kernel, :math:`\\nu = 1.5` and :math:`\\nu = 2.5`
    have closed-form expressions, and :math:`\\nu \\to \\infty` recovers the RBF
    (squared-exponential) kernel.

    Attributes:
        nu: Smoothness parameter :math:`\\nu > 0`.
        lengthscale: Lengthscale parameter :math:`\\ell > 0`.
    """

    def __init__(self, nu: float = 1.5, lengthscale: float = 1) -> None:
        """Initialize Matern kernel.

        Args:
            nu: Smoothness parameter. Common values: 0.5, 1.5, 2.5, ``np.inf``.
            lengthscale: Lengthscale parameter (must be positive).
        """
        validate_positive_float(lengthscale, "lengthscale")
        self.nu = nu
        self.lengthscale = lengthscale

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.maximum(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1), 0))
        dists /= self.lengthscale

        if self.nu == 0.5:
            return np.asarray(np.exp(-dists))
        if self.nu == np.inf:
            return np.asarray(np.exp(-0.5 * dists**2))
        if self.nu == 1.5:
            sqrt3d = np.sqrt(3) * dists
            return np.asarray((1 + sqrt3d) * np.exp(-sqrt3d))
        if self.nu == 2.5:
            sqrt5d = np.sqrt(5) * dists
            return np.asarray((1 + sqrt5d + 5 * dists**2 / 3) * np.exp(-sqrt5d))

        K = np.ones_like(dists)
        nonzero = dists > 0
        factor = (2 ** (1 - self.nu)) / gamma(self.nu)
        K[nonzero] = factor * (dists[nonzero] ** self.nu) * kv(self.nu, dists[nonzero])
        return np.asarray(K)


def _kmeanspp_feature_space(
    gram: np.ndarray, n_clusters: int, rng: np.random.RandomState
) -> np.ndarray:
    """K-means++ initialisation in feature space using the precomputed Gram matrix.

    Args:
        gram: Gram matrix of shape ``(n, n)``.
        n_clusters: Number of seed indices to select.
        rng: Random state.

    Returns:
        Integer array of shape ``(n_clusters,)`` with selected data indices.
    """
    n = gram.shape[0]
    diag_K = np.diag(gram)
    init_indices = np.empty(n_clusters, dtype=int)
    init_indices[0] = rng.choice(n)

    for k in range(1, n_clusters):
        min_dist_sq = np.full(n, np.inf)
        for j in range(k):
            d2 = diag_K - 2 * gram[:, init_indices[j]] + diag_K[init_indices[j]]
            np.maximum(d2, 0, out=d2)
            np.minimum(min_dist_sq, d2, out=min_dist_sq)
        total = min_dist_sq.sum()
        if not np.isfinite(total) or total <= 0:
            remaining = np.setdiff1d(np.arange(n), init_indices[:k], assume_unique=False)
            init_indices[k] = rng.choice(remaining)
        else:
            probs = min_dist_sq / total
            init_indices[k] = rng.choice(n, p=probs)

    return init_indices


class KernelKMeans(BaseClusterer):
    """Kernel K-means clustering (dual representation).

    Minimises the within-cluster sum of squared distances in the feature space
    :math:`\\mathcal{H}` induced by a kernel :math:`k`:

    .. math::
        J = \\sum_{k=1}^K \\sum_{i \\in C_k} \\|\\varphi(X_i) - c_k\\|_{\\mathcal{H}}^2

    where :math:`c_k = \\frac{1}{|C_k|} \\sum_{j \\in C_k} \\varphi(X_j)` is the cluster
    centroid in feature space. Since :math:`\\varphi` may not have an explicit form, all
    computations are expressed through the Gram matrix :math:`K_{ij} = k(X_i, X_j)`:

    .. math::
        \\|\\varphi(X_i) - c_k\\|^2 = K_{ii}
        - \\frac{2}{|C_k|} \\sum_{j \\in C_k} K_{ij}
        + \\frac{1}{|C_k|^2} \\sum_{j,l \\in C_k} K_{jl}

    With a linear kernel this recovers standard K-means.

    Attributes:
        n_clusters: Number of clusters.
        kernel: Kernel function.
        n_init: Number of random restarts (best result is kept).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        random_state: Random seed.

    Example:
        >>> import numpy as np
        >>> from geomexp.kernels import KernelKMeans, RBFKernel
        >>> X = np.random.randn(100, 2)
        >>> model = KernelKMeans(n_clusters=3, kernel=RBFKernel(gamma=0.5))
        >>> result = model.fit(X)
        >>> print(result.assignments.shape)
        (100,)
    """

    def __init__(
        self,
        n_clusters: int,
        kernel: Kernel,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-7,
        random_state: int | None = None,
    ) -> None:
        """Initialize kernel K-means clusterer.

        Args:
            n_clusters: Number of clusters.
            kernel: Kernel function instance.
            n_init: Number of independent restarts. The run with the lowest objective is returned.
            max_iter: Maximum number of iterations per restart.
            tol: Convergence tolerance.
            random_state: Random seed.
        """
        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state
        )
        self.kernel = kernel
        self.n_init = n_init

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Fit kernel K-means with multiple random restarts.

        The Gram matrix is computed once and reused across all ``n_init`` restarts.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            ClusterResult from the best restart.
        """
        X = self._validate_input(X)
        self._gram = self.kernel.gram_matrix(X)
        return self._fit_best_of_restarts(X, self.n_init)

    def _initialize(self, X: np.ndarray) -> dict[str, object]:
        """Initialize weight vectors via k-means++ in feature space."""
        n = len(X)
        init_indices = _kmeanspp_feature_space(self._gram, self.n_clusters, self._rng)

        center_weights = np.zeros((self.n_clusters, n))
        for k, idx in enumerate(init_indices):
            center_weights[k, idx] = 1

        return {
            "center_weights": center_weights,
            "assignments": self._assign_kernel(self._gram, center_weights),
            "prev_assignments": None,
        }

    def _assign_kernel(self, gram: np.ndarray, center_weights: np.ndarray) -> np.ndarray:
        """Assign points using squared distance in feature space.

        Computes :math:`\\|\\varphi(X_i) - c_k\\|^2 = K_{ii} - 2(Kw_k)_i + w_k^\\top K w_k`
        for each point and cluster, then assigns to the nearest.
        """
        diag_K = np.diag(gram)
        costs = np.empty((gram.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            Kw = gram @ center_weights[k]
            costs[:, k] = diag_K - 2 * Kw + float(center_weights[k] @ Kw)
        return np.asarray(np.argmin(costs, axis=1))

    def _fit_iteration(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """One iteration: assign, handle empties, update weights, re-assign."""
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)

        state["prev_assignments"] = assignments.copy()
        state["assignments"] = self._assign_kernel(self._gram, center_weights)
        assignments = state["assignments"]
        assert isinstance(assignments, np.ndarray)

        n = self._gram.shape[0]
        if any(np.sum(assignments == k) == 0 for k in range(self.n_clusters)):
            cw = center_weights.copy()
            for k in range(self.n_clusters):
                if np.sum(assignments == k) == 0:
                    cw[k] = 0
                    cw[k, self._rng.choice(n)] = 1
            state["center_weights"] = cw
            state["assignments"] = self._assign_kernel(self._gram, cw)
            assignments = state["assignments"]
            assert isinstance(assignments, np.ndarray)

        new_cw = np.zeros((self.n_clusters, n))
        for k in range(self.n_clusters):
            mask = assignments == k
            n_k = int(np.sum(mask))
            if n_k > 0:
                new_cw[k, mask] = 1 / n_k

        state["center_weights"] = new_cw
        state["assignments"] = self._assign_kernel(self._gram, new_cw)
        return state

    def _additional_convergence_check(self, state: dict[str, object]) -> bool:
        """Check partition stability: stop if assignments unchanged."""
        prev = state.get("prev_assignments")
        curr = state.get("assignments")
        if isinstance(prev, np.ndarray) and isinstance(curr, np.ndarray):
            return bool(np.array_equal(prev, curr))
        return False

    def _compute_objective(self, X: np.ndarray, state: dict[str, object]) -> float:
        """Compute within-cluster sum of squared feature-space distances.

        .. math::
            J = \\frac{1}{n} \\sum_{k=1}^K \\sum_{i \\in C_k}
            \\bigl(K_{ii} - 2(Kw_k)_i + w_k^\\top K w_k\\bigr)
        """
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)

        diag_K = np.diag(self._gram)
        objective: float = 0
        for k in range(self.n_clusters):
            mask = assignments == k
            if not np.any(mask):
                continue
            Kw = self._gram @ center_weights[k]
            objective += float(np.sum(diag_K[mask] - 2 * Kw[mask] + center_weights[k] @ Kw))
        return objective / len(X)

    def _extract_result(
        self, state: dict[str, object], objective: float, n_iterations: int, converged: bool
    ) -> ClusterResult:
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        return ClusterResult(
            assignments=assignments,
            centers=center_weights,
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
            metadata={"center_weights": center_weights},
        )


class KernelGeometricExpectileClustering(BaseClusterer):
    """Kernelised geometric expectile clustering (Algorithm 2).

    In the feature space :math:`\\mathcal{H}` induced by a kernel :math:`k`, centroids and index
    vectors lie in the span of the data (Proposition 3.1):

    .. math::
        c_k = \\sum_{j=1}^n \\beta_{kj}\\,\\varphi(X_j),
        \\quad
        u_k = \\sum_{j=1}^n \\alpha_{kj}\\,\\varphi(X_j).

    All norms and inner products reduce to operations on the Gram matrix
    :math:`K_{ij} = k(X_i, X_j)`:

    - Squared residual norm:
      :math:`d_{ik}^2 = K_{ii} - 2(K\\beta_k)_i + \\beta_k^\\top K \\beta_k`
    - Index-residual inner product:
      :math:`p_{ik} = (K\\alpha_k)_i - \\alpha_k^\\top K \\beta_k`

    Setting :math:`r = 0` recovers kernel K-means.

    Attributes:
        n_clusters: Number of clusters.
        kernel: Kernel function.
        index_radius: Radius constraint :math:`r \\in [0, 1)`.
        n_init: Number of random restarts (best result is kept).
        max_iter: Maximum number of outer iterations.
        tol: Convergence tolerance on objective change.
        center_lr: Learning rate for gradient descent on centroid weights.
        center_steps: Number of gradient steps per centroid update.
        random_state: Random seed.

    Example:
        >>> import numpy as np
        >>> from geomexp.kernels import KernelGeometricExpectileClustering, RBFKernel
        >>> X = np.random.randn(100, 2)
        >>> model = KernelGeometricExpectileClustering(
        ...     n_clusters=3, kernel=RBFKernel(gamma=0.5), index_radius=0.5
        ... )
        >>> result = model.fit(X)
        >>> print(result.metadata["index_weights"].shape)
        (3, 100)
    """

    def __init__(
        self,
        n_clusters: int,
        kernel: Kernel,
        index_radius: float = 0.5,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-7,
        center_lr: float = 0.1,
        center_steps: int = 50,
        random_state: int | None = None,
    ) -> None:
        """Initialize kernelised geometric expectile clusterer.

        Args:
            n_clusters: Number of clusters.
            kernel: Kernel function instance.
            index_radius: Radius constraint :math:`r \\in [0, 1)` on index vectors.
            n_init: Number of independent restarts.
            max_iter: Maximum number of iterations per restart.
            tol: Convergence tolerance on objective change.
            center_lr: Learning rate for gradient descent on centroid weight vectors.
            center_steps: Number of gradient steps per centroid update.
            random_state: Random seed for reproducibility.

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
        self.kernel = kernel
        self.index_radius = index_radius
        self.n_init = n_init
        self.center_lr = center_lr
        self.center_steps = center_steps

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Fit kernelised geometric expectile clustering with multiple random restarts.

        The Gram matrix is computed once and reused across all ``n_init`` restarts.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            ClusterResult from the best restart.
        """
        X = self._validate_input(X)
        self._gram = self.kernel.gram_matrix(X)
        self._diag_K = np.diag(self._gram)
        return self._fit_best_of_restarts(X, self.n_init)

    def _initialize(self, X: np.ndarray) -> dict[str, object]:
        """Initialize centroid weights via k-means++ in feature space, index weights to zero."""
        n = len(X)
        init_indices = _kmeanspp_feature_space(self._gram, self.n_clusters, self._rng)

        center_weights = np.zeros((self.n_clusters, n))
        for k, idx in enumerate(init_indices):
            center_weights[k, idx] = 1

        return {
            "center_weights": center_weights,
            "index_weights": np.zeros((self.n_clusters, n)),
            "assignments": self._assign_by_distance(center_weights),
            "prev_assignments": None,
        }

    def _residual_quantities(
        self, center_weights_k: np.ndarray, index_weights_k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute residual squared norms, norms, and index inner products for one cluster.

        Args:
            center_weights_k: Centroid weight vector :math:`\\beta_k` of shape ``(n,)``.
            index_weights_k: Index weight vector :math:`\\alpha_k` of shape ``(n,)``.

        Returns:
            Tuple ``(d_sq, d, p)`` each of shape ``(n,)`` where :math:`d_{ik}^2` is the squared
            residual norm, :math:`d_{ik}` the norm, and :math:`p_{ik}` the index-residual inner
            product.
        """
        Kb = self._gram @ center_weights_k
        bKb = float(center_weights_k @ Kb)
        d_sq = self._diag_K - 2 * Kb + bKb
        np.maximum(d_sq, 0, out=d_sq)

        Ka = self._gram @ index_weights_k
        return d_sq, np.sqrt(d_sq), Ka - float(index_weights_k @ Kb)

    def _assign_by_distance(self, center_weights: np.ndarray) -> np.ndarray:
        """Assign points by squared feature-space distance (used for initialisation)."""
        costs = np.empty((self._gram.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            Kb = self._gram @ center_weights[k]
            costs[:, k] = self._diag_K - 2 * Kb + float(center_weights[k] @ Kb)
        return np.asarray(np.argmin(costs, axis=1))

    def _assign_expectile(
        self, center_weights: np.ndarray, index_weights: np.ndarray
    ) -> np.ndarray:
        """Assign each point to the cluster minimising the expectile loss in feature space."""
        costs = np.empty((self._gram.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            d_sq, d, p = self._residual_quantities(center_weights[k], index_weights[k])
            costs[:, k] = 0.5 * (d_sq + d * p)
        return np.asarray(np.argmin(costs, axis=1))

    def _fit_iteration(self, X: np.ndarray, state: dict[str, object]) -> dict[str, object]:
        """One outer iteration: assign, handle empties, update centers, update indices."""
        assignments = state["assignments"]
        assert isinstance(assignments, np.ndarray)
        state["prev_assignments"] = assignments.copy()

        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)

        state["assignments"] = self._assign_expectile(center_weights, index_weights)
        state = self._handle_empty_clusters(state)
        state = self._update_centers(state)
        state = self._update_indices(state)

        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)
        state["assignments"] = self._assign_expectile(center_weights, index_weights)
        return state

    def _handle_empty_clusters(self, state: dict[str, object]) -> dict[str, object]:
        """Reinitialise empty clusters to random data points with zero index weights."""
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)

        if not any(np.sum(assignments == k) == 0 for k in range(self.n_clusters)):
            return state

        cw, iw = center_weights.copy(), index_weights.copy()
        for k in range(self.n_clusters):
            if np.sum(assignments == k) == 0:
                cw[k] = 0
                cw[k, self._rng.choice(self._gram.shape[0])] = 1
                iw[k] = 0

        state["center_weights"] = cw
        state["index_weights"] = iw
        state["assignments"] = self._assign_expectile(cw, iw)
        return state

    def _update_centers(self, state: dict[str, object]) -> dict[str, object]:
        """Update centroid weight vectors via gradient descent on the cluster-wise risk.

        For each cluster, takes ``center_steps`` gradient steps on :math:`\\beta_k` using the
        Hilbert gradient. With a single cluster member, the centroid is set directly.
        """
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)

        new_cw = center_weights.copy()
        for k in range(self.n_clusters):
            mask = assignments == k
            n_k = int(np.sum(mask))
            if n_k <= 1:
                if n_k == 1:
                    new_cw[k] = 0
                    new_cw[k, mask] = 1
                continue

            beta = new_cw[k].copy()
            for _ in range(self.center_steps):
                beta -= self.center_lr * self._center_gradient_weights(
                    beta, index_weights[k], mask, n_k
                )
            new_cw[k] = beta

        state["center_weights"] = new_cw
        return state

    def _center_gradient_weights(
        self, beta: np.ndarray, alpha: np.ndarray, mask: np.ndarray, n_k: int
    ) -> np.ndarray:
        """Compute the Hilbert gradient weight vector for centroid update.

        The Hilbert gradient of :math:`\\ell_{u_k}(\\varphi(X_i) - c_k)` w.r.t. :math:`c_k` has
        weight vector :math:`g_i = (\\beta - e_i)(1 + p_i / 2d_i) - d_i \\alpha / 2`. This
        method returns the mean over cluster members.
        """
        Kb = self._gram @ beta
        d_sq = self._diag_K - 2 * Kb + float(beta @ Kb)
        np.maximum(d_sq, 0, out=d_sq)
        d = np.sqrt(d_sq)

        Ka = self._gram @ alpha
        p = Ka - float(alpha @ Kb)

        active = mask & (d > 1e-12)
        if not np.any(active):
            return np.zeros_like(beta)

        w = np.zeros(len(d))
        w[active] = 1 + 0.5 * p[active] / d[active]

        return (float(np.sum(w)) * beta - w - 0.5 * float(np.sum(d[active])) * alpha) / n_k

    def _update_indices(self, state: dict[str, object]) -> dict[str, object]:
        """Update index weight vectors via best-response (equations 3.2--3.3).

        For each cluster, computes :math:`\\delta_k` and normalises to obtain
        :math:`\\alpha_k^* = -r \\, \\delta_k / \\sqrt{\\delta_k^\\top K \\delta_k}`.
        """
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)

        new_iw = index_weights.copy()
        if self.index_radius == 0:
            state["index_weights"] = new_iw
            return state

        for k in range(self.n_clusters):
            mask = assignments == k
            if not np.any(mask):
                continue

            _, d, _ = self._residual_quantities(center_weights[k], index_weights[k])
            delta = np.zeros(self._gram.shape[0])
            delta[mask] = d[mask]
            delta -= float(np.sum(d[mask])) * center_weights[k]

            norm_delta = np.sqrt(max(float(delta @ (self._gram @ delta)), 0))
            if norm_delta > 1e-12:
                new_iw[k] = -self.index_radius * delta / norm_delta

        state["index_weights"] = new_iw
        return state

    def _compute_objective(self, X: np.ndarray, state: dict[str, object]) -> float:
        """Compute empirical risk in feature space.

        .. math::
            \\hat{\\mathcal{R}} = \\frac{1}{n} \\sum_{k=1}^K
            \\sum_{i \\in C_k} \\tfrac{1}{2}\\bigl(d_{ik}^2 + d_{ik}\\,p_{ik}\\bigr)
        """
        assignments = state["assignments"]
        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)

        objective: float = 0
        for k in range(self.n_clusters):
            mask = assignments == k
            if not np.any(mask):
                continue
            d_sq, d, p = self._residual_quantities(center_weights[k], index_weights[k])
            objective += float(np.sum(0.5 * (d_sq[mask] + d[mask] * p[mask])))
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
        center_weights = state["center_weights"]
        index_weights = state["index_weights"]
        assert isinstance(assignments, np.ndarray)
        assert isinstance(center_weights, np.ndarray)
        assert isinstance(index_weights, np.ndarray)
        return ClusterResult(
            assignments=assignments,
            centers=center_weights,
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
            metadata={"center_weights": center_weights, "index_weights": index_weights},
        )
