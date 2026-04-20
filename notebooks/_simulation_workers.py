"""Worker functions for ``simulation_study.ipynb`` parallelisation.

Extracted into a separate module so that :class:`~concurrent.futures.ProcessPoolExecutor` can pickle
them.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from geomexp import (
    GeometricExpectileClustering,
    KMeans,
    WeightedEuclideanGeometry,
)
from geomexp.evaluation import (
    adjusted_rand_index,
    misclassification_error,
    variation_of_information,
)

K: int = 3
N_RESTARTS: int = 20


def init_globals(k: int, n_restarts: int) -> None:
    """Overwrite module-level constants from the notebook process initialiser."""
    global K, N_RESTARTS
    K = k
    N_RESTARTS = n_restarts


def make_basis(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the four-term Fourier basis on an equispaced grid over :math:`[0, 1]`.

    Returns ``(t, phi)`` where *t* has *m* points and *phi* has columns
    :math:`\\sin(2\\pi t),\\; \\cos(2\\pi t),\\; \\sin(4\\pi t),\\; \\cos(4\\pi t)`.
    """
    t = np.linspace(0, 1, m)
    phi = np.column_stack(
        [
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t),
            np.cos(4 * np.pi * t),
        ]
    )
    return t, phi


def cluster_centers(n_clusters: int, p: int, R: float) -> tuple[np.ndarray, list[float]]:
    """Arrange *n_clusters* centres in :math:`\\mathbb{R}^p` at equally-spaced angles.

    Coordinates 1--2 lie on a circle of radius *R*; coordinates :math:`j \\ge 3` have amplitude
    :math:`R/2` with a phase shift of :math:`\\pi j / p`.
    """
    angles = [2 * np.pi * k / n_clusters for k in range(n_clusters)]
    centers = np.zeros((n_clusters, p))
    for k in range(n_clusters):
        centers[k, 0] = R * np.cos(angles[k])
        centers[k, 1] = R * np.sin(angles[k])
        for j in range(2, p):
            centers[k, j] = (R / 2) * np.cos(angles[k] + np.pi * j / p)
    return centers, angles


def generate_scenario1(
    seed: int,
    K: int = 3,
    n_per_cluster: int = 150,
    m: int = 100,
    p: int = 4,
    R: float = 2,
    sigma_eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Scenario 1: symmetric Gaussian clusters with rotation-aligned anisotropic covariance.

    Each cluster *k* draws coefficients from :math:`\\mathcal{N}(\\mu_k, \\Sigma_k)` where the
    leading 2x2 block of :math:`\\Sigma_k` is rotated by the cluster angle :math:`\\theta_k` with
    eigenvalues :math:`0.45^2` (radial) and :math:`0.12^2` (transverse).
    """
    rng = np.random.default_rng(seed)
    _, phi = make_basis(m)
    centers_c, angles = cluster_centers(K, p, R)

    covs = []
    for k in range(K):
        theta = angles[k]
        Q2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        D2 = np.diag([0.45**2, 0.12**2])
        Sigma = np.zeros((p, p))
        Sigma[:2, :2] = Q2 @ D2 @ Q2.T
        Sigma[2:, 2:] = np.diag([0.18**2, 0.12**2])
        covs.append(Sigma)

    coeffs_list, labels_list = [], []
    for k in range(K):
        ck = rng.multivariate_normal(centers_c[k], covs[k], size=n_per_cluster)
        coeffs_list.append(ck)
        labels_list.append(np.full(n_per_cluster, k))

    coeffs_all = np.vstack(coeffs_list)
    y_true = np.concatenate(labels_list)
    X = coeffs_all @ phi.T + rng.normal(0, sigma_eps, (len(y_true), m))
    return X, y_true


def generate_scenario2(
    seed: int,
    K: int = 3,
    n_per_cluster: int = 150,
    m: int = 100,
    p: int = 4,
    R: float = 2,
    sigma_c: float = 0.1,
    tau: float = 1,
    sigma_eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Scenario 2: isotropic Gaussian spread plus cyclic exponential tails in coefficient space.

    Each coefficient vector is :math:`c_i = \\mu_k + \\sigma_c \\eta_i + Z_i \\hat{d}_k` where
    :math:`\\eta_i \\sim \\mathcal{N}(0, I_p)` and :math:`Z_i \\sim \\mathrm{Exp}(\\tau)`.  The
    tail direction :math:`\\hat{d}_k` points from centre *k* toward centre *k + 1* cyclically.
    """
    rng = np.random.default_rng(seed)
    _, phi = make_basis(m)
    centers_c, _ = cluster_centers(K, p, R)

    tail_dirs = np.zeros((K, p))
    for k in range(K):
        d = centers_c[(k + 1) % K] - centers_c[k]
        tail_dirs[k] = d / np.linalg.norm(d)

    coeffs_list, labels_list = [], []
    for k in range(K):
        sym = sigma_c * rng.standard_normal((n_per_cluster, p))
        Z = rng.exponential(tau, (n_per_cluster, 1))
        ck = centers_c[k] + sym + Z * tail_dirs[k]
        coeffs_list.append(ck)
        labels_list.append(np.full(n_per_cluster, k))

    coeffs_all = np.vstack(coeffs_list)
    y_true = np.concatenate(labels_list)
    X = coeffs_all @ phi.T + rng.normal(0, sigma_eps, (len(y_true), m))
    return X, y_true


def generate_scenario3(
    seed: int,
    K: int = 3,
    n_per_cluster: int = 150,
    m: int = 100,
    p: int = 4,
    R: float = 2,
    sigma_c: float = 0.1,
    sigma_eps: float = 0.1,
    tau: float = 1,
    sigma_a: float = 0.15,
    ramp_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Scenario 3: symmetric Fourier baseline plus cluster-specific ramp perturbations.

    Each curve is :math:`X_i(t) = c_i \\Phi(t) + \\alpha A_i \\tilde{\\rho}_k(t) + \\varepsilon_i`
    where the amplitude :math:`A_i = \\sigma_a \\xi_i + Z_i` combines a Gaussian and an exponential
    component, and :math:`\\tilde{\\rho}_k` is a unit-:math:`L^2`-norm ramp with onset :math:`s_k`.
    """
    rng = np.random.default_rng(seed)
    t, phi = make_basis(m)
    centers_c, _ = cluster_centers(K, p, R)

    s_k = np.array([0.1, 0.4, 0.7])
    ramp_fns: list[np.ndarray] = []
    for s in s_k:
        rho = np.maximum(t - s, 0)
        norm_rho = np.sqrt(np.mean(rho**2))
        if norm_rho > 0:
            rho = rho / norm_rho
        ramp_fns.append(rho)
    ramps = np.array(ramp_fns)

    curves_list, labels_list = [], []
    for k in range(K):
        sym = sigma_c * rng.standard_normal((n_per_cluster, p))
        ck = centers_c[k] + sym
        labels_list.append(np.full(n_per_cluster, k))
        baseline = ck @ phi.T
        A = sigma_a * rng.standard_normal(n_per_cluster) + rng.exponential(tau, n_per_cluster)
        Xk = baseline + ramp_scale * A[:, None] * ramps[k][None, :]
        Xk += rng.normal(0, sigma_eps, (n_per_cluster, m))
        curves_list.append(Xk)

    y_true = np.concatenate(labels_list)
    X = np.vstack(curves_list)
    return X, y_true


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_restarts: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Fit K-means with *n_restarts* random restarts, returning ``(labels, objective, elapsed)``."""
    t0 = time.perf_counter()
    best = None
    for trial in range(n_restarts):
        km = KMeans(n_clusters=n_clusters, max_iter=100, tol=1e-4, random_state=seed + trial)
        res = km.fit(X)
        if best is None or res.objective < best.objective:
            best = res
    assert best is not None
    elapsed = time.perf_counter() - t0
    return best.assignments, best.objective, elapsed


def fit_gec(
    X: np.ndarray,
    n_clusters: int,
    r: float,
    n_restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit GEC with *r* and *n_restarts* restarts, returning ``(labels, centres, obj, elapsed)``."""
    dim = X.shape[1]
    geom = WeightedEuclideanGeometry(np.ones(dim) / dim)
    gec = GeometricExpectileClustering(
        n_clusters=n_clusters,
        index_radius=r,
        geometry=geom,
        n_init=n_restarts,
        max_iter=100,
        tol=1e-7,
        center_lr=0.1,
        center_steps=50,
        random_state=seed,
    )
    t0 = time.perf_counter()
    res = gec.fit(X)
    elapsed = time.perf_counter() - t0
    return res.assignments, res.centers, res.objective, elapsed


def _asymmetry_statistic(
    X: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
    geom: WeightedEuclideanGeometry,
) -> float:
    """Projected skewness along the best-response (skew) direction.

    For each cluster *k* with residuals :math:`t_i = x_i - c_k`:

    1. Compute the skew vector :math:`s_k = \\frac{1}{n_k} \\sum_i \\|t_i\\|_H \\, t_i`.
    2. Project residuals onto the skew direction :math:`d_k = s_k / \\|s_k\\|_H`.
    3. Compute the univariate skewness :math:`|\\gamma_k|` of the projected values.

    Returns the population-weighted aggregate :math:`\\mathcal{A} = \\sum_k (n_k / n)|\\gamma_k|`.
    This avoids the curse-of-dimensionality problem of omnidirectional statistics by reducing each
    cluster to a single informative direction first.
    """
    n_clusters = len(centers)
    n = len(X)
    A = 0
    for k in range(n_clusters):
        mask = assignments == k
        n_k = int(np.sum(mask))
        if n_k < 5:
            continue
        residuals = X[mask] - centers[k]
        norms = geom.norm(residuals)

        s_k = np.mean(norms[:, np.newaxis] * residuals, axis=0)
        s_norm = float(geom.norm(s_k))
        if s_norm < 1e-12:
            continue
        d_k = s_k / s_norm

        z = geom.inner(residuals, np.broadcast_to(d_k, residuals.shape))
        z_c = z - np.mean(z)
        m2 = float(np.mean(z_c**2))
        m3 = float(np.mean(z_c**3))
        if m2 < 1e-30:
            continue
        gamma_k = abs(m3 / m2**1.5)
        A += (n_k / n) * gamma_k
    return A


def tune_r(
    X: np.ndarray,
    n_clusters: int,
    n_restarts: int,
    seed: int,
    r_max: float = 0.99,
    lam: float = 2.5,
    n_refine: int = 1,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Select *r* via iterative projected-skewness refinement.

    1. Fit K-means (:math:`r = 0`) to obtain initial centres and assignments.
    2. Compute the projected skewness statistic :math:`\\mathcal{A}` from residuals.
    3. Map via exponential saturation: :math:`r = r_{\\max}(1 - e^{-\\lambda \\mathcal{A}})`.
    4. Fit GEC with this *r*, recompute :math:`\\mathcal{A}`, update *r*.  Repeat *n_refine* times.

    Returns ``(r, assignments, centres, elapsed)`` so the caller can reuse the last GEC fit.
    """
    dim = X.shape[1]
    geom = WeightedEuclideanGeometry(np.ones(dim) / dim)

    best = None
    for trial in range(n_restarts):
        km = KMeans(n_clusters=n_clusters, max_iter=100, tol=1e-4, random_state=seed + trial)
        res = km.fit(X)
        if best is None or res.objective < best.objective:
            best = res
    assert best is not None

    centers = best.centers
    assignments = best.assignments

    r: float = 0
    last_gec_time: float = 0
    gec_labels = assignments
    gec_centers = centers
    for _ in range(1 + n_refine):
        A = _asymmetry_statistic(X, centers, assignments, geom)
        r = float(r_max * (1 - np.exp(-lam * A)))
        if r < 1e-6:
            break
        gec_labels, gec_centers, _, last_gec_time = fit_gec(
            X,
            n_clusters,
            r,
            n_restarts,
            seed,
        )
        centers = gec_centers
        assignments = gec_labels

    return r, gec_labels, gec_centers, last_gec_time


def compute_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
    runtime: float,
) -> dict[str, float]:
    """Compute ARI, VI, misclassification error and store the wall-clock runtime."""
    return {
        "ari": adjusted_rand_index(y_true, labels),
        "vi": variation_of_information(y_true, labels),
        "miscl": misclassification_error(y_true, labels),
        "runtime": runtime,
    }


def run_one_replication(
    args: tuple[int, Any, dict[str, Any]],
) -> dict[str, Any]:
    """Run one MC replication: fit K-means and GEC (with auto-tuned *r*) on generated data."""
    mc_seed, generate_fn, gen_kwargs = args
    X, y_true = generate_fn(seed=mc_seed, **gen_kwargs)

    km_labels, _, km_time = fit_kmeans(X, K, N_RESTARTS, mc_seed)
    best_r, gec_labels, _, gec_time = tune_r(X, K, N_RESTARTS, mc_seed)

    return {
        "kmeans": compute_metrics(y_true, km_labels, km_time),
        "gec": compute_metrics(y_true, gec_labels, gec_time),
        "selected_r": float(best_r),
    }


def sweep_worker(
    args: tuple[int, Any, dict[str, Any], float],
) -> tuple[float, dict[str, Any]]:
    """Unpack ``(seed, gen_fn, kwargs, param_value)`` and run one replication."""
    seed, gen_fn, kwargs, val = args
    return val, run_one_replication((seed, gen_fn, kwargs))


def r_stability_worker(
    args: tuple[int, Any, dict[str, Any], float, np.ndarray],
) -> tuple[int, float, float, list[float]]:
    """Worker for r-stability heatmap study.

    Args: (seed, gen_fn, gen_kwargs, tau, r_grid)
    Returns: (seed, tau, selected_r, miscl_errors)

    For each (seed, tau):
    1. Generate data with the given tau
    2. Run adaptive tuning to get r_selected (once)
    3. For each r in r_grid, fit GEC and record miscl error
    """
    seed, gen_fn, gen_kwargs, tau, r_grid = args
    kwargs_with_tau = {**gen_kwargs, "tau": tau}
    X, y_true = gen_fn(seed=seed, **kwargs_with_tau)

    selected_r, _, _, _ = tune_r(X, K, N_RESTARTS, seed)

    miscl_errors = []
    for r_value in r_grid:
        y_pred, _, _, _ = fit_gec(X, K, r_value, N_RESTARTS, seed)
        miscl_errors.append(misclassification_error(y_true, y_pred))

    return seed, tau, selected_r, miscl_errors


def selected_r_lambda_worker(
    args: tuple[int, Any, dict[str, Any], float, float],
) -> tuple[int, float, float, float]:
    """Worker for lightweight selected-*r* lambda overlays.

    Args: (seed, gen_fn, gen_kwargs, tau, lam)
    Returns: (seed, tau, lam, selected_r)

    This is intentionally cheaper than ``r_stability_worker``:
    it runs adaptive tuning once and does not sweep over an ``r_grid``.
    """
    seed, gen_fn, gen_kwargs, tau, lam = args
    kwargs_with_tau = {**gen_kwargs, "tau": tau}
    X, _ = gen_fn(seed=seed, **kwargs_with_tau)

    selected_r, _, _, _ = tune_r(X, K, N_RESTARTS, seed, lam=lam)
    return seed, tau, lam, selected_r
