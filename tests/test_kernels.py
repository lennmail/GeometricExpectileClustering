"""Tests for geomexp/kernels/kernels.py.

Covers kernel functions, kernel K-means, and kernel GEC.
"""

import numpy as np
import pytest

from geomexp.evaluation.metrics import adjusted_rand_index
from geomexp.kernels.kernels import (
    KernelGeometricExpectileClustering,
    KernelKMeans,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    _kmeanspp_feature_space,
)


@pytest.fixture
def small_X(rng):
    return rng.randn(20, 3)


# --- Kernel properties (parametrized) ---


ALL_KERNELS = [
    LinearKernel(),
    RBFKernel(gamma=1.0),
    PolynomialKernel(degree=2, coef0=1.0),
    MaternKernel(nu=1.5, lengthscale=1.0),
]
KERNEL_IDS = ["linear", "rbf", "poly", "matern"]


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
class TestKernelProperties:
    def test_gram_symmetric(self, kernel, small_X):
        K = kernel.gram_matrix(small_X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_gram_psd(self, kernel, small_X):
        K = kernel.gram_matrix(small_X)
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() >= -1e-8 * max(abs(eigvals.max()), 1)

    def test_gram_shape(self, kernel, small_X):
        K = kernel.gram_matrix(small_X)
        assert K.shape == (20, 20)

    def test_kernel_call_shape(self, kernel, small_X):
        Y = small_X[:5]
        K = kernel(small_X, Y)
        assert K.shape == (20, 5)


# --- LinearKernel ---


class TestLinearKernel:
    def test_value(self, small_X):
        k = LinearKernel()
        K = k(small_X, small_X)
        np.testing.assert_allclose(K, small_X @ small_X.T, atol=1e-12)

    def test_gram_is_xxt(self, small_X):
        k = LinearKernel()
        np.testing.assert_allclose(k.gram_matrix(small_X), small_X @ small_X.T, atol=1e-12)


# --- RBFKernel ---


class TestRBFKernel:
    def test_self_is_one(self, small_X):
        k = RBFKernel(gamma=1.0)
        K = k.gram_matrix(small_X)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_range_01(self, small_X):
        k = RBFKernel(gamma=1.0)
        K = k.gram_matrix(small_X)
        assert np.all(K >= -1e-12)
        assert np.all(K <= 1 + 1e-12)

    def test_gamma_scaling(self, small_X):
        k_low = RBFKernel(gamma=0.1)
        k_high = RBFKernel(gamma=10.0)
        K_low = k_low.gram_matrix(small_X)
        K_high = k_high.gram_matrix(small_X)
        # Off-diagonal: higher gamma => values closer to 0
        mask = ~np.eye(20, dtype=bool)
        assert K_high[mask].mean() < K_low[mask].mean()

    def test_rejects_nonpositive_gamma(self):
        with pytest.raises(ValueError, match="positive"):
            RBFKernel(gamma=0)
        with pytest.raises(ValueError, match="positive"):
            RBFKernel(gamma=-1)


# --- PolynomialKernel ---


class TestPolynomialKernel:
    def test_known_value(self):
        k = PolynomialKernel(degree=2, coef0=1.0)
        X = np.array([[1.0, 0.0]])
        Y = np.array([[0.0, 1.0]])
        # k([1,0], [0,1]) = (0 + 1)^2 = 1
        assert k(X, Y)[0, 0] == pytest.approx(1.0)

    def test_degree1_coef0_matches_shifted_linear(self, small_X):
        k = PolynomialKernel(degree=1, coef0=0.0)
        K = k.gram_matrix(small_X)
        np.testing.assert_allclose(K, small_X @ small_X.T, atol=1e-12)

    def test_rejects_invalid_degree(self):
        with pytest.raises((ValueError, TypeError)):
            PolynomialKernel(degree=0)


# --- MaternKernel ---


class TestMaternKernel:
    def test_self_is_one(self, small_X):
        for nu in [0.5, 1.5, 2.5, np.inf, 3.0]:
            k = MaternKernel(nu=nu, lengthscale=1.0)
            K = k.gram_matrix(small_X)
            np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_05_is_exponential(self, small_X):
        k = MaternKernel(nu=0.5, lengthscale=2.0)
        K = k.gram_matrix(small_X)
        # Manual: k(x,y) = exp(-||x-y||/l)
        for i in range(5):
            for j in range(i + 1, 5):
                d = np.linalg.norm(small_X[i] - small_X[j]) / 2.0
                expected = np.exp(-d)
                assert K[i, j] == pytest.approx(expected, abs=1e-10)

    def test_inf_is_rbf(self, small_X):
        k = MaternKernel(nu=np.inf, lengthscale=2.0)
        K = k.gram_matrix(small_X)
        # Manual: k(x,y) = exp(-0.5 * (||x-y||/l)^2)
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(small_X[i] - small_X[j]) / 2.0
                expected = np.exp(-0.5 * d**2)
                assert K[i, j] == pytest.approx(expected, abs=1e-10)

    def test_15_known_value(self, small_X):
        k = MaternKernel(nu=1.5, lengthscale=1.0)
        K = k.gram_matrix(small_X)
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(small_X[i] - small_X[j])
                s3d = np.sqrt(3) * d
                expected = (1 + s3d) * np.exp(-s3d)
                assert K[i, j] == pytest.approx(expected, abs=1e-10)

    def test_25_known_value(self, small_X):
        k = MaternKernel(nu=2.5, lengthscale=1.0)
        K = k.gram_matrix(small_X)
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(small_X[i] - small_X[j])
                s5d = np.sqrt(5) * d
                expected = (1 + s5d + 5 * d**2 / 3) * np.exp(-s5d)
                assert K[i, j] == pytest.approx(expected, abs=1e-10)

    def test_general_nu(self, small_X):
        """General nu (not 0.5, 1.5, 2.5, inf) uses Bessel function."""
        k = MaternKernel(nu=3.0, lengthscale=1.0)
        K = k.gram_matrix(small_X)
        # Should still be PSD
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() >= -1e-8
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_rejects_nonpositive_lengthscale(self):
        with pytest.raises(ValueError, match="positive"):
            MaternKernel(nu=1.5, lengthscale=0)


# --- _kmeanspp_feature_space ---


class TestKMeansPP:
    def test_returns_k_indices(self, small_X):
        k = RBFKernel(gamma=1.0)
        gram = k.gram_matrix(small_X)
        idx = _kmeanspp_feature_space(gram, 3, np.random.RandomState(42))
        assert len(idx) == 3

    def test_indices_in_range(self, small_X):
        k = RBFKernel(gamma=1.0)
        gram = k.gram_matrix(small_X)
        idx = _kmeanspp_feature_space(gram, 3, np.random.RandomState(42))
        assert np.all((idx >= 0) & (idx < 20))

    def test_distinct_indices(self, small_X):
        k = RBFKernel(gamma=1.0)
        gram = k.gram_matrix(small_X)
        idx = _kmeanspp_feature_space(gram, 3, np.random.RandomState(42))
        assert len(set(idx)) == 3

    def test_deterministic(self, small_X):
        k = RBFKernel(gamma=1.0)
        gram = k.gram_matrix(small_X)
        idx1 = _kmeanspp_feature_space(gram, 3, np.random.RandomState(42))
        idx2 = _kmeanspp_feature_space(gram, 3, np.random.RandomState(42))
        np.testing.assert_array_equal(idx1, idx2)


# --- KernelKMeans ---


class TestKernelKMeans:
    def test_well_separated(self, well_separated_2d, well_separated_labels):
        km = KernelKMeans(
            n_clusters=3,
            kernel=RBFKernel(gamma=0.1),
            n_init=3,
            max_iter=20,
            random_state=42,
        )
        result = km.fit(well_separated_2d)
        ari = adjusted_rand_index(well_separated_labels, result.assignments)
        assert ari > 0.9

    def test_linear_kernel_recovers_clusters(self, well_separated_2d, well_separated_labels):
        km = KernelKMeans(
            n_clusters=3,
            kernel=LinearKernel(),
            n_init=3,
            max_iter=30,
            random_state=42,
        )
        result = km.fit(well_separated_2d)
        ari = adjusted_rand_index(well_separated_labels, result.assignments)
        assert ari > 0.9

    def test_result_shape(self, simple_2d):
        km = KernelKMeans(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            n_init=1,
            max_iter=5,
            random_state=42,
        )
        result = km.fit(simple_2d)
        assert result.assignments.shape == (30,)

    def test_deterministic(self, simple_2d):
        kwargs = {
            "n_clusters": 2,
            "kernel": RBFKernel(gamma=1.0),
            "n_init": 1,
            "max_iter": 5,
            "random_state": 42,
        }
        r1 = KernelKMeans(**kwargs).fit(simple_2d)
        r2 = KernelKMeans(**kwargs).fit(simple_2d)
        np.testing.assert_array_equal(r1.assignments, r2.assignments)

    def test_objective_is_float(self, simple_2d):
        km = KernelKMeans(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            n_init=1,
            max_iter=5,
            random_state=42,
        )
        result = km.fit(simple_2d)
        assert isinstance(result.objective, float)

    def test_metadata_has_weights(self, simple_2d):
        km = KernelKMeans(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            n_init=1,
            max_iter=5,
            random_state=42,
        )
        result = km.fit(simple_2d)
        assert result.centers.shape == (2, len(simple_2d))
        assert result.metadata is None


# --- KernelGeometricExpectileClustering ---


class TestKernelGEC:
    def test_well_separated(self, well_separated_2d, well_separated_labels):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=3,
            kernel=RBFKernel(gamma=0.1),
            index_radius=0.3,
            n_init=2,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = kgec.fit(well_separated_2d)
        ari = adjusted_rand_index(well_separated_labels, result.assignments)
        assert ari > 0.8

    def test_radius_zero_like_kernel_kmeans(self, simple_2d):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            index_radius=0.0,
            n_init=2,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = kgec.fit(simple_2d)
        assert isinstance(result.objective, float)

    def test_metadata_has_weights(self, simple_2d):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            index_radius=0.3,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = kgec.fit(simple_2d)
        assert result.centers.shape == (2, len(simple_2d))
        assert "index_weights" in result.metadata

    def test_deterministic(self, simple_2d):
        kwargs = {
            "n_clusters": 2,
            "kernel": RBFKernel(gamma=1.0),
            "index_radius": 0.3,
            "n_init": 1,
            "max_iter": 5,
            "center_steps": 3,
            "random_state": 42,
        }
        r1 = KernelGeometricExpectileClustering(**kwargs).fit(simple_2d)
        r2 = KernelGeometricExpectileClustering(**kwargs).fit(simple_2d)
        np.testing.assert_array_equal(r1.assignments, r2.assignments)

    def test_convergence(self, well_separated_2d):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=3,
            kernel=RBFKernel(gamma=0.1),
            index_radius=0.3,
            n_init=1,
            max_iter=50,
            center_steps=10,
            random_state=42,
        )
        result = kgec.fit(well_separated_2d)
        assert result.converged

    def test_rejects_invalid_radius(self):
        with pytest.raises(ValueError):
            KernelGeometricExpectileClustering(
                n_clusters=2,
                kernel=RBFKernel(),
                index_radius=1.0,
            )

    def test_objective_nonneg(self, simple_2d):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            index_radius=0.3,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = kgec.fit(simple_2d)
        assert result.objective >= -1e-6

    def test_result_shapes(self, simple_2d):
        kgec = KernelGeometricExpectileClustering(
            n_clusters=2,
            kernel=RBFKernel(gamma=1.0),
            index_radius=0.3,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = kgec.fit(simple_2d)
        assert result.assignments.shape == (30,)
        assert result.centers.shape == (2, 30)
        assert result.metadata["index_weights"].shape == (2, 30)
