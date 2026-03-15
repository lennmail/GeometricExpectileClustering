"""Tests for geomexp/clustering/geometry.py.

Covers Hilbert space axioms (inner product, norm) for all three concrete geometries
using parametrized property-based tests.
"""

import numpy as np
import pytest

from geomexp.clustering.geometry import (
    EuclideanGeometry,
    GramGeometry,
    WeightedEuclideanGeometry,
)

# --- Inner product properties (parametrized over geometries) ---


class TestInnerProductProperties:
    """Test inner product axioms: bilinearity, symmetry, positive-definiteness."""

    def test_inner_self_nonneg(self, any_geometry, rng):
        a = rng.randn(2)
        assert any_geometry.inner(a, a) >= -1e-14

    def test_inner_zero_vector(self, any_geometry):
        z = np.zeros(2)
        a = np.array([1.0, 2.0])
        assert abs(any_geometry.inner(z, a)) < 1e-14
        assert abs(any_geometry.inner(a, z)) < 1e-14

    def test_inner_symmetry(self, any_geometry, rng):
        a, b = rng.randn(2), rng.randn(2)
        np.testing.assert_allclose(any_geometry.inner(a, b), any_geometry.inner(b, a), atol=1e-12)

    def test_inner_linearity_scalar(self, any_geometry, rng):
        a, b = rng.randn(2), rng.randn(2)
        alpha = 2.7
        np.testing.assert_allclose(
            any_geometry.inner(alpha * a, b), alpha * any_geometry.inner(a, b), atol=1e-12
        )

    def test_inner_linearity_addition(self, any_geometry, rng):
        a, b, c = rng.randn(2), rng.randn(2), rng.randn(2)
        np.testing.assert_allclose(
            any_geometry.inner(a + b, c),
            any_geometry.inner(a, c) + any_geometry.inner(b, c),
            atol=1e-12,
        )

    def test_cauchy_schwarz(self, any_geometry, rng):
        """Cauchy-Schwarz: |<a, b>| <= ||a|| * ||b||."""
        a, b = rng.randn(2), rng.randn(2)
        lhs = abs(any_geometry.inner(a, b))
        rhs = any_geometry.norm(a) * any_geometry.norm(b)
        assert lhs <= rhs + 1e-12


# --- Norm properties (parametrized over geometries) ---


class TestNormProperties:
    """Test norm axioms: non-negativity, homogeneity, triangle inequality."""

    def test_norm_nonneg(self, any_geometry, rng):
        a = rng.randn(2)
        assert any_geometry.norm(a) >= -1e-14

    def test_norm_zero(self, any_geometry):
        assert any_geometry.norm(np.zeros(2)) < 1e-14

    def test_norm_positive_definite(self, any_geometry, rng):
        a = rng.randn(2)
        # Random vector is almost surely nonzero
        assert any_geometry.norm(a) > 1e-10

    def test_norm_homogeneity(self, any_geometry, rng):
        a = rng.randn(2)
        alpha = -3.14
        np.testing.assert_allclose(
            any_geometry.norm(alpha * a), abs(alpha) * any_geometry.norm(a), atol=1e-12
        )

    def test_triangle_inequality(self, any_geometry, rng):
        a, b = rng.randn(2), rng.randn(2)
        assert any_geometry.norm(a + b) <= any_geometry.norm(a) + any_geometry.norm(b) + 1e-12

    def test_squared_norm_equals_inner(self, any_geometry, rng):
        a = rng.randn(2)
        expected = any_geometry.inner(a, a)
        np.testing.assert_allclose(any_geometry.squared_norm(a), expected, atol=1e-12)

    def test_squared_norm_equals_norm_sq(self, any_geometry, rng):
        a = rng.randn(2)
        expected = any_geometry.norm(a) ** 2
        np.testing.assert_allclose(any_geometry.squared_norm(a), expected, atol=1e-12)


# --- Euclidean-specific ---


class TestEuclideanGeometry:
    def test_inner_known_value(self):
        g = EuclideanGeometry()
        assert g.inner(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(11.0)

    def test_norm_known_value(self):
        g = EuclideanGeometry()
        assert g.norm(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_squared_norm_known_value(self):
        g = EuclideanGeometry()
        assert g.squared_norm(np.array([3.0, 4.0])) == pytest.approx(25.0)

    def test_batched(self):
        g = EuclideanGeometry()
        X = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
        norms = g.norm(X)
        np.testing.assert_allclose(norms, [1.0, 1.0, 5.0])

    def test_inner_batched(self):
        g = EuclideanGeometry()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = g.inner(X, Y)
        np.testing.assert_allclose(result, [17.0, 53.0])


# --- Weighted Euclidean-specific ---


class TestWeightedEuclideanGeometry:
    def test_inner_known_value(self):
        g = WeightedEuclideanGeometry(np.array([2.0, 3.0]))
        # inner([1,1], [1,1]) = 2*1*1 + 3*1*1 = 5
        assert g.inner(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(5.0)

    def test_norm_known_value(self):
        g = WeightedEuclideanGeometry(np.array([4.0, 1.0]))
        # norm([1,2]) = sqrt(4*1 + 1*4) = sqrt(8)
        assert g.norm(np.array([1.0, 2.0])) == pytest.approx(np.sqrt(8.0))

    def test_unit_weights_matches_euclidean(self, rng):
        g_euc = EuclideanGeometry()
        g_w = WeightedEuclideanGeometry(np.ones(5))
        a, b = rng.randn(5), rng.randn(5)
        np.testing.assert_allclose(g_w.inner(a, b), g_euc.inner(a, b), atol=1e-12)
        np.testing.assert_allclose(g_w.norm(a), g_euc.norm(a), atol=1e-12)

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="positive"):
            WeightedEuclideanGeometry(np.array([1.0, -0.5]))

    def test_rejects_zero_weight(self):
        with pytest.raises(ValueError, match="positive"):
            WeightedEuclideanGeometry(np.array([1.0, 0.0]))

    def test_batched(self):
        g = WeightedEuclideanGeometry(np.array([1.0, 2.0]))
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        norms = g.norm(X)
        np.testing.assert_allclose(norms, [1.0, np.sqrt(2.0)])


# --- Gram geometry-specific ---


class TestGramGeometry:
    def test_inner_known_value(self):
        G = np.array([[2.0, 1.0], [1.0, 3.0]])
        g = GramGeometry(G)
        # inner([1,0], [0,1]) = [1,0] @ [[2,1],[1,3]] @ [0,1] = [1,0] @ [1,3] = 1
        assert g.inner(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_identity_matches_euclidean(self, rng):
        g_euc = EuclideanGeometry()
        g_gram = GramGeometry(np.eye(5))
        a, b = rng.randn(5), rng.randn(5)
        np.testing.assert_allclose(g_gram.inner(a, b), g_euc.inner(a, b), atol=1e-12)
        np.testing.assert_allclose(g_gram.norm(a), g_euc.norm(a), atol=1e-12)

    def test_rejects_non_psd(self):
        with pytest.raises(ValueError, match="positive semi-definite"):
            GramGeometry(np.array([[1.0, 0.0], [0.0, -1.0]]))

    def test_batched(self):
        G = np.array([[2.0, 0.0], [0.0, 1.0]])
        g = GramGeometry(G)
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        norms = g.norm(X)
        np.testing.assert_allclose(norms, [np.sqrt(2.0), 1.0])

    def test_norm_nonneg_for_psd(self, rng):
        """GramGeometry norms should be non-negative for PSD matrices."""
        G = np.array([[2.0, 0.5], [0.5, 1.0]])
        g = GramGeometry(G)
        for _ in range(50):
            a = rng.randn(2)
            assert g.norm(a) >= -1e-14
