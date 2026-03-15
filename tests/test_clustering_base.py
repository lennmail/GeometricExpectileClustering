"""Tests for geomexp/clustering/clustering_base.py.

Covers expectile_loss, ClusterResult, BaseClusterer, and IterativeClusterer.
"""

import numpy as np
import pytest

from geomexp.clustering.clustering_base import (
    BaseClusterer,
    ClusterResult,
    IterativeClusterer,
    expectile_loss,
)
from geomexp.clustering.geometry import EuclideanGeometry, WeightedEuclideanGeometry

# --- expectile_loss ---


class TestExpectileLoss:
    def test_zero_residual(self):
        """ell_u(0) = 0 regardless of u."""
        res = np.zeros(3)
        u = np.array([0.3, 0.2, 0.1])
        assert expectile_loss(res, u) == pytest.approx(0.0, abs=1e-14)

    def test_zero_index_reduces_to_half_norm_sq(self, rng):
        """With u=0: ell_0(t) = 0.5 * ||t||^2 (standard K-means loss)."""
        t = rng.randn(10, 3)
        u = np.zeros(3)
        losses = expectile_loss(t, u)
        expected = 0.5 * np.sum(t**2, axis=1)
        np.testing.assert_allclose(losses, expected, atol=1e-12)

    def test_nonneg_when_index_in_unit_ball(self, rng):
        """ell_u(t) >= 0 when ||u|| < 1 (Proposition 2.1)."""
        for _ in range(100):
            t = rng.randn(3)
            u = rng.randn(3)
            u = u / np.linalg.norm(u) * 0.9  # ||u|| = 0.9 < 1
            loss = expectile_loss(t, u)
            assert loss >= -1e-12, f"Loss should be non-negative, got {loss}"

    def test_asymmetry(self, rng):
        """ell_u(t) != ell_u(-t) when u != 0 and t != 0."""
        t = np.array([1.0, 2.0])
        u = np.array([0.5, 0.0])
        assert expectile_loss(t, u) != pytest.approx(expectile_loss(-t, u), abs=1e-10)

    def test_known_value(self):
        """Hand-computed: t=[1,0], u=[0.5,0], ||t||=1, <u,t>=0.5.
        Loss = 0.5*(1 + 1*0.5) = 0.75.
        """
        t = np.array([1.0, 0.0])
        u = np.array([0.5, 0.0])
        assert expectile_loss(t, u) == pytest.approx(0.75)

    def test_1d_input(self):
        """Single residual vector returns scalar."""
        t = np.array([1.0, 0.0])
        u = np.array([0.0, 0.0])
        result = expectile_loss(t, u)
        assert np.isscalar(result) or result.ndim == 0

    def test_batch_input(self, rng):
        """Batch (n, d) returns (n,) array."""
        t = rng.randn(20, 5)
        u = rng.randn(5) * 0.3
        result = expectile_loss(t, u)
        assert result.shape == (20,)

    def test_with_weighted_geometry(self, rng):
        """Loss computed with WeightedEuclideanGeometry uses weighted norms."""
        geom = WeightedEuclideanGeometry(np.array([2.0, 1.0]))
        t = np.array([1.0, 0.0])
        u = np.zeros(2)
        # Loss = 0.5 * ||t||_w^2 = 0.5 * (2*1 + 1*0) = 1.0
        assert expectile_loss(t, u, geometry=geom) == pytest.approx(1.0)

    def test_with_euclidean_geometry_matches_default(self, rng):
        """Explicitly passing EuclideanGeometry matches geometry=None."""
        geom = EuclideanGeometry()
        t = rng.randn(10, 3)
        u = rng.randn(3) * 0.5
        np.testing.assert_allclose(
            expectile_loss(t, u), expectile_loss(t, u, geometry=geom), atol=1e-12
        )

    def test_per_sample_index(self, rng):
        """(n, d) indices with (n, d) residuals."""
        t = rng.randn(5, 3)
        u = rng.randn(5, 3) * 0.3
        result = expectile_loss(t, u)
        assert result.shape == (5,)


# --- ClusterResult ---


class TestClusterResult:
    def test_construction(self):
        cr = ClusterResult(
            assignments=np.array([0, 1]),
            centers=np.array([[0.0], [1.0]]),
            objective=0.5,
            n_iterations=3,
        )
        assert cr.converged is True
        assert cr.metadata is None

    def test_with_metadata(self):
        cr = ClusterResult(
            assignments=np.array([0]),
            centers=np.array([[0.0]]),
            objective=0.0,
            n_iterations=1,
            metadata={"key": "value"},
        )
        assert cr.metadata["key"] == "value"


# --- BaseClusterer (via minimal concrete subclass) ---


class _MinimalClusterer(BaseClusterer):
    """Minimal concrete subclass for testing BaseClusterer."""

    def _initialize(self, X):
        centers = self._initialize_centers_random(X)
        return {"centers": centers, "assignments": self._assign_to_nearest_centers(X, centers)}

    def _fit_iteration(self, X, state):
        # Just re-assign (no update) — will converge immediately
        state["assignments"] = self._assign_to_nearest_centers(X, state["centers"])
        return state

    def _compute_objective(self, X, state):
        centers, asgn = state["centers"], state["assignments"]
        return float(np.sum((X - centers[asgn]) ** 2))

    def _extract_result(self, state, objective, n_iterations, converged):
        return ClusterResult(
            assignments=state["assignments"],
            centers=state["centers"],
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
        )


class TestBaseClusterer:
    def test_rejects_invalid_n_clusters(self):
        with pytest.raises(ValueError):
            _MinimalClusterer(n_clusters=0)

    def test_rejects_invalid_max_iter(self):
        with pytest.raises((ValueError, TypeError)):
            _MinimalClusterer(n_clusters=2, max_iter=0)

    def test_rejects_invalid_tol(self):
        with pytest.raises(ValueError):
            _MinimalClusterer(n_clusters=2, tol=-1)

    def test_validate_input_rejects_1d(self):
        m = _MinimalClusterer(n_clusters=2)
        with pytest.raises(ValueError, match="2-dimensional"):
            m.fit(np.array([1.0, 2.0, 3.0]))

    def test_validate_input_rejects_too_few_samples(self):
        m = _MinimalClusterer(n_clusters=5)
        with pytest.raises(ValueError, match="n_samples=2"):
            m.fit(np.array([[1.0], [2.0]]))

    def test_random_center_init(self, rng):
        m = _MinimalClusterer(n_clusters=3, random_state=42)
        X = rng.randn(20, 2)
        centers = m._initialize_centers_random(X)
        assert centers.shape == (3, 2)
        # Each center should be a data point
        for c in centers:
            assert any(np.allclose(c, x) for x in X)

    def test_assign_to_nearest(self):
        m = _MinimalClusterer(n_clusters=2)
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        asgn = m._assign_to_nearest_centers(X, centers)
        np.testing.assert_array_equal(asgn, [0, 1])

    def test_fit_returns_cluster_result(self, simple_2d):
        m = _MinimalClusterer(n_clusters=2, random_state=42)
        result = m.fit(simple_2d)
        assert isinstance(result, ClusterResult)
        assert result.assignments.shape == (len(simple_2d),)

    def test_additional_convergence_hook_default(self):
        m = _MinimalClusterer(n_clusters=2)
        assert m._additional_convergence_check({}) is False


# --- IterativeClusterer ---


class _MinimalIterative(IterativeClusterer):
    """Minimal concrete subclass for testing IterativeClusterer."""

    def _initialize(self, X):
        centers = self._initialize_centers_random(X)
        return {"centers": centers, "assignments": self._assign_to_nearest_centers(X, centers)}

    def _assignment_step(self, X, state):
        state["assignments"] = self._assign_to_nearest_centers(X, state["centers"])
        return state

    def _update_step(self, X, state):
        centers = state["centers"].copy()
        for k in range(self.n_clusters):
            mask = state["assignments"] == k
            if np.any(mask):
                centers[k] = np.mean(X[mask], axis=0)
        state["centers"] = centers
        return state

    def _compute_objective(self, X, state):
        return float(np.sum((X - state["centers"][state["assignments"]]) ** 2))

    def _extract_result(self, state, objective, n_iterations, converged):
        return ClusterResult(
            assignments=state["assignments"],
            centers=state["centers"],
            objective=objective,
            n_iterations=n_iterations,
            converged=converged,
        )


class TestIterativeClusterer:
    def test_fit_converges(self, simple_2d):
        m = _MinimalIterative(n_clusters=2, random_state=42)
        result = m.fit(simple_2d)
        assert isinstance(result, ClusterResult)
        assert result.converged

    def test_empty_cluster_handling(self):
        """Force an empty cluster and verify reinit."""
        m = _MinimalIterative(n_clusters=2, random_state=42)
        X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        # Manipulate state to create an empty cluster
        state = {
            "centers": np.array([[0.1, 0.1], [100.0, 100.0]]),
            "assignments": np.array([0, 0, 0]),  # cluster 1 is empty
        }
        new_state = m._handle_empty_clusters_in_state(X, state)
        asgn = new_state["assignments"]
        # After reinit, there should be no empty clusters
        for k in range(2):
            assert np.sum(asgn == k) > 0

    def test_nonempty_clusters_unchanged(self, rng):
        """When no clusters are empty, state is unchanged."""
        m = _MinimalIterative(n_clusters=2, random_state=42)
        X = rng.randn(10, 2)
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        asgn = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        state = {"centers": centers.copy(), "assignments": asgn.copy()}
        new_state = m._handle_empty_clusters_in_state(X, state)
        np.testing.assert_array_equal(new_state["centers"], centers)
