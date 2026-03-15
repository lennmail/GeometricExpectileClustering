"""Tests for geomexp/clustering/index_strategies.py."""

import numpy as np
import pytest

from geomexp.clustering.geometry import WeightedEuclideanGeometry
from geomexp.clustering.index_strategies import (
    BestResponseIndexStrategy,
    ClusterSpecificIndexStrategy,
    CustomIndexStrategy,
    GlobalIndexStrategy,
)

# --- GlobalIndexStrategy ---


class TestGlobalIndexStrategy:
    def test_initialize_shape(self):
        s = GlobalIndexStrategy(direction=np.array([1.0, 0.0]))
        idx = s.initialize(3, 2, 0.5, np.random.RandomState(0))
        assert idx.shape == (3, 2)

    def test_initialize_all_same(self):
        s = GlobalIndexStrategy(direction=np.array([1.0, 0.0]))
        idx = s.initialize(3, 2, 0.5, np.random.RandomState(0))
        np.testing.assert_array_equal(idx[0], idx[1])
        np.testing.assert_array_equal(idx[1], idx[2])

    def test_initialize_norm_equals_radius(self):
        s = GlobalIndexStrategy(direction=np.array([3.0, 4.0]))
        idx = s.initialize(2, 2, 0.7, np.random.RandomState(0))
        for row in idx:
            assert np.linalg.norm(row) == pytest.approx(0.7, abs=1e-12)

    def test_update_returns_copy(self):
        s = GlobalIndexStrategy(direction=np.array([1.0, 0.0]))
        idx = s.initialize(2, 2, 0.5, np.random.RandomState(0))
        updated = s.update(np.zeros((3, 2)), np.array([0, 0, 1]), np.zeros((2, 2)), idx, 0.5)
        assert updated is not idx
        np.testing.assert_array_equal(updated, idx)

    def test_rejects_zero_direction(self):
        with pytest.raises(ValueError, match="non-zero"):
            GlobalIndexStrategy(direction=np.array([0.0, 0.0]))

    def test_rejects_wrong_dimension(self):
        s = GlobalIndexStrategy(direction=np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="features"):
            s.initialize(2, 2, 0.5, np.random.RandomState(0))

    def test_direction_proportional(self):
        """Output direction should be proportional to input direction."""
        s = GlobalIndexStrategy(direction=np.array([3.0, 4.0]))
        idx = s.initialize(1, 2, 0.5, np.random.RandomState(0))
        # Should point in same direction as [3, 4]
        expected_dir = np.array([3.0, 4.0]) / 5.0
        actual_dir = idx[0] / np.linalg.norm(idx[0])
        np.testing.assert_allclose(actual_dir, expected_dir, atol=1e-12)


# --- ClusterSpecificIndexStrategy ---


class TestClusterSpecificIndexStrategy:
    def test_initialize_shape(self):
        dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
        s = ClusterSpecificIndexStrategy(directions=dirs)
        idx = s.initialize(2, 2, 0.5, np.random.RandomState(0))
        assert idx.shape == (2, 2)

    def test_initialize_norms(self):
        dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
        s = ClusterSpecificIndexStrategy(directions=dirs)
        idx = s.initialize(2, 2, 0.6, np.random.RandomState(0))
        for row in idx:
            assert np.linalg.norm(row) == pytest.approx(0.6, abs=1e-12)

    def test_zero_direction_yields_zero(self):
        dirs = np.array([[1.0, 0.0], [0.0, 0.0]])
        s = ClusterSpecificIndexStrategy(directions=dirs)
        idx = s.initialize(2, 2, 0.5, np.random.RandomState(0))
        np.testing.assert_array_equal(idx[1], [0.0, 0.0])
        assert np.linalg.norm(idx[0]) == pytest.approx(0.5, abs=1e-12)

    def test_wrong_n_clusters_raises(self):
        dirs = np.array([[1.0, 0.0]])  # 1 direction
        s = ClusterSpecificIndexStrategy(directions=dirs)
        with pytest.raises(ValueError, match="Expected 3"):
            s.initialize(3, 2, 0.5, np.random.RandomState(0))

    def test_update_returns_copy(self):
        dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
        s = ClusterSpecificIndexStrategy(directions=dirs)
        idx = s.initialize(2, 2, 0.5, np.random.RandomState(0))
        updated = s.update(np.zeros((3, 2)), np.array([0, 0, 1]), np.zeros((2, 2)), idx, 0.5)
        assert updated is not idx


# --- BestResponseIndexStrategy ---


class TestBestResponseIndexStrategy:
    def test_initialize_zeros(self):
        s = BestResponseIndexStrategy()
        idx = s.initialize(3, 4, 0.5, np.random.RandomState(0))
        np.testing.assert_array_equal(idx, np.zeros((3, 4)))

    def test_update_norms(self, rng):
        """After update, each nonzero index row should have norm == index_radius."""
        s = BestResponseIndexStrategy()
        X = rng.randn(30, 2)
        # Create a skewed cluster
        X[:15, 0] += 5  # cluster 0 offset in x
        assignments = np.array([0] * 15 + [1] * 15)
        centers = np.array([X[:15].mean(axis=0), X[15:].mean(axis=0)])
        indices = np.zeros((2, 2))

        new_idx = s.update(X, assignments, centers, indices, 0.5)
        for row in new_idx:
            norm = np.linalg.norm(row)
            if norm > 1e-12:
                assert norm == pytest.approx(0.5, abs=1e-10)

    def test_known_direction(self):
        """With a cluster whose points are all to the right of center,
        s_k should point right, so u_k* should point left (u = -r * s_k / ||s_k||).
        """
        s = BestResponseIndexStrategy()
        center = np.array([0.0, 0.0])
        # All points to the right of center
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assignments = np.array([0, 0, 0])
        centers = np.array([center])
        indices = np.zeros((1, 2))

        new_idx = s.update(X, assignments, centers, indices, 0.5)
        # s_k = sum ||x_i - c||_H * (x_i - c) = 1*[1,0] + 2*[2,0] + 3*[3,0] = [14, 0]
        # u_k = -0.5 * [14, 0] / 14 = [-0.5, 0]
        np.testing.assert_allclose(new_idx[0], [-0.5, 0.0], atol=1e-12)

    def test_empty_cluster_unchanged(self):
        s = BestResponseIndexStrategy()
        X = np.array([[1.0, 0.0], [2.0, 0.0]])
        assignments = np.array([0, 0])  # cluster 1 empty
        centers = np.array([[1.5, 0.0], [10.0, 10.0]])
        indices = np.array([[0.0, 0.0], [0.3, 0.3]])

        new_idx = s.update(X, assignments, centers, indices, 0.5)
        np.testing.assert_array_equal(new_idx[1], [0.3, 0.3])

    def test_symmetric_cluster_unchanged(self):
        """Perfectly symmetric cluster: s_k = 0, index should stay the same."""
        s = BestResponseIndexStrategy()
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        assignments = np.array([0, 0, 0, 0])
        centers = np.array([[0.0, 0.0]])
        indices = np.array([[0.3, 0.0]])

        new_idx = s.update(X, assignments, centers, indices, 0.5)
        # s_k = 1*[1,0] + 1*[-1,0] + 1*[0,1] + 1*[0,-1] = [0, 0]
        # norm_s = 0, so index should be unchanged
        np.testing.assert_array_equal(new_idx[0], [0.3, 0.0])

    def test_with_weighted_geometry(self, rng):
        geom = WeightedEuclideanGeometry(np.array([2.0, 1.0]))
        s = BestResponseIndexStrategy(geometry=geom)
        X = rng.randn(20, 2)
        assignments = np.array([0] * 10 + [1] * 10)
        centers = np.array([X[:10].mean(axis=0), X[10:].mean(axis=0)])
        indices = np.zeros((2, 2))

        new_idx = s.update(X, assignments, centers, indices, 0.5)
        for row in new_idx:
            norm = geom.norm(row)
            if norm > 1e-12:
                assert norm == pytest.approx(0.5, abs=1e-10)


# --- CustomIndexStrategy ---


class TestCustomIndexStrategy:
    def test_with_init_fn(self):
        def my_init(n_clusters, n_features, index_radius, rng):
            return np.ones((n_clusters, n_features)) * 0.1

        s = CustomIndexStrategy(update_fn=lambda *a: a[3], init_fn=my_init)
        idx = s.initialize(2, 3, 0.5, np.random.RandomState(0))
        np.testing.assert_array_equal(idx, np.ones((2, 3)) * 0.1)

    def test_without_init_fn(self):
        s = CustomIndexStrategy(update_fn=lambda *a: a[3])
        idx = s.initialize(2, 3, 0.5, np.random.RandomState(0))
        np.testing.assert_array_equal(idx, np.zeros((2, 3)))

    def test_update_fn_called(self):
        called = {"count": 0}

        def my_update(X, assignments, centers, indices, index_radius):
            called["count"] += 1
            return indices * 2

        s = CustomIndexStrategy(update_fn=my_update)
        idx = np.ones((2, 3))
        result = s.update(np.zeros((5, 3)), np.array([0, 0, 1, 1, 1]), np.zeros((2, 3)), idx, 0.5)
        assert called["count"] == 1
        np.testing.assert_array_equal(result, idx * 2)
