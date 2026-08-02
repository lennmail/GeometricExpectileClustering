"""Tests for geomexp/clustering/policies.py."""

import numpy as np

from geomexp.clustering.geometry import WeightedEuclideanGeometry
from geomexp.clustering.policies import (
    FarthestPointRule,
    LowestIndexTieBreak,
    RandomReinitRule,
    RandomTieBreak,
)

# --- LowestIndexTieBreak ---


class TestLowestIndexTieBreak:
    def test_clear_winner(self):
        rule = LowestIndexTieBreak()
        costs = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        rng = np.random.RandomState(0)
        asgn = rule(costs, rng)
        np.testing.assert_array_equal(asgn, [0, 1])

    def test_tie_picks_lowest_index(self):
        rule = LowestIndexTieBreak()
        costs = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]])
        rng = np.random.RandomState(0)
        asgn = rule(costs, rng)
        # np.argmin picks first occurrence
        np.testing.assert_array_equal(asgn, [0, 1])

    def test_single_sample(self):
        rule = LowestIndexTieBreak()
        costs = np.array([[5.0, 3.0]])
        rng = np.random.RandomState(0)
        asgn = rule(costs, rng)
        np.testing.assert_array_equal(asgn, [1])


# --- RandomTieBreak ---


class TestRandomTieBreak:
    def test_clear_winner(self):
        rule = RandomTieBreak()
        costs = np.array([[1.0, 5.0], [5.0, 1.0]])
        rng = np.random.RandomState(0)
        asgn = rule(costs, rng)
        np.testing.assert_array_equal(asgn, [0, 1])

    def test_tie_distribution(self):
        """On repeated exact ties, both clusters should be picked."""
        rule = RandomTieBreak()
        # Exact tie between cluster 0 and 1 for all samples
        costs = np.array([[1.0, 1.0]] * 200)
        rng = np.random.RandomState(42)
        asgn = rule(costs, rng)
        counts = np.bincount(asgn, minlength=2)
        # Each cluster should get some assignments
        assert counts[0] > 10
        assert counts[1] > 10

    def test_reproducible_with_same_seed(self):
        rule = RandomTieBreak()
        costs = np.array([[1.0, 1.0]] * 50)
        asgn1 = rule(costs, np.random.RandomState(42))
        asgn2 = rule(costs, np.random.RandomState(42))
        np.testing.assert_array_equal(asgn1, asgn2)


# --- RandomReinitRule ---


class TestRandomReinitRule:
    def test_replaces_empty(self):
        rule = RandomReinitRule()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assignments = np.array([0, 0, 0])  # cluster 1 empty
        centers = np.array([[0.5, 0.5], [10.0, 10.0]])
        indices = np.array([[0.3, 0.0], [0.3, 0.0]])
        rng = np.random.RandomState(42)

        new_c, _new_i = rule(X, assignments, centers, indices, rng)
        # Cluster 1 center should now be one of the data points
        assert any(np.allclose(new_c[1], x) for x in X)

    def test_index_reset_to_zero(self):
        rule = RandomReinitRule()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assignments = np.array([0, 0])
        centers = np.array([[0.5, 0.5], [10.0, 10.0]])
        indices = np.array([[0.3, 0.0], [0.3, 0.2]])
        rng = np.random.RandomState(42)

        _, new_i = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_i[1], [0.0, 0.0])

    def test_multiple_empty_get_distinct_centers(self):
        """Empty clusters draw without repetition, even from a small pool."""
        rule = RandomReinitRule()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assignments = np.zeros(len(X), dtype=int)  # clusters 1, 2 empty
        centers = np.zeros((3, 2))
        indices = np.full((3, 2), 0.3)

        for seed in range(25):
            new_c, _ = rule(X, assignments, centers, indices, np.random.RandomState(seed))
            assert not np.array_equal(new_c[1], new_c[2]), f"duplicate center at seed {seed}"

    def test_nonempty_untouched(self):
        rule = RandomReinitRule()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assignments = np.array([0, 1])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        indices = np.array([[0.3, 0.0], [0.0, 0.3]])
        rng = np.random.RandomState(42)

        new_c, new_i = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_c, centers)
        np.testing.assert_array_equal(new_i, indices)


# --- FarthestPointRule ---


class TestFarthestPointRule:
    def test_picks_farthest(self):
        rule = FarthestPointRule()
        # Points at 0, 1, 10. All assigned to cluster 0 (center=0).
        # Cluster 1 is empty. Farthest from assigned center = point at 10.
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        assignments = np.array([0, 0, 0])
        centers = np.array([[0.0, 0.0], [99.0, 99.0]])
        indices = np.array([[0.0, 0.0], [0.3, 0.0]])
        rng = np.random.RandomState(42)

        new_c, _new_i = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_c[1], [10.0, 0.0])

    def test_index_reset(self):
        rule = FarthestPointRule()
        X = np.array([[0.0, 0.0], [10.0, 0.0]])
        assignments = np.array([0, 0])
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        indices = np.array([[0.0, 0.0], [0.5, 0.5]])
        rng = np.random.RandomState(42)

        _, new_i = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_i[1], [0.0, 0.0])

    def test_with_weighted_geometry(self):
        geom = WeightedEuclideanGeometry(np.array([1.0, 4.0]))
        rule = FarthestPointRule(geometry=geom)
        # Under weighting [1, 4], y-distance is magnified.
        # Point (0, 3) has weighted dist from (0,0) = sqrt(0 + 4*9) = 6
        # Point (5, 0) has weighted dist from (0,0) = sqrt(25 + 0) = 5
        X = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 3.0]])
        assignments = np.array([0, 0, 0])
        centers = np.array([[0.0, 0.0], [99.0, 99.0]])
        indices = np.zeros((2, 2))
        rng = np.random.RandomState(42)

        new_c, _ = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_c[1], [0.0, 3.0])

    def test_no_empty_noop(self):
        rule = FarthestPointRule()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assignments = np.array([0, 1])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        indices = np.array([[0.1, 0.0], [0.0, 0.1]])
        rng = np.random.RandomState(42)

        new_c, new_i = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_c, centers)
        np.testing.assert_array_equal(new_i, indices)

    def test_multiple_empty_get_distinct_centers(self):
        """Several empty clusters take successively farther points, never the same one."""
        rule = FarthestPointRule()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0], [20.0, 0.0]])
        assignments = np.zeros(len(X), dtype=int)  # clusters 1, 2, 3 all empty
        centers = np.zeros((4, 2))
        indices = np.full((4, 2), 0.3)
        rng = np.random.RandomState(42)

        new_c, _ = rule(X, assignments, centers, indices, rng)
        np.testing.assert_array_equal(new_c[1], [20.0, 0.0])
        np.testing.assert_array_equal(new_c[2], [9.0, 0.0])
        np.testing.assert_array_equal(new_c[3], [5.0, 0.0])
        assert len({tuple(c) for c in new_c}) == len(new_c)
