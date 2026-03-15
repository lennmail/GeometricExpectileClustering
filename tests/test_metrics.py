"""Tests for geomexp/evaluation/metrics.py."""

import numpy as np
import pytest

from geomexp.clustering.clustering_algorithms import KMeans
from geomexp.evaluation.metrics import (
    adjusted_rand_index,
    davies_bouldin,
    misclassification_error,
    normalized_mutual_info,
    radius_ratio,
    run_methods,
    silhouette,
    stability_score,
    variation_of_information,
)

# --- ARI ---


class TestARI:
    def test_perfect(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(a, a) == pytest.approx(1.0)

    def test_symmetric(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([1, 1, 0, 0])
        assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_index(b, a))

    def test_permutation_invariant(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([2, 2, 0, 0, 1, 1])
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_random_near_zero(self, rng):
        a = rng.randint(0, 5, 200)
        b = rng.randint(0, 5, 200)
        assert abs(adjusted_rand_index(a, b)) < 0.2


# --- NMI ---


class TestNMI:
    def test_perfect(self):
        a = np.array([0, 0, 1, 1])
        assert normalized_mutual_info(a, a) == pytest.approx(1.0)

    def test_symmetric(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([1, 0, 0, 1])
        assert normalized_mutual_info(a, b) == pytest.approx(normalized_mutual_info(b, a))

    def test_range(self, rng):
        a = rng.randint(0, 3, 50)
        b = rng.randint(0, 3, 50)
        nmi = normalized_mutual_info(a, b)
        assert 0 <= nmi <= 1 + 1e-10


# --- VI ---


class TestVI:
    def test_perfect(self):
        a = np.array([0, 0, 1, 1])
        assert variation_of_information(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([1, 0, 0, 1])
        assert variation_of_information(a, b) == pytest.approx(variation_of_information(b, a))

    def test_nonneg(self, rng):
        a = rng.randint(0, 3, 50)
        b = rng.randint(0, 3, 50)
        assert variation_of_information(a, b) >= -1e-12

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            variation_of_information(np.array([0, 1]), np.array([0, 1, 2]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            variation_of_information(np.array([]), np.array([]))

    def test_known_value(self):
        """Two singleton clusters vs one big cluster."""
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 0, 0, 0])
        vi = variation_of_information(a, b)
        # VI > 0 since partitions differ
        assert vi > 0

    def test_all_same(self):
        """Both all-same: VI = 0."""
        a = np.array([0, 0, 0, 0])
        assert variation_of_information(a, a) == pytest.approx(0.0, abs=1e-12)


# --- Misclassification Error ---


class TestMisclassificationError:
    def test_perfect(self):
        a = np.array([0, 0, 1, 1])
        assert misclassification_error(a, a) == pytest.approx(0.0)

    def test_permutation_invariant(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([1, 1, 0, 0])
        assert misclassification_error(a, b) == pytest.approx(0.0)

    def test_range(self, rng):
        a = rng.randint(0, 3, 50)
        b = rng.randint(0, 3, 50)
        err = misclassification_error(a, b)
        assert 0 <= err <= 1 + 1e-10

    def test_known_value(self):
        """Half wrong: error = 0.5."""
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 1, 1])  # 1 wrong out of 4 under optimal permutation
        err = misclassification_error(a, b)
        assert err == pytest.approx(0.25)


# --- Silhouette ---


class TestSilhouette:
    def test_range(self, well_separated_2d, well_separated_labels):
        s = silhouette(well_separated_2d, well_separated_labels)
        assert -1 <= s <= 1

    def test_single_cluster_returns_zero(self, well_separated_2d):
        labels = np.zeros(len(well_separated_2d), dtype=int)
        assert silhouette(well_separated_2d, labels) == 0

    def test_each_its_own_returns_zero(self):
        X = np.random.randn(5, 2)
        labels = np.arange(5)
        assert silhouette(X, labels) == 0


# --- Davies-Bouldin ---


class TestDaviesBouldin:
    def test_nonneg(self, well_separated_2d, well_separated_labels):
        db = davies_bouldin(well_separated_2d, well_separated_labels)
        assert db >= 0

    def test_single_cluster_returns_inf(self, well_separated_2d):
        labels = np.zeros(len(well_separated_2d), dtype=int)
        assert davies_bouldin(well_separated_2d, labels) == float("inf")

    def test_each_its_own_returns_inf(self):
        X = np.random.randn(5, 2)
        labels = np.arange(5)
        assert davies_bouldin(X, labels) == float("inf")


# --- Stability Score ---


class TestStabilityScore:
    def test_returns_float(self, well_separated_2d):
        score = stability_score(
            well_separated_2d,
            lambda: KMeans(n_clusters=3, random_state=0),
            n_resamples=3,
            rng=np.random.default_rng(42),
        )
        assert isinstance(score, float)

    def test_well_separated_high_stability(self, well_separated_2d):
        score = stability_score(
            well_separated_2d,
            lambda: KMeans(n_clusters=3, random_state=0),
            n_resamples=5,
            rng=np.random.default_rng(42),
        )
        assert score > 0.5


# --- Radius Ratio ---


class TestRadiusRatio:
    def test_shape(self):
        X = np.random.randn(20, 2)
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        assignments = np.array([0] * 10 + [1] * 10)
        rr = radius_ratio(X, centers, assignments)
        assert rr.shape == (2,)

    def test_nonneg(self, rng):
        X = rng.randn(30, 2)
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        assignments = np.array([0] * 15 + [1] * 15)
        rr = radius_ratio(X, centers, assignments)
        assert np.all(rr >= 1 - 1e-10)

    def test_single_point_cluster(self):
        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        centers = np.array([[0.0, 0.0], [5.0, 5.0]])
        assignments = np.array([0, 1])
        rr = radius_ratio(X, centers, assignments)
        # Single point => ratio stays at default (1)
        np.testing.assert_array_equal(rr, [1.0, 1.0])


# --- run_methods ---


class TestRunMethods:
    def test_returns_dict(self, simple_2d):
        methods = [{"name": "km", "cls": KMeans, "kwargs": {"n_clusters": 2}}]
        results = run_methods(simple_2d, methods, n_inits=2, base_seed=0)
        assert isinstance(results, dict)
        assert "km" in results

    def test_multiple_methods(self, simple_2d):
        methods = [
            {"name": "km1", "cls": KMeans, "kwargs": {"n_clusters": 2}},
            {"name": "km2", "cls": KMeans, "kwargs": {"n_clusters": 2}},
        ]
        results = run_methods(simple_2d, methods, n_inits=2, base_seed=0)
        assert len(results) == 2

    def test_picks_best(self, simple_2d):
        methods = [{"name": "km", "cls": KMeans, "kwargs": {"n_clusters": 2}}]
        results = run_methods(simple_2d, methods, n_inits=5, base_seed=0)
        best = results["km"]
        # Verify it picked the best among runs
        for trial in range(5):
            single = KMeans(n_clusters=2, random_state=trial).fit(simple_2d)
            assert best.objective <= single.objective + 1e-10
