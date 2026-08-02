"""Tests for geomexp/clustering/clustering_algorithms.py (KMeans and GEC)."""

import numpy as np
import pytest

from geomexp.clustering.clustering_algorithms import GeometricExpectileClustering, KMeans
from geomexp.clustering.clustering_base import ClusterResult
from geomexp.clustering.geometry import GramGeometry, WeightedEuclideanGeometry
from geomexp.clustering.index_strategies import (
    ClusterSpecificIndexStrategy,
    CustomIndexStrategy,
    GlobalIndexStrategy,
)
from geomexp.clustering.policies import FarthestPointRule, RandomTieBreak
from geomexp.evaluation.metrics import adjusted_rand_index

# --- KMeans ---


class TestKMeans:
    def test_well_separated(self, well_separated_2d, well_separated_labels):
        km = KMeans(n_clusters=3, random_state=42)
        result = km.fit(well_separated_2d)
        ari = adjusted_rand_index(well_separated_labels, result.assignments)
        assert ari > 0.95

    def test_centers_are_means(self, well_separated_2d):
        km = KMeans(n_clusters=3, random_state=42)
        result = km.fit(well_separated_2d)
        for k in range(3):
            mask = result.assignments == k
            cluster_mean = well_separated_2d[mask].mean(axis=0)
            np.testing.assert_allclose(result.centers[k], cluster_mean, atol=1e-6)

    def test_deterministic_with_seed(self, well_separated_2d):
        r1 = KMeans(n_clusters=3, random_state=0).fit(well_separated_2d)
        r2 = KMeans(n_clusters=3, random_state=0).fit(well_separated_2d)
        np.testing.assert_array_equal(r1.assignments, r2.assignments)

    def test_result_shapes(self, well_separated_2d):
        result = KMeans(n_clusters=3, random_state=42).fit(well_separated_2d)
        assert result.assignments.shape == (150,)
        assert result.centers.shape == (3, 2)

    def test_converges(self, well_separated_2d):
        result = KMeans(n_clusters=3, random_state=42).fit(well_separated_2d)
        assert result.converged

    def test_single_cluster(self, simple_2d):
        result = KMeans(n_clusters=1, random_state=42).fit(simple_2d)
        np.testing.assert_array_equal(result.assignments, np.zeros(len(simple_2d)))
        np.testing.assert_allclose(result.centers[0], simple_2d.mean(axis=0), atol=1e-8)

    def test_n_clusters_equals_n(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = KMeans(n_clusters=3, random_state=42).fit(X)
        assert len(set(result.assignments)) == 3

    def test_objective_nonneg(self, well_separated_2d):
        result = KMeans(n_clusters=3, random_state=42).fit(well_separated_2d)
        assert result.objective >= 0


# --- GeometricExpectileClustering ---


class TestGEC:
    def test_well_separated(self, well_separated_2d, well_separated_labels):
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=0.5,
            n_init=3,
            max_iter=20,
            center_steps=10,
            random_state=42,
        )
        result = gec.fit(well_separated_2d)
        ari = adjusted_rand_index(well_separated_labels, result.assignments)
        assert ari > 0.9

    def test_radius_zero_like_kmeans(self, well_separated_2d):
        """GEC with r=0 should behave like K-means."""
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=0.0,
            n_init=3,
            max_iter=30,
            center_steps=10,
            random_state=42,
        )
        km = KMeans(n_clusters=3, random_state=42, max_iter=30)
        r_gec = gec.fit(well_separated_2d)
        r_km = km.fit(well_separated_2d)
        # Both should recover clusters well
        ari_gec = adjusted_rand_index(np.array([0] * 50 + [1] * 50 + [2] * 50), r_gec.assignments)
        ari_km = adjusted_rand_index(np.array([0] * 50 + [1] * 50 + [2] * 50), r_km.assignments)
        assert ari_gec > 0.9
        assert ari_km > 0.9

    def test_index_norms(self, well_separated_2d):
        r = 0.6
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=r,
            n_init=2,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = gec.fit(well_separated_2d)
        indices = result.metadata["indices"]
        for row in indices:
            norm = np.linalg.norm(row)
            if norm > 1e-12:
                assert norm == pytest.approx(r, abs=0.05)

    def test_deterministic_with_seed(self, simple_2d):
        kwargs = {
            "n_clusters": 2,
            "index_radius": 0.3,
            "n_init": 2,
            "max_iter": 10,
            "center_steps": 5,
            "random_state": 42,
        }
        r1 = GeometricExpectileClustering(**kwargs).fit(simple_2d)
        r2 = GeometricExpectileClustering(**kwargs).fit(simple_2d)
        np.testing.assert_array_equal(r1.assignments, r2.assignments)

    def test_multi_restart_picks_best(self, simple_2d):
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            n_init=5,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        # Run single inits and check best is <= any single
        for trial in range(5):
            single = GeometricExpectileClustering(
                n_clusters=2,
                index_radius=0.3,
                n_init=1,
                max_iter=10,
                center_steps=5,
                random_state=42 + trial,
            ).fit(simple_2d)
            # Best should be <= single (with floating tolerance)
            assert result.objective <= single.objective + 1e-6

    def test_objective_nonneg(self, well_separated_2d):
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=0.5,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(well_separated_2d)
        assert result.objective >= 0

    def test_metadata_contains_indices(self, simple_2d):
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.5,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert "indices" in result.metadata
        assert result.metadata["indices"].shape == (2, 2)

    def test_result_shapes(self, simple_2d):
        result = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.5,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        ).fit(simple_2d)
        assert result.assignments.shape == (30,)
        assert result.centers.shape == (2, 2)

    def test_with_global_strategy(self, simple_2d):
        strategy = GlobalIndexStrategy(direction=np.array([1.0, 0.0]))
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            index_strategy=strategy,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_cluster_specific_strategy(self, simple_2d):
        dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
        strategy = ClusterSpecificIndexStrategy(directions=dirs)
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            index_strategy=strategy,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_custom_strategy(self, simple_2d):
        strategy = CustomIndexStrategy(update_fn=lambda X, a, c, i, r: i)
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            index_strategy=strategy,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_weighted_geometry(self, simple_2d):
        geom = WeightedEuclideanGeometry(np.array([1.0, 2.0]))
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            geometry=geom,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_gram_geometry(self, simple_2d):
        G = np.array([[2.0, 0.5], [0.5, 1.0]])
        geom = GramGeometry(G)
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            geometry=geom,
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_random_tiebreak(self, simple_2d):
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            tie_break_rule=RandomTieBreak(),
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_with_farthest_point_rule(self, simple_2d):
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            empty_cluster_rule=FarthestPointRule(),
            n_init=1,
            max_iter=5,
            center_steps=3,
            random_state=42,
        )
        result = gec.fit(simple_2d)
        assert isinstance(result, ClusterResult)

    def test_rejects_invalid_index_radius(self):
        with pytest.raises(ValueError):
            GeometricExpectileClustering(n_clusters=2, index_radius=1.0)

    def test_rejects_invalid_center_lr(self):
        with pytest.raises(ValueError):
            GeometricExpectileClustering(n_clusters=2, center_lr=0)

    def test_rejects_invalid_center_steps(self):
        with pytest.raises((ValueError, TypeError)):
            GeometricExpectileClustering(n_clusters=2, center_steps=0)

    def test_single_point_cluster(self):
        """Cluster with a single point: center should be that point."""
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        gec = GeometricExpectileClustering(
            n_clusters=2,
            index_radius=0.3,
            n_init=1,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = gec.fit(X)
        # Each cluster has exactly one point, so centers should be exact
        for k in range(2):
            mask = result.assignments == k
            if np.sum(mask) == 1:
                np.testing.assert_allclose(result.centers[k], X[mask][0], atol=1e-6)

    def test_convergence_flag(self, well_separated_2d):
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=0.5,
            n_init=1,
            max_iter=100,
            center_steps=10,
            random_state=42,
        )
        result = gec.fit(well_separated_2d)
        assert result.converged

    def test_high_dimensional(self, high_dim_data):
        gec = GeometricExpectileClustering(
            n_clusters=3,
            index_radius=0.3,
            n_init=2,
            max_iter=10,
            center_steps=5,
            random_state=42,
        )
        result = gec.fit(high_dim_data)
        assert result.assignments.shape == (60,)
        assert result.centers.shape == (3, 10)


class TestRandomStateSemantics:
    """random_state=None must be nondeterministic; an explicit seed must be reproducible."""

    @staticmethod
    def _rng_key(model):
        """Fingerprint the RNG the last restart ran with."""
        return tuple(model._rng.get_state()[1][:4])

    def test_none_does_not_fall_back_to_a_fixed_seed(self, simple_2d):
        """None used to be silently treated as seed 0, making every run identical."""
        keys = set()
        for _ in range(5):
            model = GeometricExpectileClustering(n_clusters=2, index_radius=0.5, n_init=1)
            model.fit(simple_2d)
            keys.add(self._rng_key(model))
        assert len(keys) > 1

    def test_explicit_seed_uses_that_seed(self, simple_2d):
        first = GeometricExpectileClustering(
            n_clusters=2, index_radius=0.5, n_init=1, random_state=7
        )
        second = GeometricExpectileClustering(
            n_clusters=2, index_radius=0.5, n_init=1, random_state=7
        )
        first.fit(simple_2d)
        second.fit(simple_2d)
        assert self._rng_key(first) == self._rng_key(second)

    def test_explicit_seed_is_reproducible(self, simple_2d):
        def run():
            return GeometricExpectileClustering(
                n_clusters=2, index_radius=0.5, n_init=2, random_state=7
            ).fit(simple_2d)

        first, second = run(), run()
        np.testing.assert_array_equal(first.assignments, second.assignments)
        np.testing.assert_allclose(first.centers, second.centers)
