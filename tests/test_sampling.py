"""Tests for geomexp/utils/sampling.py."""

import numpy as np

from geomexp.utils.sampling import draw_distinct_indices


class TestDrawDistinctIndices:
    def test_returns_requested_count(self):
        draws = draw_distinct_indices(10, 4, np.random.RandomState(0))
        assert len(draws) == 4

    def test_indices_in_range(self):
        draws = draw_distinct_indices(5, 5, np.random.RandomState(0))
        assert all(0 <= i < 5 for i in draws)

    def test_distinct_when_population_suffices(self):
        for seed in range(50):
            draws = draw_distinct_indices(4, 4, np.random.RandomState(seed))
            assert len(set(draws)) == 4

    def test_single_draw_matches_plain_choice(self):
        """One index must consume the stream exactly as rng.choice(n) does."""
        expected = int(np.random.RandomState(7).choice(12))
        assert draw_distinct_indices(12, 1, np.random.RandomState(7)) == [expected]

    def test_exhausted_population_terminates(self):
        """Asking for more than the population exists cannot loop forever."""
        draws = draw_distinct_indices(2, 5, np.random.RandomState(0))
        assert len(draws) == 5

    def test_zero_count(self):
        assert draw_distinct_indices(5, 0, np.random.RandomState(0)) == []

    def test_reproducible_with_same_seed(self):
        a = draw_distinct_indices(20, 6, np.random.RandomState(3))
        b = draw_distinct_indices(20, 6, np.random.RandomState(3))
        assert a == b
