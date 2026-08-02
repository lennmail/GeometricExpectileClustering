"""Random sampling helpers shared by the clustering algorithms."""

from __future__ import annotations

import numpy as np


def draw_distinct_indices(n: int, count: int, rng: np.random.RandomState) -> list[int]:
    """Draw ``count`` distinct indices from ``range(n)``, redrawing on collision.

    Used to reseed empty clusters: handing two of them the same data point produces duplicate
    centers, which collapse again on the next assignment step and leave the clusters empty.

    Indices are drawn one at a time rather than through a single vectorised call, so drawing a
    single index consumes the random stream exactly as ``rng.choice(n)`` does.

    Args:
        n: Size of the population to draw from.
        count: Number of indices to draw. Values above ``n`` exhaust the population, after which
            repeats are unavoidable and are returned as drawn.
        rng: Random state.

    Returns:
        List of ``count`` indices, distinct whenever ``count <= n``.
    """
    chosen: list[int] = []
    seen: set[int] = set()

    for _ in range(count):
        idx = int(rng.choice(n))
        while idx in seen and len(seen) < n:
            idx = int(rng.choice(n))
        seen.add(idx)
        chosen.append(idx)

    return chosen
