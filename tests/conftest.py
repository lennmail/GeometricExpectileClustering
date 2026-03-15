"""Shared test fixtures for the geomexp test suite."""

import matplotlib
import numpy as np
import pytest

from geomexp.clustering.clustering_base import ClusterResult
from geomexp.clustering.geometry import (
    EuclideanGeometry,
    GramGeometry,
    WeightedEuclideanGeometry,
)

matplotlib.use("Agg")


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def well_separated_2d(rng):
    """3-cluster dataset with wide separation (150 points, 2D).

    Centers at (-5,0), (5,0), (0,8) with std=0.5.
    """
    c0 = rng.randn(50, 2) * 0.5 + np.array([-5, 0])
    c1 = rng.randn(50, 2) * 0.5 + np.array([5, 0])
    c2 = rng.randn(50, 2) * 0.5 + np.array([0, 8])
    return np.vstack([c0, c1, c2])


@pytest.fixture
def well_separated_labels():
    return np.array([0] * 50 + [1] * 50 + [2] * 50)


@pytest.fixture
def simple_2d(rng):
    """Small 2D dataset (30 points)."""
    c0 = rng.randn(15, 2) * 0.3 + np.array([-3, 0])
    c1 = rng.randn(15, 2) * 0.3 + np.array([3, 0])
    return np.vstack([c0, c1])


@pytest.fixture
def high_dim_data(rng):
    """10-D dataset (60 points, 3 clusters)."""
    centers = rng.randn(3, 10) * 5
    data = []
    for c in centers:
        data.append(rng.randn(20, 10) * 0.5 + c)
    return np.vstack(data)


@pytest.fixture
def euclidean_geometry():
    return EuclideanGeometry()


@pytest.fixture
def weighted_geometry():
    return WeightedEuclideanGeometry(np.array([1.0, 2.0]))


@pytest.fixture
def gram_geometry():
    G = np.array([[2.0, 0.5], [0.5, 1.0]])
    return GramGeometry(G)


@pytest.fixture(params=["euclidean", "weighted", "gram"])
def any_geometry(request, euclidean_geometry, weighted_geometry, gram_geometry):
    """Parametrized fixture over all three geometry types."""
    return {"euclidean": euclidean_geometry, "weighted": weighted_geometry, "gram": gram_geometry}[
        request.param
    ]


@pytest.fixture
def sample_cluster_result():
    """A pre-built ClusterResult for visualization and alignment tests."""
    centers = np.array([[0.0, 0.0], [3.0, 3.0]])
    assignments = np.array([0, 0, 0, 1, 1, 1])
    indices = np.array([[0.3, 0.0], [0.0, 0.3]])
    return ClusterResult(
        assignments=assignments,
        centers=centers,
        objective=1.5,
        n_iterations=5,
        converged=True,
        metadata={"indices": indices},
    )
