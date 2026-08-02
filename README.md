[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lennmail.github.io/GeometricExpectileClustering/)
[![Coverage](https://img.shields.io/endpoint?url=https://lennmail.github.io/GeometricExpectileClustering/coverage.json)](https://lennmail.github.io/GeometricExpectileClustering/)
[![CI](https://github.com/lennmail/GeometricExpectileClustering/actions/workflows/ci.yml/badge.svg)](https://github.com/lennmail/GeometricExpectileClustering/actions/workflows/ci.yml)

# Geometric Expectile Clustering

A Python library implementing geometric $K$-expectile clustering (GKE), a Lloyd-type algorithm that extends K-means with directional asymmetry via the geometric expectile loss. Developed as part of a Master's thesis at EPFL & Oxford. The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/lennmail/GeometricExpectileClustering.git
cd GeometricExpectileClustering
uv sync
```

## Getting started

### Basic clustering

```python
import numpy as np
from geomexp import GeometricExpectileClustering

X = np.random.randn(300, 2)

gec = GeometricExpectileClustering(n_clusters=3, index_radius=0.5, random_state=42)
result = gec.fit(X)

result.assignments  # cluster labels
result.centers      # cluster centroids
result.metadata["indices"]  # learned index vectors
```

Setting `index_radius=0` recovers standard K-means. As the radius approaches 1, clusters become more directionally asymmetric.

### Custom index vectors

By default, index vectors are learned adaptively via `BestResponseIndexStrategy`. You can fix them instead:

```python
from geomexp import GeometricExpectileClustering, GlobalIndexStrategy

strategy = GlobalIndexStrategy(direction=np.array([1, 0]))
gec = GeometricExpectileClustering(
    n_clusters=3, index_radius=0.5, index_strategy=strategy, random_state=42
)
```

`ClusterSpecificIndexStrategy` sets a fixed direction per cluster, and `CustomIndexStrategy` accepts a callable.

### Geometries

The default geometry is Euclidean. For functional data on a grid, pass quadrature weights via `WeightedEuclideanGeometry`. For data represented as basis coefficients, use `GramGeometry` with the basis Gram matrix. These change the inner product used throughout the algorithm (assignment, center updates, index updates).

### Visualization

```python
from geomexp.visualization import PlotStyle, ClusterVisualizer

style = PlotStyle(figsize=(6, 4), fontsize=10, contour_color="blue")
viz = ClusterVisualizer(style)
viz.plot_cluster_assignments(X, result)
```

`PlotStyle` centralises all visual parameters (colors, sizes, fonts) so that figures in a study share a uniform look. `ClusterVisualizer` provides methods for scatter plots, decision boundaries, density contours, and expectile loss curves.

`PlotStyle` renders text through LaTeX by default, which requires a working LaTeX installation. Pass `use_latex=False` to fall back to matplotlib's built-in text rendering.

## Implementing your own clustering algorithm

Subclass `BaseClusterer` and implement four methods:

- `_initialize(X)` — set up initial state (centers, assignments, etc.) and return a state dict.
- `_fit_iteration(X, state)` — one iteration of the algorithm; return the updated state dict.
- `_compute_objective(X, state)` — return a scalar loss value for convergence checking.
- `_extract_result(state, objective, n_iterations, converged)` — pack the final state into a `ClusterResult`.

The base class provides the convergence loop (`fit`), input validation, and random center initialisation. For Lloyd-type algorithms that follow an assign-update pattern, inherit from `IterativeClusterer` instead — it provides a default `_fit_iteration` that calls `_assignment_step` and `_update_step`, so you only need to implement those two hooks.

## License

TBD
