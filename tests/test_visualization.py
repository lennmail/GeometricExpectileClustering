"""Tests for geomexp/visualization/visualization.py.

All tests use the Agg backend (set in conftest.py) to avoid display requirements.
"""

from matplotlib.figure import Figure
import numpy as np
import pytest

from geomexp.clustering.clustering_base import ClusterResult
from geomexp.visualization.visualization import (
    ClusterVisualizer,
    PlotStyle,
    _align_result_to_reference,
)


@pytest.fixture
def viz():
    return ClusterVisualizer(PlotStyle(use_latex=False, dpi=72))


@pytest.fixture
def X_2d(rng):
    return rng.randn(30, 2)


@pytest.fixture
def result_2d():
    return ClusterResult(
        assignments=np.array([0] * 15 + [1] * 15),
        centers=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        objective=1.0,
        n_iterations=5,
        converged=True,
        metadata={"indices": np.array([[0.3, 0.0], [0.0, 0.3]])},
    )


@pytest.fixture
def result_no_indices():
    return ClusterResult(
        assignments=np.array([0] * 15 + [1] * 15),
        centers=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        objective=1.0,
        n_iterations=5,
        converged=True,
    )


# --- PlotStyle ---


class TestPlotStyle:
    def test_defaults(self):
        s = PlotStyle()
        assert s.figsize == (3.5, 3.5)
        assert s.dpi == 300

    def test_palette_populated(self):
        s = PlotStyle()
        assert s.color_palette is not None
        assert len(s.color_palette) >= 5

    def test_custom_values(self):
        s = PlotStyle(figsize=(10, 8), dpi=150, use_latex=False)
        assert s.figsize == (10, 8)
        assert s.dpi == 150


# --- ClusterVisualizer ---


class TestClusterVisualizer:
    def test_rejects_non_plotstyle(self):
        with pytest.raises(TypeError, match="PlotStyle"):
            ClusterVisualizer({"figsize": (5, 5)})

    def test_construction(self):
        viz = ClusterVisualizer(PlotStyle(use_latex=False))
        assert viz.style is not None


class TestPlotClusterAssignments:
    def test_returns_figure(self, viz, X_2d, result_2d):
        fig = viz.plot_cluster_assignments(X_2d, result_2d)
        assert isinstance(fig, Figure)

    def test_rejects_non_2d(self, viz, result_2d):
        X_3d = np.random.randn(15, 3)
        with pytest.raises(ValueError, match="2D"):
            viz.plot_cluster_assignments(X_3d, result_2d)

    def test_with_options(self, viz, X_2d, result_2d):
        fig = viz.plot_cluster_assignments(
            X_2d, result_2d, show_centers=False, show_indices=False, show_legend=True
        )
        assert isinstance(fig, Figure)

    def test_no_indices_in_metadata(self, viz, X_2d, result_no_indices):
        fig = viz.plot_cluster_assignments(X_2d, result_no_indices, show_indices=True)
        assert isinstance(fig, Figure)


class TestPlotDecisionBoundaries:
    def test_returns_figure(self, viz, X_2d, result_2d):
        fig = viz.plot_decision_boundaries(X_2d, result_2d, resolution=10)
        assert isinstance(fig, Figure)

    def test_rejects_non_2d(self, viz, result_2d):
        with pytest.raises(ValueError, match="2D"):
            viz.plot_decision_boundaries(np.random.randn(15, 3), result_2d)

    def test_with_limits(self, viz, X_2d, result_2d):
        fig = viz.plot_decision_boundaries(
            X_2d, result_2d, resolution=10, x_lim=(-5, 5), y_lim=(-5, 5)
        )
        assert isinstance(fig, Figure)

    def test_no_indices(self, viz, X_2d, result_no_indices):
        fig = viz.plot_decision_boundaries(X_2d, result_no_indices, resolution=10)
        assert isinstance(fig, Figure)

    def test_show_options(self, viz, X_2d, result_2d):
        fig = viz.plot_decision_boundaries(
            X_2d,
            result_2d,
            resolution=10,
            show_points=False,
            show_contour_lines=False,
            show_centers=False,
            show_regions=False,
        )
        assert isinstance(fig, Figure)


class TestPlotDensityContours:
    def test_returns_figure(self, viz, X_2d):
        fig = viz.plot_density_contours(X_2d)
        assert isinstance(fig, Figure)

    def test_rejects_non_2d(self, viz):
        with pytest.raises(ValueError, match="2D"):
            viz.plot_density_contours(np.random.randn(15, 3))


class TestPlotDensityWithBoundaries:
    def test_returns_figure(self, viz, X_2d, result_2d):
        fig = viz.plot_density_with_boundaries(X_2d, result_2d, resolution=10)
        assert isinstance(fig, Figure)

    def test_rejects_non_2d(self, viz, result_2d):
        with pytest.raises(ValueError, match="2D"):
            viz.plot_density_with_boundaries(np.random.randn(15, 3), result_2d)


class TestPlotExpectileCurves:
    def test_returns_figure(self, viz):
        fig = viz.plot_expectile_curves(
            center=np.array([0.0, 0.0]),
            index_vector=np.array([0.3, 0.0]),
            resolution=30,
        )
        assert isinstance(fig, Figure)

    def test_no_arrow(self, viz):
        fig = viz.plot_expectile_curves(
            center=np.array([0.0, 0.0]),
            index_vector=np.array([0.0, 0.0]),  # zero vector, no arrow
            show_index_arrow=False,
            resolution=30,
        )
        assert isinstance(fig, Figure)


class TestPlotLossComparison:
    def test_returns_figure(self, viz):
        funcs = [("sq", lambda x: x**2), ("abs", lambda x: np.abs(x))]
        fig = viz.plot_loss_comparison(funcs)
        assert isinstance(fig, Figure)

    def test_with_legend(self, viz):
        funcs = [("sq", lambda x: x**2)]
        fig = viz.plot_loss_comparison(funcs, show_legend=True, title="Test")
        assert isinstance(fig, Figure)


class TestCreateComparisonFigure:
    def test_assignments(self, viz, X_2d, result_2d):
        results = [("A", result_2d), ("B", result_2d)]
        fig = viz.create_comparison_figure(X_2d, results, plot_type="assignments")
        assert isinstance(fig, Figure)

    def test_boundaries(self, viz, X_2d, result_2d):
        results = [("A", result_2d)]
        fig = viz.create_comparison_figure(
            X_2d,
            results,
            plot_type="boundaries",
            resolution=10,
        )
        assert isinstance(fig, Figure)

    def test_density(self, viz, X_2d, result_2d):
        results = [("A", result_2d)]
        fig = viz.create_comparison_figure(X_2d, results, plot_type="density")
        assert isinstance(fig, Figure)


class TestSaveFigure:
    def test_save(self, viz, X_2d, result_2d, tmp_path):
        fig = viz.plot_cluster_assignments(X_2d, result_2d)
        path = tmp_path / "test.png"
        ClusterVisualizer.save_figure(fig, str(path))
        assert path.exists()


# --- _align_result_to_reference ---


class TestAlignResultToReference:
    def test_identity(self, sample_cluster_result):
        aligned = _align_result_to_reference(sample_cluster_result, sample_cluster_result)
        np.testing.assert_array_equal(aligned.assignments, sample_cluster_result.assignments)

    def test_permuted_labels(self):
        ref = ClusterResult(
            assignments=np.array([0, 0, 1, 1]),
            centers=np.array([[0.0, 0.0], [5.0, 5.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
        )
        target = ClusterResult(
            assignments=np.array([1, 1, 0, 0]),
            centers=np.array([[5.0, 5.0], [0.0, 0.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
        )
        aligned = _align_result_to_reference(ref, target)
        np.testing.assert_array_equal(aligned.assignments, ref.assignments)
        np.testing.assert_allclose(aligned.centers, ref.centers)

    def test_different_k_returns_target(self):
        ref = ClusterResult(
            assignments=np.array([0, 0, 1]),
            centers=np.array([[0.0], [1.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
        )
        target = ClusterResult(
            assignments=np.array([0, 1, 2]),
            centers=np.array([[0.0], [1.0], [2.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
        )
        aligned = _align_result_to_reference(ref, target)
        # Different k => return target unchanged
        np.testing.assert_array_equal(aligned.assignments, target.assignments)

    def test_metadata_indices_permuted(self):
        ref = ClusterResult(
            assignments=np.array([0, 0, 1, 1]),
            centers=np.array([[0.0, 0.0], [5.0, 5.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
            metadata={"indices": np.array([[0.1, 0.0], [0.0, 0.1]])},
        )
        target = ClusterResult(
            assignments=np.array([1, 1, 0, 0]),
            centers=np.array([[5.0, 5.0], [0.0, 0.0]]),
            objective=1.0,
            n_iterations=1,
            converged=True,
            metadata={"indices": np.array([[0.0, 0.1], [0.1, 0.0]])},
        )
        aligned = _align_result_to_reference(ref, target)
        np.testing.assert_allclose(aligned.metadata["indices"], ref.metadata["indices"])
