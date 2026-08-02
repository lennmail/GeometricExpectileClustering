"""Visualization module for clustering algorithms.

Provides plotting utilities for clustering analysis with consistent styling. All visual
configuration flows through :class:`PlotStyle`, which must be passed to
:class:`ClusterVisualizer` to ensure uniform appearance across all figures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from geomexp.clustering.clustering_base import ClusterResult


def _align_result_to_reference(reference: ClusterResult, target: ClusterResult) -> ClusterResult:
    """Align cluster labels of ``target`` to those of ``reference`` by permutation matching.

    Finds the permutation of cluster indices that minimises the sum of Euclidean distances
    between matched center pairs, then relabels ``target`` accordingly. This ensures
    consistent colouring when comparing results side by side.

    Args:
        reference: The result whose labelling is treated as canonical.
        target: The result to relabel.

    Returns:
        A new ``ClusterResult`` with relabelled assignments, centers, and metadata indices.
    """
    n_clusters = len(reference.centers)
    if n_clusters != len(target.centers):
        return target

    best_perm: tuple[int, ...] | None = None
    best_cost = np.inf
    for perm in permutations(range(n_clusters)):
        cost = sum(
            float(np.linalg.norm(reference.centers[i] - target.centers[perm[i]]))
            for i in range(n_clusters)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    inv_perm = [0] * n_clusters
    for new_label, old_label in enumerate(best_perm):
        inv_perm[old_label] = new_label

    new_assignments = np.array([inv_perm[a] for a in target.assignments])
    new_centers = np.empty_like(target.centers)
    for new_label, old_label in enumerate(best_perm):
        new_centers[new_label] = target.centers[old_label]

    new_metadata: dict[str, object] | None = None
    if target.metadata is not None:
        new_metadata = dict(target.metadata)
        if "indices" in new_metadata:
            old_indices = cast(np.ndarray, new_metadata["indices"])
            new_indices = np.empty_like(old_indices)
            for new_label, old_label in enumerate(best_perm):
                new_indices[new_label] = old_indices[old_label]
            new_metadata["indices"] = new_indices

    return ClusterResult(
        assignments=new_assignments,
        centers=new_centers,
        objective=target.objective,
        n_iterations=target.n_iterations,
        converged=target.converged,
        metadata=new_metadata,
    )


@dataclass
class PlotStyle:
    """Central styling configuration for all cluster visualisations.

    Every plot produced by :class:`ClusterVisualizer` reads its visual parameters from this object,
    ensuring that all figures in a study share a uniform appearance. Construct a ``PlotStyle``
    instance first, then pass it to ``ClusterVisualizer(style)``.

    Attributes:
        figsize: Default figure size ``(width, height)`` in inches.
        dpi: Resolution in dots per inch.
        point_size: Scatter point size for data points.
        center_size: Scatter point size for cluster centers.
        center_linewidth: Edge line width for center markers.
        alpha_data: Opacity of data points.
        alpha_contour: Opacity of coloured decision regions.
        color_palette: List of hex colour strings cycled over clusters.
        center_color: Colour of center markers.
        center_marker: Marker shape for centers.
        arrow_color: Colour of index vector arrows.
        arrow_linewidth: Line width of index vector arrows.
        arrow_head_width: Head width of index vector arrows.
        arrow_head_length: Head length of index vector arrows.
        arrow_scale: Multiplicative scaling applied to index vector arrows for visibility.
        contour_color: Colour of decision boundary lines.
        contour_linewidth: Line width of decision boundary contours.
        contour_alpha: Opacity of decision boundary contour lines.
        curve_linewidth: Line width of expectile level-set contours.
        curve_alpha: Opacity of expectile level-set contour lines.
        kde_color: Colour of KDE density contour lines.
        kde_linewidth: Line width of KDE density contour lines.
        kde_alpha: Opacity of KDE density contour lines.
        loss_linewidth: Line width of loss function curves.
        decision_boundary_resolution: Grid resolution for boundary computation.
        decision_boundary_show_points: Whether to overlay data points on boundary plots.
        decision_boundary_show_contour_lines: Whether to draw boundary contour lines.
        decision_boundary_show_centers: Whether to show center markers on boundary plots.
        decision_boundary_show_regions: Whether to fill decision regions with colour.
        decision_boundary_color: Override colour for boundary lines (``None`` uses palette).
        axis_linewidth: Width of axis spines and tick marks.
        grid_alpha: Opacity of grid lines.
        grid_linewidth: Width of grid lines.
        fontsize: Base font size for labels and titles.
        use_latex: Whether to enable LaTeX rendering via ``text.usetex``.
    """

    figsize: tuple[float, float] = (3.5, 3.5)
    dpi: int = 300
    point_size: float = 1.5
    center_size: float = 15
    center_linewidth: float = 0.8
    alpha_data: float = 0.7
    alpha_contour: float = 0.15
    color_palette: list[str] | None = None
    center_color: str = "black"
    center_marker: str = "o"
    arrow_color: str = "black"
    arrow_linewidth: float = 0.8
    arrow_head_width: float = 0.15
    arrow_head_length: float = 0.2
    arrow_scale: float = 1
    contour_color: str = "black"
    contour_linewidth: float = 0.5
    contour_alpha: float = 0.9
    curve_linewidth: float = 0.8
    curve_alpha: float = 0.7
    kde_color: str = "gray"
    kde_linewidth: float = 0.4
    kde_alpha: float = 0.6
    loss_linewidth: float = 1
    decision_boundary_resolution: int = 300
    decision_boundary_show_points: bool = True
    decision_boundary_show_contour_lines: bool = True
    decision_boundary_show_centers: bool = True
    decision_boundary_show_regions: bool = True
    decision_boundary_color: str | None = None
    axis_linewidth: float = 0.6
    grid_alpha: float = 0.15
    grid_linewidth: float = 0.3
    fontsize: float = 9
    use_latex: bool = True

    def __post_init__(self) -> None:
        if self.color_palette is None:
            self.color_palette = ["#FF5733", "#C70039", "#900C3F", "#581845", "#FFC300"]


class ClusterVisualizer:
    """Plotting engine for cluster analysis visualisations.

    All plotting methods read visual configuration from the ``style`` attribute.
    A :class:`PlotStyle` instance **must** be passed at construction time; there is no default.

    Args:
        style: A :class:`PlotStyle` instance that governs all visual parameters.

    Raises:
        TypeError: If ``style`` is not a :class:`PlotStyle`.

    Example:
        >>> style = PlotStyle(figsize=(5, 4), use_latex=False)
        >>> viz = ClusterVisualizer(style)
    """

    def __init__(self, style: PlotStyle) -> None:
        if not isinstance(style, PlotStyle):
            raise TypeError(
                f"ClusterVisualizer requires a PlotStyle instance, got {type(style).__name__}"
            )
        self.style = style
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Apply ``self.style`` settings to matplotlib rcParams."""
        fs = self.style.fontsize
        font_params: dict[str, object] = {
            "axes.labelsize": fs,
            "font.size": fs,
            "legend.fontsize": fs - 1,
            "xtick.labelsize": fs - 1,
            "ytick.labelsize": fs - 1,
            "axes.titlesize": fs,
        }
        if self.style.use_latex:
            font_params.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{fourier}",
                }
            )
        plt.rcParams.update(font_params)

        plt.rcParams.update(
            {
                "figure.dpi": self.style.dpi,
                "savefig.dpi": self.style.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "axes.linewidth": self.style.axis_linewidth,
                "axes.edgecolor": "black",
                "xtick.major.width": self.style.axis_linewidth,
                "ytick.major.width": self.style.axis_linewidth,
                "xtick.minor.width": self.style.axis_linewidth * 0.7,
                "ytick.minor.width": self.style.axis_linewidth * 0.7,
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.linewidth": self.style.grid_linewidth,
                "legend.frameon": True,
                "legend.edgecolor": "black",
                "legend.framealpha": 1,
                "legend.fancybox": False,
            }
        )

    def _compute_costs(self, grid_points: np.ndarray, result: ClusterResult) -> np.ndarray:
        """Compute per-cluster costs for every grid point.

        Args:
            grid_points: Array of shape ``(n_grid, 2)``.
            result: Clustering result with centers and optional index metadata.

        Returns:
            Cost array of shape ``(n_grid, n_clusters)``.
        """
        n_clusters = len(result.centers)
        costs = np.zeros((len(grid_points), n_clusters))

        if result.metadata is not None and "indices" in result.metadata:
            indices = cast(np.ndarray, result.metadata["indices"])
            for k in range(n_clusters):
                residuals = grid_points - result.centers[k]
                dists = np.linalg.norm(residuals, axis=1)
                costs[:, k] = 0.5 * dists**2 + 0.5 * dists * np.sum(indices[k] * residuals, axis=1)
        else:
            for k in range(n_clusters):
                costs[:, k] = np.sum((grid_points - result.centers[k]) ** 2, axis=1)

        return costs

    def _draw_boundary_lines(
        self,
        ax: Axes,
        xx: np.ndarray,
        yy: np.ndarray,
        costs: np.ndarray,
        n_clusters: int,
        linewidth: float,
        color: str | None = None,
    ) -> None:
        """Draw decision boundary lines via pairwise cost-difference contours.

        For each pair of clusters that share an edge, a zero-level contour of their cost difference
        is drawn, clipped to the region where one of the two clusters is closest.

        Args:
            ax: Matplotlib axes to draw on.
            xx: Meshgrid x-coordinates.
            yy: Meshgrid y-coordinates.
            costs: Flattened cost array of shape ``(n_grid, n_clusters)``.
            n_clusters: Number of clusters.
            linewidth: Line width for boundary contours.
            color: Override boundary colour (``None`` uses ``self.style.contour_color``).
        """
        line_color = self.style.contour_color if color is None else color
        assignments = np.argmin(costs, axis=1).reshape(xx.shape)

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                mask_i = assignments == i
                mask_j = assignments == j
                if not (mask_i.any() and mask_j.any()):
                    continue

                diff = (costs[:, i] - costs[:, j]).reshape(xx.shape)
                masked_diff = np.where(mask_i | mask_j, diff, np.nan)
                ax.contour(
                    xx,
                    yy,
                    masked_diff,
                    levels=[0],
                    colors=line_color,
                    linewidths=linewidth,
                    alpha=self.style.contour_alpha,
                )

    def plot_cluster_assignments(
        self,
        X: np.ndarray,
        result: ClusterResult,
        ax: Axes | None = None,
        show_centers: bool = True,
        show_indices: bool = True,
        show_legend: bool = False,
        title: str | None = None,
    ) -> Figure:
        """Plot data points coloured by cluster assignment.

        Args:
            X: Data array of shape ``(n_samples, 2)``.
            result: Clustering result.
            ax: Matplotlib axes (created if ``None``).
            show_centers: Whether to overlay center markers.
            show_indices: Whether to draw index vector arrows.
            show_legend: Whether to show a cluster legend.
            title: Optional axes title.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if X.shape[1] != 2:
            raise ValueError("Only 2D data can be visualized")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        assert self.style.color_palette is not None
        for k in range(len(result.centers)):
            mask = result.assignments == k
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=self.style.color_palette[k % len(self.style.color_palette)],
                s=self.style.point_size,
                alpha=self.style.alpha_data,
                edgecolors="none",
                label=f"{k}" if show_legend else None,
            )

        if show_indices and result.metadata is not None and "indices" in result.metadata:
            indices = cast(np.ndarray, result.metadata["indices"])
            for k in range(len(result.centers)):
                u_k = indices[k] * self.style.arrow_scale
                if np.linalg.norm(u_k) > 1e-10:
                    c = result.centers[k]
                    ax.annotate(
                        "",
                        xy=(c[0] + u_k[0], c[1] + u_k[1]),
                        xytext=(c[0], c[1]),
                        arrowprops={
                            "arrowstyle": (
                                f"->,head_width={self.style.arrow_head_width}"
                                f",head_length={self.style.arrow_head_length}"
                            ),
                            "lw": self.style.arrow_linewidth,
                            "color": self.style.arrow_color,
                            "shrinkA": 0,
                            "shrinkB": 0,
                        },
                        zorder=8,
                    )

        if show_centers:
            ax.scatter(
                result.centers[:, 0],
                result.centers[:, 1],
                facecolors="none",
                edgecolors=self.style.center_color,
                s=self.style.center_size,
                marker=self.style.center_marker,
                linewidths=self.style.center_linewidth,
                zorder=10,
            )

        self._style_axes(ax, X, title, show_legend)
        return fig

    def plot_decision_boundaries(
        self,
        X: np.ndarray,
        result: ClusterResult,
        ax: Axes | None = None,
        title: str | None = None,
        resolution: int | None = None,
        show_points: bool | None = None,
        show_contour_lines: bool | None = None,
        show_centers: bool | None = None,
        show_regions: bool | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        boundary_color: str | None = None,
    ) -> Figure:
        """Plot decision boundaries (and optionally regions, data points, centers).

        Args:
            X: Data array of shape ``(n_samples, 2)``.
            result: Clustering result.
            ax: Matplotlib axes (created if ``None``).
            title: Optional axes title.
            resolution: Grid resolution for boundary computation. Defaults to
                ``self.style.decision_boundary_resolution``.
            show_points: Overlay data points. Defaults to style setting.
            show_contour_lines: Draw boundary contour lines. Defaults to style setting.
            show_centers: Show center markers. Defaults to style setting.
            show_regions: Fill decision regions with colour. Defaults to style setting.
            x_lim: Manual x-axis limits.
            y_lim: Manual y-axis limits.
            boundary_color: Override colour for boundary lines.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if X.shape[1] != 2:
            raise ValueError("Only 2D data can be visualized")

        s = self.style
        resolution = resolution or s.decision_boundary_resolution
        show_points = show_points if show_points is not None else s.decision_boundary_show_points
        show_contour_lines = (
            show_contour_lines
            if show_contour_lines is not None
            else s.decision_boundary_show_contour_lines
        )
        show_centers = (
            show_centers if show_centers is not None else s.decision_boundary_show_centers
        )
        show_regions = (
            show_regions if show_regions is not None else s.decision_boundary_show_regions
        )

        if boundary_color is None:
            boundary_color = s.decision_boundary_color
        if boundary_color is None:
            assert s.color_palette is not None
            boundary_color = s.color_palette[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=s.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        if x_lim is not None:
            x_min, x_max = x_lim
        else:
            x_margin = (X[:, 0].max() - X[:, 0].min()) * 0.1
            x_min, x_max = X[:, 0].min() - x_margin, X[:, 0].max() + x_margin

        if y_lim is not None:
            y_min, y_max = y_lim
        else:
            y_margin = (X[:, 1].max() - X[:, 1].min()) * 0.1
            y_min, y_max = X[:, 1].min() - y_margin, X[:, 1].max() + y_margin

        pad_x, pad_y = (x_max - x_min) * 0.02, (y_max - y_min) * 0.02
        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, resolution),
            np.linspace(y_min - pad_y, y_max + pad_y, resolution),
        )

        n_clusters = len(result.centers)
        costs = self._compute_costs(np.c_[xx.ravel(), yy.ravel()], result)

        assert s.color_palette is not None
        colors = [s.color_palette[k % len(s.color_palette)] for k in range(n_clusters)]

        if show_regions:
            Z = np.argmin(costs, axis=1).reshape(xx.shape)
            ax.pcolormesh(
                xx,
                yy,
                Z,
                cmap=ListedColormap(colors),
                norm=BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters),
                alpha=s.alpha_contour,
                shading="auto",
                zorder=0,
            )

        if show_contour_lines:
            self._draw_boundary_lines(
                ax,
                xx,
                yy,
                costs,
                n_clusters,
                linewidth=s.contour_linewidth,
                color=boundary_color,
            )

        if show_points:
            for k in range(n_clusters):
                mask = result.assignments == k
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=colors[k],
                    s=s.point_size,
                    alpha=s.alpha_data,
                    edgecolors="none",
                )

        if show_centers:
            ax.scatter(
                result.centers[:, 0],
                result.centers[:, 1],
                facecolors="none",
                edgecolors=s.center_color,
                s=s.center_size,
                marker=s.center_marker,
                linewidths=s.center_linewidth,
                zorder=10,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=s.grid_alpha, linewidth=s.grid_linewidth)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if title:
            ax.set_title(title, fontsize=s.fontsize)

        return fig

    def plot_density_contours(
        self,
        X: np.ndarray,
        ax: Axes | None = None,
        title: str | None = None,
        levels: int = 20,
        bandwidth: float = 0.2,
    ) -> Figure:
        """Plot KDE density contours of the data.

        Args:
            X: Data array of shape ``(n_samples, 2)``.
            ax: Matplotlib axes (created if ``None``).
            title: Optional axes title.
            levels: Number of contour levels.
            bandwidth: KDE bandwidth parameter.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if X.shape[1] != 2:
            raise ValueError("Only 2D data can be visualized")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        kde = gaussian_kde(X.T, bw_method=bandwidth)
        x_margin = (X[:, 0].max() - X[:, 0].min()) * 0.1
        y_margin = (X[:, 1].max() - X[:, 1].min()) * 0.1
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - x_margin, X[:, 0].max() + x_margin, 150),
            np.linspace(X[:, 1].min() - y_margin, X[:, 1].max() + y_margin, 150),
        )
        ax.contour(
            xx,
            yy,
            kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape),
            levels=levels,
            colors=self.style.kde_color,
            linewidths=self.style.kde_linewidth,
            alpha=self.style.kde_alpha,
        )

        self._style_axes(ax, X, title, show_legend=False)
        return fig

    def plot_density_with_boundaries(
        self,
        X: np.ndarray,
        result: ClusterResult,
        ax: Axes | None = None,
        title: str | None = None,
        resolution: int = 300,
        kde_levels: int = 20,
        kde_bandwidth: float = 0.2,
        show_centers: bool = True,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
    ) -> Figure:
        """Overlay KDE density contours with decision boundary lines.

        Args:
            X: Data array of shape ``(n_samples, 2)``.
            result: Clustering result.
            ax: Matplotlib axes (created if ``None``).
            title: Optional axes title.
            resolution: Grid resolution for boundary computation.
            kde_levels: Number of KDE contour levels.
            kde_bandwidth: KDE bandwidth parameter.
            show_centers: Whether to overlay center markers.
            x_lim: Manual x-axis limits.
            y_lim: Manual y-axis limits.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if X.shape[1] != 2:
            raise ValueError("Only 2D data can be visualized")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        if x_lim is not None:
            x_min, x_max = x_lim
        else:
            x_margin = (X[:, 0].max() - X[:, 0].min()) * 0.1
            x_min, x_max = X[:, 0].min() - x_margin, X[:, 0].max() + x_margin

        if y_lim is not None:
            y_min, y_max = y_lim
        else:
            y_margin = (X[:, 1].max() - X[:, 1].min()) * 0.1
            y_min, y_max = X[:, 1].min() - y_margin, X[:, 1].max() + y_margin

        pad_x, pad_y = (x_max - x_min) * 0.02, (y_max - y_min) * 0.02
        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, resolution),
            np.linspace(y_min - pad_y, y_max + pad_y, resolution),
        )

        kde = gaussian_kde(X.T, bw_method=kde_bandwidth)
        ax.contour(
            xx,
            yy,
            kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape),
            levels=kde_levels,
            colors=self.style.kde_color,
            linewidths=self.style.kde_linewidth,
            alpha=self.style.kde_alpha,
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        costs = self._compute_costs(grid_points, result)
        self._draw_boundary_lines(
            ax,
            xx,
            yy,
            costs,
            len(result.centers),
            linewidth=self.style.contour_linewidth,
        )

        if show_centers:
            ax.scatter(
                result.centers[:, 0],
                result.centers[:, 1],
                facecolors="none",
                edgecolors=self.style.center_color,
                s=self.style.center_size,
                marker=self.style.center_marker,
                linewidths=self.style.center_linewidth,
                zorder=10,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=self.style.grid_alpha, linewidth=self.style.grid_linewidth)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if title:
            ax.set_title(title, fontsize=self.style.fontsize)

        return fig

    def plot_expectile_curves(
        self,
        center: np.ndarray,
        index_vector: np.ndarray,
        cost_levels: np.ndarray | None = None,
        ax: Axes | None = None,
        title: str | None = None,
        show_index_arrow: bool = True,
        curve_color: str | None = None,
        extent: float = 3,
        resolution: int = 300,
    ) -> Figure:
        """Plot level curves of the expectile loss around a single center.

        Args:
            center: Center point of shape ``(2,)``.
            index_vector: Index vector of shape ``(2,)``.
            cost_levels: Loss values at which to draw contours.
            ax: Matplotlib axes (created if ``None``).
            title: Optional axes title.
            show_index_arrow: Whether to draw the index vector arrow.
            curve_color: Override colour for contour lines.
            extent: Half-width of the plotting window around ``center``.
            resolution: Grid resolution.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        if curve_color is None:
            assert self.style.color_palette is not None
            curve_color = self.style.color_palette[0]
        if cost_levels is None:
            cost_levels = np.linspace(0.5, 3, 8)

        xx, yy = np.meshgrid(
            np.linspace(center[0] - extent, center[0] + extent, resolution),
            np.linspace(center[1] - extent, center[1] + extent, resolution),
        )
        rx, ry = xx - center[0], yy - center[1]
        norms = np.sqrt(rx**2 + ry**2)
        Z = 0.5 * (norms**2 + norms * (index_vector[0] * rx + index_vector[1] * ry))

        ax.contour(
            xx,
            yy,
            Z,
            levels=cost_levels,
            colors=curve_color,
            linewidths=self.style.curve_linewidth,
            alpha=self.style.curve_alpha,
        )
        ax.scatter(
            center[0],
            center[1],
            facecolors="none",
            edgecolors=self.style.center_color,
            s=self.style.center_size,
            marker=self.style.center_marker,
            zorder=10,
            linewidths=self.style.center_linewidth,
        )

        if show_index_arrow and np.linalg.norm(index_vector) > 1e-10:
            ax.annotate(
                "",
                xy=(center[0] + index_vector[0], center[1] + index_vector[1]),
                xytext=(center[0], center[1]),
                arrowprops={
                    "arrowstyle": (
                        f"->,head_width={self.style.arrow_head_width}"
                        f",head_length={self.style.arrow_head_length}"
                    ),
                    "lw": self.style.arrow_linewidth,
                    "color": self.style.arrow_color,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=8,
            )

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_aspect("equal")
        ax.grid(True, alpha=self.style.grid_alpha, linewidth=self.style.grid_linewidth)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if title:
            ax.set_title(title, fontsize=self.style.fontsize)

        return fig

    def plot_loss_comparison(
        self,
        loss_functions: list[tuple[str, Callable[[np.ndarray], np.ndarray]]],
        x_range: tuple[float, float] = (-3, 3),
        n_points: int = 300,
        ax: Axes | None = None,
        title: str | None = None,
        show_legend: bool = True,
    ) -> Figure:
        """Plot one-dimensional loss function comparisons.

        Args:
            loss_functions: List of ``(name, callable)`` pairs.
            x_range: Domain ``(x_min, x_max)`` for the plot.
            n_points: Number of evaluation points.
            ax: Matplotlib axes (created if ``None``).
            title: Optional axes title.
            show_legend: Whether to display a legend.

        Returns:
            The matplotlib ``Figure`` containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = cast(Figure, ax.get_figure())
            assert fig is not None

        x = np.linspace(x_range[0], x_range[1], n_points)
        assert self.style.color_palette is not None
        for idx, (name, loss_fn) in enumerate(loss_functions):
            ax.plot(
                x,
                loss_fn(x),
                label=name,
                color=self.style.color_palette[idx % len(self.style.color_palette)],
                linewidth=self.style.loss_linewidth,
            )

        ax.axhline(y=0, color="black", linewidth=0.4, linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.4, linestyle="--", alpha=0.3)
        ax.grid(True, alpha=self.style.grid_alpha, linewidth=self.style.grid_linewidth)
        ax.set_xlabel(r"Distance $d$")
        ax.set_ylabel(r"Loss $L(d)$")

        if show_legend:
            legend = ax.legend(frameon=True, loc="best", edgecolor="black", fancybox=False)
            legend.get_frame().set_linewidth(self.style.axis_linewidth)

        if title:
            ax.set_title(title, fontsize=self.style.fontsize)

        return fig

    def create_comparison_figure(
        self,
        X: np.ndarray,
        results: list[tuple[str, ClusterResult]],
        plot_type: str = "assignments",
        nrows: int | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        align_labels: bool = True,
        **plot_kwargs: Any,
    ) -> Figure:
        """Create a multi-panel comparison figure.

        Args:
            X: Data array of shape ``(n_samples, 2)``.
            results: List of ``(name, ClusterResult)`` pairs for each panel.
            plot_type: One of ``"assignments"``, ``"boundaries"``, or ``"density"``.
            nrows: Number of subplot rows (inferred if ``None``).
            ncols: Number of subplot columns (inferred if ``None``).
            figsize: Override figure size.
            align_labels: Whether to align cluster labels across panels via permutation matching.
            **plot_kwargs: Extra keyword arguments forwarded to the per-panel plot method.

        Returns:
            The matplotlib ``Figure`` containing the comparison.
        """
        if align_labels and len(results) > 1 and plot_type != "density":
            ref = results[0][1]
            results = [results[0]] + [
                (n, _align_result_to_reference(ref, r)) for n, r in results[1:]
            ]

        n_plots = len(results)

        if nrows is None and ncols is None:
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
        elif nrows is None:
            assert ncols is not None
            nrows = (n_plots + ncols - 1) // ncols
        elif ncols is None:
            ncols = (n_plots + nrows - 1) // nrows

        assert nrows is not None and ncols is not None
        if figsize is None:
            figsize = (self.style.figsize[0] * ncols, self.style.figsize[0] * nrows)

        share = plot_type != "boundaries"
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=share, sharey=share)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, result) in enumerate(results):
            if plot_type == "density":
                self.plot_density_contours(X, ax=axes[idx], title=name, **plot_kwargs)
            elif plot_type == "assignments":
                self.plot_cluster_assignments(X, result, ax=axes[idx], title=name, **plot_kwargs)
            elif plot_type == "boundaries":
                self.plot_decision_boundaries(X, result, ax=axes[idx], title=name, **plot_kwargs)

        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        for idx in range(min(n_plots, nrows * ncols)):
            row, col = divmod(idx, ncols)
            if col > 0:
                axes[idx].set_ylabel("")
            if row < nrows - 1 and idx + ncols < n_plots:
                axes[idx].set_xlabel("")

        plt.tight_layout()
        return fig

    def _style_axes(
        self, ax: Axes, X: np.ndarray, title: str | None = None, show_legend: bool = False
    ) -> None:
        """Apply common axis styling: equal aspect, grid, labels, limits."""
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        max_range = max(x_max - x_min, y_max - y_min)
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        margin = 0.15 * max_range

        ax.set_xlim(x_center - max_range / 2 - margin, x_center + max_range / 2 + margin)
        ax.set_ylim(y_center - max_range / 2 - margin, y_center + max_range / 2 + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=self.style.grid_alpha, linewidth=self.style.grid_linewidth)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        if title:
            ax.set_title(title, fontsize=self.style.fontsize)
        if show_legend:
            legend = ax.legend(
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                handletextpad=0.3,
                columnspacing=0.5,
                edgecolor="black",
                fancybox=False,
            )
            for text in legend.get_texts():
                text.set_fontsize(self.style.fontsize - 2)
            legend.get_frame().set_linewidth(self.style.axis_linewidth)

    @staticmethod
    def save_figure(
        fig: Figure, filename: str, dpi: int | None = None, transparent: bool = False
    ) -> None:
        """Save a figure to disk.

        Args:
            fig: The figure to save.
            filename: Output file path (extension determines format).
            dpi: Override resolution (``None`` uses the figure's default).
            transparent: Whether to use a transparent background.
        """
        fig.savefig(
            filename, dpi=dpi, bbox_inches="tight", transparent=transparent, pad_inches=0.05
        )
