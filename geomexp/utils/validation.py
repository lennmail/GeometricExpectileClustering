"""Input validation utilities for the geomexp package.

Centralised validation functions used throughout the library to enforce parameter
constraints and provide consistent, informative error messages.
"""

from __future__ import annotations

import numpy as np


def validate_data_array(X: object) -> np.ndarray:
    """Validate and coerce input data to a 2-D float64 array.

    Args:
        X: Input data, coerced via :func:`numpy.asarray`.

    Returns:
        Validated array of shape ``(n_samples, n_features)`` with dtype ``float64``.

    Raises:
        ValueError: If the resulting array is not 2-D or has zero features.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {arr.ndim}D array")
    if arr.shape[1] == 0:
        raise ValueError("X has 0 features")
    return arr


def validate_n_clusters(n_clusters: int, n_samples: int | None = None) -> None:
    """Validate the number of clusters.

    Args:
        n_clusters: Requested number of clusters.
        n_samples: If provided, also checks that ``n_clusters <= n_samples``.

    Raises:
        ValueError: If ``n_clusters < 1`` or exceeds ``n_samples``.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_samples is not None and n_samples < n_clusters:
        raise ValueError(f"n_samples={n_samples} must be >= n_clusters={n_clusters}")


def validate_positive_int(value: int, name: str) -> None:
    """Validate that a parameter is a positive integer.

    Args:
        value: The value to check.
        name: Parameter name for the error message.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 1``.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def validate_tolerance(tol: float) -> None:
    """Validate a convergence tolerance (must be non-negative).

    Args:
        tol: Tolerance value.

    Raises:
        ValueError: If ``tol < 0``.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative")


def validate_index_radius(r: float) -> None:
    """Validate the index radius constraint :math:`r \\in [0, 1)`.

    Args:
        r: Index radius value.

    Raises:
        ValueError: If ``r`` is outside :math:`[0, 1)`.
    """
    if not 0 <= r < 1:
        raise ValueError("Index radius must be in [0, 1)")


def validate_positive_float(value: float, name: str) -> None:
    """Validate that a parameter is strictly positive.

    Args:
        value: The value to check.
        name: Parameter name for the error message.

    Raises:
        ValueError: If ``value <= 0``.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_weights(weights: object) -> np.ndarray:
    """Validate a 1-D array of positive quadrature weights.

    Args:
        weights: Weight array, coerced via :func:`numpy.asarray`.

    Returns:
        Validated 1-D ``float64`` array.

    Raises:
        ValueError: If the array is not 1-D or contains non-positive entries.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"Weights must be 1-D, got {w.ndim}-D")
    if np.any(w <= 0):
        raise ValueError("All weights must be positive")
    return w


def validate_gram_matrix(G: object) -> np.ndarray:
    """Validate a Gram matrix (must be square, 2-D, symmetric, and positive semi-definite).

    Checks PSD by verifying that all eigenvalues are :math:`\\geq -\\epsilon` where
    :math:`\\epsilon` is a small tolerance scaled by the matrix norm. Symmetry is checked first:
    :func:`numpy.linalg.eigvalsh` reads only the lower triangle, so without it an asymmetric
    matrix would pass the PSD check and then define a non-symmetric inner product.

    Args:
        G: Gram matrix, coerced via :func:`numpy.asarray`.

    Returns:
        Validated 2-D ``float64`` array.

    Raises:
        ValueError: If the matrix is not square, not 2-D, not symmetric, or not positive
            semi-definite.
    """
    arr = np.asarray(G, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Gram Matrix must be square, got shape {arr.shape}")
    if not np.allclose(arr, arr.T):
        raise ValueError("Gram Matrix must be symmetric")
    eigvals = np.linalg.eigvalsh(arr)
    if float(eigvals.min()) < -1e-8 * max(float(np.abs(eigvals).max()), 1):
        raise ValueError("Gram Matrix must be positive semi-definite")
    return arr
