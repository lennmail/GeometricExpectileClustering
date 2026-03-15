"""Hilbert space geometry for clustering algorithms.

Provides abstractions for inner products and norms so that the same clustering algorithms work
unchanged in Euclidean space, weighted :math:`L^2` (functional data on grids), or general
Gram-matrix spaces (basis representations).

The three concrete geometries cover the common cases:

* :class:`EuclideanGeometry` -- standard dot product (identity weights).
* :class:`WeightedEuclideanGeometry` -- diagonal quadrature weights, approximating
  :math:`\\langle f, g \\rangle_{L^2} = \\int f(t)\\,g(t)\\,\\mathrm{d}t`.
* :class:`GramGeometry` -- full Gram matrix :math:`G_{ij} = \\langle \\varphi_i, \\varphi_j
  \\rangle` for basis coefficient representations.
"""

from abc import ABC, abstractmethod

import numpy as np

from geomexp.utils.validation import validate_gram_matrix, validate_weights


class HilbertGeometry(ABC):
    """Abstract Hilbert space geometry defining inner product and norm.

    All clustering geometry (assignment costs, center gradients, index strategies) flows through
    this interface, ensuring that no part of the algorithm accidentally falls back to Euclidean
    operations.
    """

    @abstractmethod
    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the inner product :math:`\\langle a, b \\rangle_H` along the last axis.

        Args:
            a: Array of shape ``(..., d)``.
            b: Array of shape ``(..., d)``, broadcastable with ``a``.

        Returns:
            Inner product values of shape ``(...)``.
        """

    @abstractmethod
    def norm(self, a: np.ndarray) -> np.ndarray:
        """Compute :math:`\\|a\\|_H` along the last axis.

        Args:
            a: Array of shape ``(..., d)``.

        Returns:
            Norm values of shape ``(...)``.
        """

    def squared_norm(self, a: np.ndarray) -> np.ndarray:
        """Compute :math:`\\|a\\|_H^2` along the last axis.

        Default implementation returns ``inner(a, a)``. Override for efficiency if the inner
        product is expensive.
        """
        return self.inner(a, a)


class EuclideanGeometry(HilbertGeometry):
    """Standard Euclidean geometry: :math:`\\langle a, b \\rangle = \\sum_i a_i b_i`.

    Equivalent to identity weights. This is the default geometry when none is specified.
    """

    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(a * b, axis=-1))

    def norm(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(np.linalg.norm(a, axis=-1))

    def squared_norm(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(a**2, axis=-1))


class WeightedEuclideanGeometry(HilbertGeometry):
    r"""Diagonal-weighted geometry: :math:`\langle a, b \rangle_w = \sum_i w_i\, a_i\, b_i`.

    Suitable for functional data discretised on a grid with quadrature weights :math:`w_i`,
    approximating

    .. math::
        \langle f, g \rangle_{L^2} = \int f(t)\, g(t)\, \mathrm{d}t
        \;\approx\; \sum_i w_i\, f(t_i)\, g(t_i).

    Args:
        weights: Positive quadrature weight array of shape ``(d,)``.
    """

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = validate_weights(weights)

    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(self.weights * a * b, axis=-1))

    def norm(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(np.sqrt(np.maximum(np.sum(self.weights * a**2, axis=-1), 0)))

    def squared_norm(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(self.weights * a**2, axis=-1))


class GramGeometry(HilbertGeometry):
    r"""General Gram-matrix geometry: :math:`\langle a, b \rangle_G = a^\top G\, b`.

    Suitable for basis coefficient representations where
    :math:`G_{ij} = \langle \varphi_i, \varphi_j \rangle`.

    Args:
        gram_matrix: Symmetric positive-definite matrix of shape ``(d, d)``.
    """

    def __init__(self, gram_matrix: np.ndarray) -> None:
        self.G = validate_gram_matrix(gram_matrix)

    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(a * np.einsum("ij,...j->...i", self.G, b), axis=-1))

    def norm(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(np.sqrt(np.maximum(self.inner(a, a), 0)))

    def squared_norm(self, a: np.ndarray) -> np.ndarray:
        return self.inner(a, a)
