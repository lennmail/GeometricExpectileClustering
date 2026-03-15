"""Tests for geomexp/utils/validation.py."""

import numpy as np
import pytest

from geomexp.utils.validation import (
    validate_data_array,
    validate_gram_matrix,
    validate_index_radius,
    validate_n_clusters,
    validate_positive_float,
    validate_positive_int,
    validate_tolerance,
    validate_weights,
)

# --- validate_data_array ---


class TestValidateDataArray:
    def test_accepts_2d_float(self):
        X = np.random.randn(20, 3)
        result = validate_data_array(X)
        assert result.shape == (20, 3)
        assert result.dtype == np.float64

    def test_accepts_list_of_lists(self):
        result = validate_data_array([[1, 2], [3, 4]])
        assert result.dtype == np.float64
        assert result.shape == (2, 2)

    def test_accepts_integer_array(self):
        result = validate_data_array(np.array([[1, 2], [3, 4]]))
        assert result.dtype == np.float64

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_data_array(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_data_array(np.ones((2, 3, 4)))

    def test_rejects_zero_features(self):
        with pytest.raises(ValueError, match="0 features"):
            validate_data_array(np.empty((10, 0)))

    def test_preserves_values(self):
        X = np.array([[1.5, 2.5], [3.5, 4.5]])
        result = validate_data_array(X)
        np.testing.assert_array_equal(result, X)


# --- validate_n_clusters ---


class TestValidateNClusters:
    def test_accepts_valid(self):
        validate_n_clusters(3, 10)

    def test_accepts_without_n_samples(self):
        validate_n_clusters(5)

    def test_accepts_one(self):
        validate_n_clusters(1)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_n_clusters(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_n_clusters(-1)

    def test_rejects_exceeds_samples(self):
        with pytest.raises(ValueError, match="n_samples=5"):
            validate_n_clusters(10, 5)

    def test_accepts_n_clusters_equals_n_samples(self):
        validate_n_clusters(5, 5)


# --- validate_positive_int ---


class TestValidatePositiveInt:
    def test_accepts_valid(self):
        validate_positive_int(5, "param")

    def test_accepts_one(self):
        validate_positive_int(1, "param")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_positive_int(0, "param")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_positive_int(-3, "param")

    def test_rejects_float(self):
        with pytest.raises(TypeError, match="integer"):
            validate_positive_int(1.5, "param")

    def test_rejects_bool(self):
        # booleans are subclass of int in Python, but True == 1 should pass
        validate_positive_int(True, "param")


# --- validate_tolerance ---


class TestValidateTolerance:
    def test_accepts_zero(self):
        validate_tolerance(0)

    def test_accepts_positive(self):
        validate_tolerance(1e-4)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_tolerance(-0.1)


# --- validate_index_radius ---


class TestValidateIndexRadius:
    def test_accepts_zero(self):
        validate_index_radius(0)

    def test_accepts_mid_range(self):
        validate_index_radius(0.5)

    def test_accepts_near_one(self):
        validate_index_radius(0.999)

    def test_rejects_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            validate_index_radius(1.0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            validate_index_radius(-0.1)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            validate_index_radius(1.5)


# --- validate_positive_float ---


class TestValidatePositiveFloat:
    def test_accepts_valid(self):
        validate_positive_float(0.1, "param")

    def test_accepts_large(self):
        validate_positive_float(1e6, "param")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            validate_positive_float(0, "param")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            validate_positive_float(-1, "param")


# --- validate_weights ---


class TestValidateWeights:
    def test_accepts_valid(self):
        result = validate_weights(np.array([1.0, 2.0, 3.0]))
        assert result.dtype == np.float64

    def test_accepts_list(self):
        result = validate_weights([1, 2, 3])
        assert result.dtype == np.float64

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_weights(np.ones((2, 3)))

    def test_rejects_zero_entry(self):
        with pytest.raises(ValueError, match="positive"):
            validate_weights(np.array([1.0, 0.0, 2.0]))

    def test_rejects_negative_entry(self):
        with pytest.raises(ValueError, match="positive"):
            validate_weights(np.array([1.0, -0.5, 2.0]))


# --- validate_gram_matrix ---


class TestValidateGramMatrix:
    def test_accepts_psd(self):
        G = np.array([[2.0, 0.5], [0.5, 1.0]])
        result = validate_gram_matrix(G)
        assert result.shape == (2, 2)

    def test_accepts_identity(self):
        result = validate_gram_matrix(np.eye(3))
        assert result.shape == (3, 3)

    def test_accepts_zero_matrix(self):
        # All-zeros is PSD (eigenvalues all 0)
        validate_gram_matrix(np.zeros((2, 2)))

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            validate_gram_matrix(np.ones((2, 3)))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="square"):
            validate_gram_matrix(np.array([1.0, 2.0]))

    def test_rejects_indefinite(self):
        # Matrix with eigenvalue -1
        G = np.array([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_gram_matrix(G)

    def test_accepts_list_of_lists(self):
        result = validate_gram_matrix([[2.0, 0.5], [0.5, 1.0]])
        assert result.dtype == np.float64
