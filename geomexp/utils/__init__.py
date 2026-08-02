"""Utility functions for the geomexp package."""

from geomexp.utils.sampling import draw_distinct_indices
from geomexp.utils.validation import (
    validate_data_array,
    validate_direction_matrix,
    validate_direction_vector,
    validate_gram_matrix,
    validate_index_radius,
    validate_n_clusters,
    validate_positive_float,
    validate_positive_int,
    validate_tolerance,
    validate_weights,
)

__all__ = [
    "draw_distinct_indices",
    "validate_data_array",
    "validate_direction_matrix",
    "validate_direction_vector",
    "validate_gram_matrix",
    "validate_index_radius",
    "validate_n_clusters",
    "validate_positive_float",
    "validate_positive_int",
    "validate_tolerance",
    "validate_weights",
]
