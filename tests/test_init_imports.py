"""Tests for public API surface (__init__.py exports)."""

import importlib


def _check_all_importable(module_path):
    mod = importlib.import_module(module_path)
    if hasattr(mod, "__all__"):
        for name in mod.__all__:
            assert hasattr(mod, name), f"{module_path}.{name} listed in __all__ but not importable"


def test_top_level_exports():
    _check_all_importable("geomexp")


def test_clustering_exports():
    _check_all_importable("geomexp.clustering")


def test_evaluation_exports():
    _check_all_importable("geomexp.evaluation")


def test_kernels_exports():
    _check_all_importable("geomexp.kernels")


def test_visualization_exports():
    _check_all_importable("geomexp.visualization")


def test_utils_exports():
    _check_all_importable("geomexp.utils")
