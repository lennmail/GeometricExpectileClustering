"""Sphinx configuration for Geometric Expectile Clustering."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Geometric Expectile Clustering"
copyright = "2026, Lennart Mailänder"  # noqa: A001
author = "Lennart Mailänder"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
