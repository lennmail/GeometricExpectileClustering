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

# sphinx-apidoc emits modules.rst as an alternative root; index.rst points at api/geomexp
# directly, so building it would only warn about a document outside every toctree.
exclude_patterns: list[str] = ["api/modules.rst"]

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

# Every package __init__ re-exports its subpackages' public names via __all__. Without this,
# autodoc documents each class once where it is defined and again under every package that
# re-exports it, producing duplicate object descriptions and ambiguous cross-references.
autodoc_default_options = {"ignore-module-all": True}

# Render "Attributes:" sections as :ivar: fields rather than standalone attribute directives,
# which would otherwise collide with the dataclass fields autodoc documents in their own right.
napoleon_use_ivar = True
