import os
import sys

sys.path.insert(0, os.path.abspath(".."))

source_suffix = [".rst"]

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
]


intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}