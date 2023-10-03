# Configuration file for the Sphinx documentation builder

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src\\auto_control_tools'))
print(sys.path)

# -- Project information

project = 'auto-control-tools'
copyright = 'Copyright (c) 2023 luizn22'
author = 'Graziella'

release = '0.0.1'
version = '0.0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
