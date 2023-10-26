# Configuration file for the Sphinx documentation builder

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src'))
print(sys.path)

# -- Project information

project = 'auto-control-tools'
copyright = 'Copyright (c) 2023 luizn22'
author = 'Auto Control Tools Developers'

release = '0.0.1'
version = '0.0.1'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',

    'sphinx.ext.duration',
    'sphinx.ext.autosummary'
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
