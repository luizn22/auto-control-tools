# Configuration file for the Sphinx documentation builder.
from configparser import ConfigParser

confparser = ConfigParser()
confparser.read('setup.cfg')

# -- Project information

project = confparser['metadata']['name']
copyright = 'Copyright (c) 2023 luizn22'
author = 'Graziella'

release = confparser['metadata']['version']
version = confparser['metadata']['version']

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
