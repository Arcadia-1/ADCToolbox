# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

from adctoolbox import __version__ as adctoolbox_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ADCToolbox'
copyright = '2026, ADCToolbox Contributors'
author = 'ADCToolbox Contributors'

version = adctoolbox_version
release = adctoolbox_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate API docs
    'sphinx.ext.napoleon',          # Google/NumPy docstring support
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other docs (numpy, scipy)
    'sphinx.ext.mathjax',           # Math equations
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.todo',              # TODO support
    'myst_parser',                  # Markdown support
    'sphinx_copybutton',            # Copy buttons for code blocks
    'sphinx_design',                # Cards, grids, and landing page blocks
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

locale_dirs = ['locale']      # where .po files live (relative to source/)
gettext_compact = False        # one .po file per source file (easier to manage)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/adctoolbox-logo.svg'
html_favicon = '_static/adctoolbox-logo.svg'
html_static_path = ['_static']

html_css_files = [
    'adctoolbox.css',
]

html_sidebars = {
    'index': [],
}

html_theme_options = {
    'github_url': 'https://github.com/Arcadia-1/ADCToolbox',
    'show_toc_level': 2,
    'navbar_align': 'left',
    'navigation_with_keys': True,
    'logo': {
        'text': 'ADCToolbox',
    },
}

# -- Napoleon settings (for Google-style docstrings) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Autosummary
autosummary_generate = True

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- MyST parser (for markdown) ----------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True


def setup(app):
    """Expose build language to templates for cross-language navigation."""

    def add_template_context(app, pagename, templatename, context, doctree):
        context['adctoolbox_language'] = app.config.language

    app.connect('html-page-context', add_template_context)
