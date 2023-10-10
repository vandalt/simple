import simple

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

project = "simple"
copyright = "2023 Thomas Vandal"
version = simple.__version__
release = simple.__version__

html_theme = "sphinx_book_theme"
html_title = "simple"
exclude_patterns = ["_build"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/vandalt/simple",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "auto"
nb_execution_excludepatterns = []
nb_execution_timeout = -1
