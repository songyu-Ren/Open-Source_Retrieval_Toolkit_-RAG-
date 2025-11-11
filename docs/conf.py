import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "Open-Source RAG Toolkit"
author = "Open-Source RAG Toolkit"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"