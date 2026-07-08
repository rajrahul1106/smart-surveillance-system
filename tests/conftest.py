"""
tests/conftest.py

Adds the project root to sys.path so ``from core.xxx import ...`` works
via the ``core`` symlink that points to the ' core' directory on disk.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
