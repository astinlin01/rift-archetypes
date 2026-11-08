import os
import sys

# src/ is a flat folder of scripts, not a package, so the scripts import each other
# by bare module name (e.g. `from feature_config import ...`). Putting src/ on the
# path lets the tests import them the same way.
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
