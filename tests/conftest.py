"""Make the tests/ directory importable so ``from fakes import ...`` resolves."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
