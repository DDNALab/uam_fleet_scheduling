"""Run the repository test suite without installing the package."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import pytest  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(PROJECT_ROOT / "tests")]))

