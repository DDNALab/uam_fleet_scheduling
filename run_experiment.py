"""Run the project directly from a source checkout.

This launcher puts ``src`` on ``sys.path`` so the flat modules there can be
imported without an editable installation first.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from experiment_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
