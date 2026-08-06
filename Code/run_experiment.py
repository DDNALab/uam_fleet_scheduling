"""Run the project directly from a source checkout.

This launcher makes the ``src`` package importable without requiring an
editable installation first. The implementation remains in
``coord_schedule_uam.experiment_runner``.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from coord_schedule_uam.experiment_runner import main  # noqa: E402


if __name__ == "__main__":
    main()

