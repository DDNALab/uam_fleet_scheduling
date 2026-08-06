from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coord_schedule_uam import config


def test_bundled_input_paths_exist():
    assert config.ACS_JSON_PATH.is_file()
    assert config.SHAPEFILE_PATH.is_file()


def test_horizon_is_divisible_into_bins():
    assert config.HORIZON_MINUTES > 0
    assert config.HORIZON_MINUTES % config.BIN_SIZE == 0
    assert config.L_PERIODS == config.HORIZON_MINUTES // config.BIN_SIZE


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
