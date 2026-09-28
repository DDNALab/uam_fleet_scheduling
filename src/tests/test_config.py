from src import config


def test_horizon_is_divisible_into_bins():
    assert config.HORIZON_MINUTES > 0
    assert config.HORIZON_MINUTES % config.BIN_SIZE == 0
    assert config.N_PERIODS == config.HORIZON_MINUTES // config.BIN_SIZE


def test_hybrid_service_parameters_are_valid():
    assert 0.0 <= config.ADVANCE_BOOKING_FRACTION <= 1.0
    assert config.BOOKED_UNSERVED_PASSENGER_COST >= config.ONDEMAND_UNSERVED_PASSENGER_COST
    assert config.CVAR_BOOKED_WEIGHT >= config.CVAR_ONDEMAND_WEIGHT >= 0.0


def test_initial_fleet_and_hard_waiting_limit():
    assert config.INITIAL_FLEET_MAX >= 0
    assert config.INITIAL_FLEET_COST >= 0
    assert config.LOS_MINUTES == 30
    assert config.LOS_PERIODS * config.BIN_SIZE == config.LOS_MINUTES


def test_new_data_file_layout():
    # User may override the directory via UAM_DATA_DIR; no data is shipped.
    from pathlib import Path
    assert config.ACS_JSON_PATH.name == 'acs_texas_2022_raw.json'
    assert config.SHAPEFILE_PATH.name == 'tl_2022_48_tract.shp'
    assert config.SHAPEFILE_PATH.parent.name == 'tl_2022_48_tract'
