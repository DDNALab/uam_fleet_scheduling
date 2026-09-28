"""Case-specific parameters must reach the real model-input builder.

This optional integration test runs when the project's regular PuLP/model-builder
modules are installed. It does not invoke the MILP solver or need geographic files.
"""
import pytest

pytest.importorskip("pulp")

from src import config
from src.diagnostics.demand_design import time_profile_for_horizon
from src.diagnostics.scale_instances import apply_case_settings
from src.scenario_generator import generate_scenarios
from src.scenario_reduction import reduce_scenarios
from src.model_builder import build_model_input


def test_12_hour_settings_reach_model_input():
    settings = config.scalability_settings(2000, mode="proportional")
    profile = time_profile_for_horizon(settings["horizon_minutes"])
    raw = generate_scenarios(
        {"Dallas CBD": 80.0}, profile, n_scenarios=3, seed=60,
        booking_fraction=.4, aircraft_passenger_ratio=.1,
    )
    reduced = reduce_scenarios(raw, target_size=2)
    mi = apply_case_settings(build_model_input(reduced), settings)
    params = mi["parameters"]
    assert params["periods"] == 48
    assert params["horizon"]["start"] == 480
    assert params["horizon"]["end"] == 1200
    assert params["initial_fleet_max"] == 40
    assert params["charging_facilities"] == 10
    assert params["max_departures"] == 57
    assert all(1 <= a["arrival_period"] <= 48
               for s in mi["scenarios"].values() for a in s["aircraft"].values())
