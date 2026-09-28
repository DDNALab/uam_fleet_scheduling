"""Focused tests for case-dependent horizons, scaled resources, and booking preflight.

Run from the project root:
  python -m pytest -q src/tests/test_scaling_profile.py
"""

import math

import numpy as np
import pytest

from src import config
from src.diagnostics.demand_design import (
    booking_capacity_preflight,
    time_profile_for_horizon,
)
from src.scenario_generator import generate_advance_bookings, generate_scenarios


@pytest.mark.parametrize(
    "passengers,hours,fleet,chargers,departures,periods",
    [(200, 4, 12, 3, 17, 16),
     (500, 6, 20, 5, 29, 24),
     (1000, 8, 30, 8, 43, 32),
     (2000, 12, 40, 10, 57, 48)],
)
def test_canonical_proportional_profiles(passengers, hours, fleet,
                                         chargers, departures, periods):
    settings = config.scalability_settings(passengers, mode="proportional")
    assert settings["horizon_minutes"] == hours * 60
    assert settings["n_periods"] == periods
    assert settings["initial_fleet_max"] == fleet
    assert settings["charging_facilities"] == chargers
    assert settings["max_departures"] == departures
    weights = time_profile_for_horizon(settings["horizon_minutes"])
    assert len(weights) == periods
    assert math.isclose(float(weights.sum()), 1, abs_tol=1e-12)
    assert np.all(weights >= 0)


def test_fixed_defaults_unchanged():
    s = config.scalability_settings(2000, mode="fixed")
    assert (s["n_periods"], s["initial_fleet_max"],
            s["charging_facilities"], s["max_departures"]) == (16, 12, 3, 17)


def test_arbitrary_horizon_requires_explicit_hours():
    with pytest.raises(ValueError, match="--horizons"):
        config.scalability_settings(700, mode="proportional")
    assert config.scalability_settings(700, mode="proportional",
                                       horizon_hours=7)["n_periods"] == 28


def test_booking_preflight_exact_nominal_capacity():
    # All eight bookings appear in period 2 and can only use its one flight
    # per 15 minutes; they cannot fit into that flight's four seats.
    bookings = {(2, "A"): 4, (2, "B"): 4}
    infeasible = booking_capacity_preflight(
        bookings, destinations=["A", "B"], n_periods=2,
        los_periods=2, seats_per_flight=4,
        max_departures_per_period=1,
    )
    assert infeasible["status"] == "Infeasible"
    feasible = booking_capacity_preflight(
        bookings, destinations=["A", "B"], n_periods=2,
        los_periods=2, seats_per_flight=4,
        max_departures_per_period=2,
    )
    assert feasible["status"] == "Feasible"
    assert feasible["nominal_flights_certificate"] == 2


def test_2000_booking_preflight_can_pass_with_12_hour_profile():
    s = config.scalability_settings(2000, mode="proportional")
    weights = time_profile_for_horizon(s["horizon_minutes"])
    expected = {f"D{i}": 2000 / 9 for i in range(9)}
    bookings = generate_advance_bookings(expected, weights, booking_fraction=0.4)
    outcome = booking_capacity_preflight(
        bookings, destinations=expected, n_periods=s["n_periods"],
        los_periods=config.LOS_PERIODS, seats_per_flight=config.EVTOL_CAPACITY,
        max_departures_per_period=s["max_departures"],
    )
    assert outcome["status"] == "Feasible"


def test_generated_scenarios_follow_case_horizon():
    weights = time_profile_for_horizon(12 * 60)
    scenarios = generate_scenarios(
        {"Dallas CBD": 80.0}, weights, n_scenarios=2,
        seed=61, booking_fraction=0.4, aircraft_passenger_ratio=0.5,
    )
    assert sum(scenarios[0].advance_bookings.values()) == 32
    for s in scenarios:
        assert len(s.common_factor) == 48
        assert all(1 <= r <= 48 for r, _ in s.on_demand_demand)
        assert all(config.HORIZON_START <= a.arrival_time <
                   config.HORIZON_START + 12 * 60 for a in s.aircraft)
