import pytest

pulp = pytest.importorskip("pulp")

from src.stochastic_model import (
    build_stochastic_model,
    add_constraints_and_objective,
    solve_model,
)


def _small_input(n_aircraft=2):
    destination = "D"
    aircraft = {
        j: {"arrival_period": 1, "initial_soc": 70.0}
        for j in range(n_aircraft)
    }
    return {
        "parameters": {
            "periods": 4,
            "base_price": 0.1,
            "peak_multiplier": 1.5,
            "shoulder_multiplier": 1.0,
            "off_multiplier": 0.5,
            "bin_size": 15,
            "los_periods": 2,
            "destination_fares": {destination: 20.0},
            "horizon": {"start": 480, "end": 540, "bin_size": 15},
            "battery_capacity": 110,
            "charging_rate": 120,
            "soc_min": 70,
            "evtol_capacity": 4,
            "charging_facilities": 2,
            "max_departures": 2,
            "initial_fleet_max": 0,
            "costs": {
                "charging_reservation": 0.0,
                "initial_fleet": 0.0,
                "commitment": 0.1,
                "added_departure": 1.0,
                "cancelled_departure": 1.0,
                "emergency_capacity": 1.0,
                "unserved_booked": 200.0,
                "unserved_on_demand": 100.0,
                "flight": 0.0,
            },
            "cvar": {
                "alpha": 0.9,
                "weight": 0.0,
                "booked_weight": 2.0,
                "on_demand_weight": 1.0,
            },
        },
        "advance_bookings": {(1, destination): 4},
        "scenarios": {
            0: {
                "probability": 1.0,
                "aircraft": aircraft,
                "on_demand_demand": {(1, destination): 2},
                "common_factor": [0, 0, 0, 0],
            }
        },
    }


def test_projected_model_has_no_aircraft_indexed_decision_tensor():
    md = build_stochastic_model(_small_input())
    add_constraints_and_objective(md)
    names = {var.name for var in md["model"].variables()}
    assert not any(name.startswith("Activation_") for name in names)
    assert any(name.startswith("DurationStart_") for name in names)
    assert any(name.startswith("BookedFlow_") for name in names)
    assert any(name.startswith("OnDemandFlow_") for name in names)


def test_known_bookings_force_protected_stage1_seat_capacity():
    md = build_stochastic_model(_small_input())
    model = add_constraints_and_objective(md)
    result = solve_model(model, solver="highs", time_limit=30, gap_rel=1e-6, verbose=False)
    assert result["status"] == "Optimal"

    v = md["variables"]
    planned = sum(pulp.value(v["n"]["D", t]) for t in md["periods"])
    assert planned >= 1.0 - 1e-7

    unserved_booked = sum(
        pulp.value(v["unserved_booked"][0, r, "D"])
        for r in md["periods"]
    )
    unserved_on = sum(
        pulp.value(v["unserved_on_demand"][0, r, "D"])
        for r in md["periods"]
    )
    assert unserved_booked <= 1e-7
    assert unserved_on <= 1e-7


def test_booked_passengers_are_protected_when_aircraft_are_scarce():
    md = build_stochastic_model(_small_input(n_aircraft=1))
    model = add_constraints_and_objective(md)
    result = solve_model(model, solver="highs", time_limit=30, gap_rel=1e-6, verbose=False)
    assert result["status"] == "Optimal"

    v = md["variables"]
    ub = sum(pulp.value(v["unserved_booked"][0, r, "D"]) for r in md["periods"])
    uo = sum(pulp.value(v["unserved_on_demand"][0, r, "D"]) for r in md["periods"])
    assert ub <= 1e-7
    assert uo >= 2.0 - 1e-7


def test_booking_only_planning_input_removes_only_on_demand_requests():
    from src.policy_benchmarks import booking_only_planning_input

    full = _small_input()
    bo = booking_only_planning_input(full)
    assert bo["advance_bookings"] == full["advance_bookings"]
    assert bo["scenarios"][0]["aircraft"] == full["scenarios"][0]["aircraft"]
    assert sum(bo["scenarios"][0]["on_demand_demand"].values()) == 0
    assert sum(full["scenarios"][0]["on_demand_demand"].values()) == 2
