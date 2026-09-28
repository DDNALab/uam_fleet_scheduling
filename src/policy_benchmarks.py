"""Helpers for policy benchmarks used in the hybrid UAM paper.

The booking-only (BO) benchmark uses advance bookings and aircraft-side
uncertainty for Stage-1 planning but deliberately ignores stochastic on-demand
requests when selecting n[d,t] and b[t]. The resulting first-stage decisions
are fixed and evaluated under the full scenario distribution.
"""

from copy import deepcopy

import pulp


def booking_only_planning_input(model_input):
    """Return a copy with on-demand requests set to zero for BO Stage-1 planning."""
    planning = deepcopy(model_input)
    for scenario in planning["scenarios"].values():
        scenario["on_demand_demand"] = {
            key: 0 for key in scenario.get("on_demand_demand", {})
        }
    return planning


def extract_first_stage_solution(model_data):
    """Extract solved integer n[d,t] and b[t] values from a model-data object."""
    variables = model_data["variables"]
    periods = model_data["periods"]
    destinations = model_data["destinations"]
    return {
        "x0": int(round(float(pulp.value(variables["initial_fleet"]) or 0.0))),
        "n": {
            (d, t): int(round(float(pulp.value(variables["n"][d, t]) or 0.0)))
            for d in destinations for t in periods
        },
        "b": {
            t: int(round(float(pulp.value(variables["b"][t]) or 0.0)))
            for t in periods
        },
    }


def fix_first_stage_decisions(model_data, first_stage):
    """Fix n and b to a previously solved policy for full-demand/OOS evaluation."""
    model = model_data["model"]
    variables = model_data["variables"]
    periods = model_data["periods"]
    destinations = model_data["destinations"]

    model += variables["initial_fleet"] == int(first_stage["x0"]), "FixInitialFleet"
    for d in destinations:
        for t in periods:
            value = int(first_stage["n"].get((d, t), 0))
            model += variables["n"][d, t] == value, f"FixPlannedDeparture_{d}_{t}"
    for t in periods:
        value = int(first_stage["b"].get(t, 0))
        model += variables["b"][t] == value, f"FixReservedCharging_{t}"
    return model_data


def solve_booking_only_benchmark(model_input, solver="auto", **solve_kwargs):
    """Solve BO planning, then evaluate the fixed policy on the full scenarios.

    This implements the manuscript benchmark exactly:
      1. keep known bookings and aircraft-side uncertainty;
      2. set on-demand requests to zero only when choosing n and b;
      3. extract and fix those first-stage decisions;
      4. evaluate recourse under the original full on-demand scenarios.
    """
    from .stochastic_model import (
        add_constraints_and_objective,
        build_stochastic_model,
        solve_model,
    )

    planning_input = booking_only_planning_input(model_input)
    planning_data = build_stochastic_model(planning_input)
    planning_model = add_constraints_and_objective(planning_data)
    planning_result = solve_model(planning_model, solver=solver, **solve_kwargs)

    first_stage = extract_first_stage_solution(planning_data)

    evaluation_data = build_stochastic_model(model_input)
    evaluation_model = add_constraints_and_objective(evaluation_data)
    fix_first_stage_decisions(evaluation_data, first_stage)
    evaluation_result = solve_model(evaluation_model, solver=solver, **solve_kwargs)

    return {
        "first_stage": first_stage,
        "planning_data": planning_data,
        "planning_result": planning_result,
        "evaluation_data": evaluation_data,
        "evaluation_result": evaluation_result,
    }
