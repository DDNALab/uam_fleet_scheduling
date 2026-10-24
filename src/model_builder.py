"""
Assemble the solver input for the two-stage stochastic program.

Takes the reduced scenarios and the baseline demand block and returns one
``model_input`` dictionary that both formulations consume unchanged:
:mod:`stochastic_model` (aircraft-indexed) and :mod:`aggregated_model`
(duration-aggregated cumulative flow).

Layout::

    {
      "parameters"   : full parameter block (resolved grid and costs),
      "destinations" : [d, ...],
      "periods"      : [1, ..., L],
      "booked_demand": {(t, d): int},        # first-stage known bookings
      "scenarios"    : {sid: {
                           "probability"      : pi_s,
                           "ondemand_demand"  : {(t, d): int},
                           "aircraft"         : {j: {
                               "arrival_period", "initial_soc",
                               "duration", "energy"}},
                           "common_factor"    : ndarray,
                       }},
      "baseline"     : passthrough for reporting (fares, distances, ...),
    }

This module creates no decision variables and calls no solver.
"""

import config
from economics import (
    destination_distances,
    destination_fares,
    flight_operating_cost,
)


# ============================================================
# Fare table
# ============================================================

def calculate_fares(vertiports=None, hub=None):
    """
    Fare table ``{destination: rho_d}`` with ``rho_d = 20 + 3 * Dist_d``.
    Fares are in dollars.

    Distances are great-circle miles from the hub, so the table is reproducible
    without the ACS shapefiles.
    """
    return destination_fares(vertiports, hub)


# ============================================================
# Scenario blocks
# ============================================================

def build_aircraft_block(scenario, parameters=None):
    """
    Convert a scenario's incoming aircraft list into a ``{j: info}`` dict.

    Each entry keeps the aircraft attributes the model may read: arrival period,
    initial state of charge, required charging duration in periods, and the
    energy drawn from the charger, which prices charging.
    """
    parameters = config.build_parameters() if parameters is None else parameters

    block = {}

    for aircraft in scenario.aircraft:

        block[int(aircraft.id)] = {
            "arrival_period": int(aircraft.arrival_period),
            "initial_soc": float(aircraft.initial_soc),
            "duration": int(aircraft.duration),
            "energy": float(aircraft.energy),
        }

    return block


def build_demand_block(scenario, destinations=None, periods=None):
    """
    Realized on-demand requests ``lambda^O_tds`` as ``{(t, d): int}``.

    Cells outside the modeled grid, or for inactive destinations, are dropped.
    """
    allowed_destinations = None if destinations is None else set(destinations)

    allowed_periods = None if periods is None else set(int(t) for t in periods)

    demand = {}

    for (t, d), value in scenario.ondemand_demand.items():

        t = int(t)

        if allowed_destinations is not None and d not in allowed_destinations:
            continue

        if allowed_periods is not None and t not in allowed_periods:
            continue

        count = int(value)

        if count > 0:
            demand[(t, d)] = count

    return demand


def build_booked_block(booked, destinations=None, periods=None):
    """
    Known advance bookings ``B_td`` as ``{(t, d): int}``.

    Booked demand is observed before uncertainty, so it is a parameter of the
    model, not a scenario attribute.
    """
    allowed_destinations = None if destinations is None else set(destinations)

    allowed_periods = None if periods is None else set(int(t) for t in periods)

    block = {}

    for (t, d), value in (booked or {}).items():

        t = int(t)

        if allowed_destinations is not None and d not in allowed_destinations:
            continue

        if allowed_periods is not None and t not in allowed_periods:
            continue

        count = int(value)

        if count > 0:
            block[(t, d)] = count

    return block


# ============================================================
# Main builder
# ============================================================

def build_model_input(
    reduced_scenarios,
    baseline=None,
    parameters=None,
    destinations=None,
    periods=None,
    **parameter_overrides,
):
    """
    Build the ``model_input`` consumed by either formulation.

    Parameters
    ----------
    reduced_scenarios : list of Scenario
        Representative scenarios with probabilities summing to one.
    baseline : dict, optional
        Output of :func:`demand_model.build_baseline_demand`. Supplies the fare
        and distance tables and the known bookings.
    parameters : dict, optional
        Pre-built parameter block. Mutually exclusive with
        ``parameter_overrides``; when omitted the block is built from
        :func:`config.build_parameters`.
    destinations, periods : sequence, optional
        Restrict the instance to a subset of the active destinations and/or a
        shorter horizon, which is how small validation instances are built.
    **parameter_overrides
        Forwarded to :func:`config.build_parameters`, e.g.
        ``booking_share=0.6`` or ``cvar={"weight": 0.0}``.

    Returns
    -------
    dict
        Solver-ready ``model_input``.
    """
    if parameters is None:
        parameters = config.build_parameters(**parameter_overrides)
    elif parameter_overrides:
        raise ValueError(
            "pass either a prepared 'parameters' block or keyword overrides, not both"
        )
    else:
        parameters = dict(parameters)

    baseline = {} if baseline is None else baseline

    reduced_scenarios = list(reduced_scenarios)

    # ---- grid -----------------------------------------------------------
    if destinations is None:
        destinations = list(
            baseline.get("destinations")
            or config.ACTIVE_DESTINATIONS
        )
    destinations = sorted(destinations)

    if periods is None:
        if baseline.get("periods"):
            periods = [int(t) for t in baseline["periods"]]
        else:
            n_periods = int(parameters["periods"])
            periods = list(range(1, n_periods + 1))
    else:
        periods = sorted(int(t) for t in periods)

    parameters["periods"] = len(periods)
    parameters["los_periods"] = config.los_periods(parameters.get("bin_size"))

    # ---- economics tables ----------------------------------------------
    distances = config_destinations_distances(baseline)

    fares = dict(baseline.get("fares") or {})

    if not fares:
        fares = destination_fares()

    flight_costs = {
        d: flight_operating_cost(distances.get(d, 0.0), parameters)
        for d in destinations
    }

    parameters["destination_fares"] = {d: fares.get(d, 0.0) for d in destinations}
    parameters["destination_distances"] = {
        d: distances.get(d, 0.0) for d in destinations
    }
    parameters["destination_flight_cost"] = flight_costs
    parameters["destinations"] = list(destinations)

    # ---- first-stage known data ----------------------------------------
    booked_demand = build_booked_block(
        baseline.get("booked", {}), destinations=destinations, periods=periods
    )

    # ---- scenario blocks -----------------------------------------------
    scenarios = {}

    for scenario in reduced_scenarios:

        scenarios[int(scenario.id)] = {
            "probability": float(scenario.probability),
            "ondemand_demand": build_demand_block(
                scenario, destinations=destinations, periods=periods
            ),
            "aircraft": build_aircraft_block(scenario, parameters),
            "common_factor": scenario.common_factor,
        }

    return {
        "parameters": parameters,
        "destinations": destinations,
        "periods": periods,
        "booked_demand": booked_demand,
        "scenarios": scenarios,
        "baseline": baseline,
    }


def config_destinations_distances(baseline):
    """Distance table from the baseline block, falling back to geometry."""
    distances = dict(baseline.get("distances") or {})

    if not distances:
        distances = destination_distances()

    return distances


# ============================================================
# Small-instance helper
# ============================================================

def build_small_instance(
    n_destinations=2,
    n_periods=6,
    n_scenarios=3,
    seed=None,
    booking_share=None,
    **parameter_overrides,
):
    """
    Self-contained tiny instance for validation and tests.

    Needs no ACS pipeline, solver or generated files: demand uses equal
    destination shares over a short horizon, scenarios are sampled, and the
    result is assembled exactly like a full run. Used to check that the two
    formulations agree.
    """
    from demand_model import build_baseline_demand
    from scenario_generator import generate_scenarios
    from scenario_reduction import reduce_scenarios

    parameters = config.build_parameters(**parameter_overrides)

    destinations = list(config.ACTIVE_DESTINATIONS[: int(n_destinations)])

    baseline = build_baseline_demand(
        destinations=destinations,
        n_periods=int(n_periods),
        booking_share=booking_share,
        total_passengers=parameters.get("total_passengers", config.TOTAL_PASSENGER_TRIPS),
    )

    raw = generate_scenarios(
        baseline,
        n_scenarios=int(n_scenarios),
        seed=parameters.get("seed") if seed is None else int(seed),
        parameters=parameters,
    )

    reduced = (
        raw
        if len(raw) <= int(n_scenarios)
        else reduce_scenarios(raw, target_size=int(n_scenarios), parameters=parameters)
    )

    return build_model_input(
        reduced,
        baseline=baseline,
        parameters=parameters,
        destinations=destinations,
        periods=list(range(1, int(n_periods) + 1)),
    )
