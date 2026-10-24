"""
Duration-aggregated cumulative-flow reformulation of the two-stage model.

Drops aircraft identities that have no operational consequence, which makes the
model far smaller than the aircraft-indexed formulation in
:mod:`stochastic_model` while preserving the same feasible first-stage policies
and the same optimal objective value. The interface mirrors that module, so the
two formulations are interchangeable::

    build_aggregated_model(model_input) -> model_data
    add_constraints_and_objective(model_data) -> pulp.LpProblem

Second-stage variables, per scenario ``s``:
    A0[s]         initially ready aircraft actually deployed
    chi[g,t]      group-to-start aircraft flow (continuous)
    w[h,t]        duration-h charging operations starting at period t
    f[d,k]        realized flights to destination d in period k
    yB[r,d,k]     booked passengers served on (d, k)
    yO[r,d,k]     on-demand passengers served on (d, k)
    etaB, etaO    unserved booked / on-demand passengers
    bplus[t]      emergency charging units
    vplus, vminus added / cancelled departures
    C[k], D[k]    charge completions and departures

Exactness holds because post-charge aircraft are interchangeable: they share
destination feasibility, capacity and destination-specific operating cost. That
holds for the homogeneous eVTOL fleet used here; it would not hold for a
heterogeneous fleet with group-dependent range, eligibility, capacity or cost.

A group is one distinct ``(arrival period, charging duration, energy)`` triple.
Arrival SoC is continuous, so the base case has one group per aircraft and the
charging-cost coefficient ``energy * P(t)`` is never averaged across aircraft
with different energy needs; the saving comes only from the integer dimension.

Pass ``group_energy_tolerance`` (kWh) to bucket energies and merge groups. That
shrinks the group count further but perturbs the charging cost coefficient, so
it is an approximation and must be reported as such.
"""

import pulp

from economics import charging_cost_per_kwh
from stochastic_model import (
    add_first_stage_constraints,
    admissible_departures,
    feasible_starts,
    formulation_size,
    name,
)


# ============================================================
# Aircraft grouping
# ============================================================

#: Energy tolerance in kWh. ``0`` keeps groups exact (one per distinct energy),
#: which is what preserves the charging cost of the indexed formulation.
GROUP_ENERGY_TOLERANCE = 0.0


def build_groups(aircraft, parameters=None, tolerance=None):
    """
    Partition one scenario's incoming aircraft into groups.

    Aircraft in a group share an arrival period, a charging duration and an
    energy requirement, so they are interchangeable for charging.

    Parameters
    ----------
    aircraft : dict
        ``{j: {"arrival_period", "duration", "energy", ...}}`` from
        :func:`model_builder.build_aircraft_block`.
    parameters : dict, optional
        Accepted for interface symmetry; currently unused.
    tolerance : float, optional
        Energy bucketing width in kWh. ``0`` (default) groups by exact energy.

    Returns
    -------
    dict
        ``{(arrival, duration, bucket): {"arrival", "duration", "energy", "N",
        "members"}}``.
    """
    tolerance = GROUP_ENERGY_TOLERANCE if tolerance is None else float(tolerance)

    groups = {}

    for j in sorted(aircraft):

        info = aircraft[j]

        arrival = int(info["arrival_period"])

        duration = int(info["duration"])

        energy = float(info["energy"])

        if tolerance > 0:

            bucket = round(energy / tolerance) * tolerance

        else:

            bucket = round(energy, 6)

        key = (arrival, duration, bucket)

        if key not in groups:

            groups[key] = {
                "arrival": arrival,
                "duration": duration,
                "energy": bucket,
                "N": 0,
                "members": [],
            }

        groups[key]["N"] += 1

        groups[key]["members"].append(j)

    return groups


def group_starts(group, horizon):
    """Charging start periods that leave room to finish by the horizon."""
    return feasible_starts(group["arrival"], group["duration"], horizon)


# ============================================================
# Model construction
# ============================================================

def build_aggregated_model(model_input, group_energy_tolerance=None):
    """
    Declare variables and index sets for the aggregated reformulation.

    Returns
    -------
    dict
        ``model_data`` consumed by :func:`add_constraints_and_objective`.
    """
    parameters = model_input["parameters"]
    scenarios = model_input["scenarios"]
    booked_demand = model_input["booked_demand"]
    destinations = list(model_input["destinations"])
    periods = [int(t) for t in model_input["periods"]]

    horizon = len(periods)
    los = int(parameters["los_periods"])
    facilities = int(parameters["charging_facilities"])
    max_fleet = int(parameters["max_initial_fleet"])

    model = pulp.LpProblem(
        name("TwoStage_eVTOL_AggregatedCumulativeFlow"),
        pulp.LpMaximize,
    )

    windows = {r: admissible_departures(r, horizon, los) for r in periods}

    # ========================================================
    # FIRST STAGE -- same variables as the indexed formulation
    # ========================================================

    x0 = pulp.LpVariable(
        name("InitialFleet"), lowBound=0, upBound=max_fleet, cat="Integer"
    )

    n = {
        (destination, t): pulp.LpVariable(
            name("PlannedDepartures", destinations.index(destination), t),
            lowBound=0,
            cat="Integer",
        )
        for destination in destinations
        for t in periods
    }

    b = {
        t: pulp.LpVariable(
            name("ChargingReservation", t),
            lowBound=0,
            upBound=facilities,
            cat="Integer",
        )
        for t in periods
    }

    protection = {}

    for (r, destination), count in booked_demand.items():

        r = int(r)

        if r not in windows or int(count) <= 0:
            continue

        for k in windows[r]:
            protection[r, destination, k] = pulp.LpVariable(
                name("BookingProtection", r, destinations.index(destination), k),
                lowBound=0,
                cat="Continuous",
            )

    # ========================================================
    # SECOND STAGE
    # ========================================================

    variables = {
        "x0": x0,
        "n": n,
        "b": b,
        "protection": protection,
        "deployed_initial": {},
        "groups": {},
        "group_charge": {},
        "duration_start": {},
        "flights": {},
        "booked_passengers": {},
        "ondemand_passengers": {},
        "unserved_booked": {},
        "unserved_ondemand": {},
        "completions": {},
        "departure_totals": {},
        "emergency_capacity": {},
        "added_departures": {},
        "cancelled_departures": {},
        "loss": {},
        "rho": {},
    }

    structure = {
        "windows": windows,
        "scenarios": {},
    }

    for scenario_id, scenario in scenarios.items():

        aircraft = scenario["aircraft"]

        ondemand = scenario["ondemand_demand"]

        groups = build_groups(aircraft, parameters, group_energy_tolerance)

        durations = sorted({group["duration"] for group in groups.values()})

        # ---- deployed initial fleet --------------------------------
        deployed_initial = pulp.LpVariable(
            name("DeployedInitialFleet", scenario_id),
            lowBound=0,
            upBound=max_fleet,
            cat="Integer",
        )

        # ---- group flow (continuous) and duration-start counts (integer) --
        group_charge = {}

        group_index = {
            key: position for position, key in enumerate(sorted(groups))
        }

        for key in sorted(groups):

            group = groups[key]

            for t in group_starts(group, horizon):

                group_charge[key, t] = pulp.LpVariable(
                    name(
                        "GroupCharge",
                        scenario_id,
                        group_index[key],
                        t,
                    ),
                    lowBound=0,
                    cat="Continuous",
                )

        duration_start = {}

        for duration in durations:

            for t in periods:

                eligible = [
                    key
                    for key in groups
                    if groups[key]["duration"] == duration
                    and t in group_starts(groups[key], horizon)
                ]

                if not eligible:
                    continue

                duration_start[duration, t] = pulp.LpVariable(
                    name("DurationStart", scenario_id, duration, t),
                    lowBound=0,
                    cat="Integer",
                )

        # ---- realized flights per (destination, departure period) ---
        flights = {
            (destination, k): pulp.LpVariable(
                name("Flights", scenario_id, destinations.index(destination), k),
                lowBound=0,
                cat="Integer",
            )
            for destination in destinations
            for k in periods
        }

        # ---- integer passenger flows by class and departure --------
        booked = {
            (int(r), destination): int(value)
            for (r, destination), value in booked_demand.items()
            if int(value) > 0
        }

        demand = {
            (int(r), destination): int(value)
            for (r, destination), value in ondemand.items()
            if int(value) > 0
        }

        booked_passengers = {}

        ondemand_passengers = {}

        unserved_booked = {}

        unserved_ondemand = {}

        for (r, destination) in booked:

            unserved_booked[r, destination] = pulp.LpVariable(
                name("UnservedBooked", scenario_id, r, destinations.index(destination)),
                lowBound=0,
                cat="Integer",
            )

            for k in windows.get(r, []):

                booked_passengers[r, destination, k] = pulp.LpVariable(
                    name(
                        "BookedFlow",
                        scenario_id,
                        r,
                        destinations.index(destination),
                        k,
                    ),
                    lowBound=0,
                    cat="Integer",
                )

        for (r, destination) in demand:

            unserved_ondemand[r, destination] = pulp.LpVariable(
                name(
                    "UnservedOndemand",
                    scenario_id,
                    r,
                    destinations.index(destination),
                ),
                lowBound=0,
                cat="Integer",
            )

            for k in windows.get(r, []):

                ondemand_passengers[r, destination, k] = pulp.LpVariable(
                    name(
                        "OndemandFlow",
                        scenario_id,
                        r,
                        destinations.index(destination),
                        k,
                    ),
                    lowBound=0,
                    cat="Integer",
                )

        # ---- charge completions and departures per period ----------
        completions = {
            k: pulp.LpVariable(
                name("ChargeCompletions", scenario_id, k), lowBound=0, cat="Continuous"
            )
            for k in periods
        }

        departure_totals = {
            k: pulp.LpVariable(
                name("DepartureTotal", scenario_id, k), lowBound=0, cat="Continuous"
            )
            for k in periods
        }

        emergency_capacity = {
            t: pulp.LpVariable(
                name("EmergencyCharge", scenario_id, t),
                lowBound=0,
                upBound=facilities,
                cat="Integer",
            )
            for t in periods
        }

        added_departures = {
            (destination, t): pulp.LpVariable(
                name("AddedDepartures", scenario_id, destinations.index(destination), t),
                lowBound=0,
                cat="Integer",
            )
            for destination in destinations
            for t in periods
        }

        cancelled_departures = {
            (destination, t): pulp.LpVariable(
                name(
                    "CancelledDepartures",
                    scenario_id,
                    destinations.index(destination),
                    t,
                ),
                lowBound=0,
                cat="Integer",
            )
            for destination in destinations
            for t in periods
        }

        loss = pulp.LpVariable(
            name("ServiceLoss", scenario_id), lowBound=0, cat="Continuous"
        )

        rho = pulp.LpVariable(
            name("CVaRExcess", scenario_id), lowBound=0, cat="Continuous"
        )

        # ---- collect per-scenario variables and index sets ---------
        variables["deployed_initial"][scenario_id] = deployed_initial
        variables["groups"][scenario_id] = groups
        variables["group_charge"][scenario_id] = group_charge
        variables["duration_start"][scenario_id] = duration_start
        variables["flights"][scenario_id] = flights
        variables["booked_passengers"][scenario_id] = booked_passengers
        variables["ondemand_passengers"][scenario_id] = ondemand_passengers
        variables["unserved_booked"][scenario_id] = unserved_booked
        variables["unserved_ondemand"][scenario_id] = unserved_ondemand
        variables["completions"][scenario_id] = completions
        variables["departure_totals"][scenario_id] = departure_totals
        variables["emergency_capacity"][scenario_id] = emergency_capacity
        variables["added_departures"][scenario_id] = added_departures
        variables["cancelled_departures"][scenario_id] = cancelled_departures
        variables["loss"][scenario_id] = loss
        variables["rho"][scenario_id] = rho

        structure["scenarios"][scenario_id] = {
            "groups": groups,
            "durations": durations,
            "booked": booked,
            "ondemand": demand,
        }

    return {
        "model": model,
        "parameters": parameters,
        "destinations": destinations,
        "periods": periods,
        "scenarios": scenarios,
        "booked_demand": booked_demand,
        "variables": variables,
        "structure": structure,
        "formulation": "aggregated",
    }


# ============================================================
# Constraints and objective
# ============================================================

def add_constraints_and_objective(model_data):
    """
    Add stage-2 constraints, the CVaR terms and the objective.

    Returns the populated :class:`pulp.LpProblem`.
    """
    model = model_data["model"]
    parameters = model_data["parameters"]
    scenarios = model_data["scenarios"]
    destinations = model_data["destinations"]
    periods = model_data["periods"]

    variables = model_data["variables"]
    structure = model_data["structure"]

    horizon = len(periods)
    capacity = int(parameters["evtol_capacity"])
    max_departures = int(parameters["max_departures"])

    x0 = variables["x0"]

    costs = parameters["costs"]

    fare = parameters["destination_fares"]
    flight_cost = parameters["destination_flight_cost"]

    alpha = float(parameters["cvar"]["alpha"])
    theta = float(parameters["cvar"]["weight"])
    omega_booked = float(parameters["cvar"]["weight_booked"])
    omega_ondemand = float(parameters["cvar"]["weight_ondemand"])

    penalty_booked = float(parameters["unserved_booked"])
    penalty_ondemand = float(parameters["unserved_ondemand"])

    # ========================================================
    # FIRST-STAGE CONSTRAINTS (shared with the indexed model)
    # ========================================================

    add_first_stage_constraints(model, model_data)

    scenario_profit = {}

    for scenario_id, scenario in scenarios.items():

        info = structure["scenarios"][scenario_id]

        groups = variables["groups"][scenario_id]
        group_charge = variables["group_charge"][scenario_id]
        duration_start = variables["duration_start"][scenario_id]
        flights = variables["flights"][scenario_id]
        booked_passengers = variables["booked_passengers"][scenario_id]
        ondemand_passengers = variables["ondemand_passengers"][scenario_id]
        unserved_booked = variables["unserved_booked"][scenario_id]
        unserved_ondemand = variables["unserved_ondemand"][scenario_id]
        completions = variables["completions"][scenario_id]
        departure_totals = variables["departure_totals"][scenario_id]
        emergency_capacity = variables["emergency_capacity"][scenario_id]
        added_departures = variables["added_departures"][scenario_id]
        cancelled_departures = variables["cancelled_departures"][scenario_id]

        durations = info["durations"]
        booked = info["booked"]
        demand = info["ondemand"]

        deployed_initial = variables["deployed_initial"][scenario_id]

        # Deployed aircraft cannot exceed the first-stage fleet.

        model += (
            deployed_initial <= x0,
            name("DeployedInitialFleetLimit", scenario_id),
        )

        # Group flow never exceeds the number of aircraft in the group.

        for key, group in groups.items():

            starts = group_starts(group, horizon)

            if not starts:
                continue

            model += (
                pulp.lpSum(group_charge[key, t] for t in starts) <= group["N"],
                name("GroupSize", scenario_id, key[0], key[1], key[2]),
            )

        # Duration-start totals: integer margin w equals the sum of the
        # continuous group flows chi. chi always has an integral optimum for
        # integer w, so relaxing it loses no optimal value.

        for (duration, t), margin in duration_start.items():

            members = [
                group_charge[key, t]
                for key in groups
                if groups[key]["duration"] == duration and (key, t) in group_charge
            ]

            model += (
                pulp.lpSum(members) == margin,
                name("DurationStartTotal", scenario_id, duration, t),
            )

        # Chargers occupied in a period are covered by reserved and emergency
        # units, the two together never exceeding the facility count.

        for period in periods:

            occupying = [
                margin
                for (duration, t), margin in duration_start.items()
                if t <= period < t + duration
            ]

            model += (
                pulp.lpSum(occupying) <= variables["b"][period] + emergency_capacity[period],
                name("ChargerOccupancy", scenario_id, period),
            )

            model += (
                emergency_capacity[period] <= int(parameters["charging_facilities"])
                - variables["b"][period],
                name("EmergencyCapacityLimit", scenario_id, period),
            )

        # Charge completions and departures per period.

        for k in periods:

            completing = [
                duration_start[duration, k - duration]
                for duration in durations
                if (duration, k - duration) in duration_start
            ]

            model += (
                completions[k] == pulp.lpSum(completing),
                name("CompletionDefinition", scenario_id, k),
            )

            model += (
                departure_totals[k]
                == pulp.lpSum(flights[destination, k] for destination in destinations),
                name("DepartureTotalDefinition", scenario_id, k),
            )

        # Cumulative flow: by period k, departures cannot exceed aircraft
        # deployed at the start plus charge completions. Initial aircraft are
        # ready before the first period, so they enter through A0, not C.

        for k in periods:

            model += (
                pulp.lpSum(departure_totals[period] for period in periods if period <= k)
                <= deployed_initial
                + pulp.lpSum(completions[period] for period in periods if period <= k),
                name("CumulativeFlow", scenario_id, k),
            )

        # Every charged aircraft departs exactly once; every start completes.

        model += (
            pulp.lpSum(departure_totals[k] for k in periods)
            == deployed_initial + pulp.lpSum(completions[k] for k in periods),
            name("DepartureCompletionBalance", scenario_id),
        )

        model += (
            pulp.lpSum(completions[k] for k in periods)
            == pulp.lpSum(duration_start.values()),
            name("CompletionStartBalance", scenario_id),
        )

        # Hub takeoff throughput limit per period.

        for k in periods:

            model += (
                departure_totals[k] <= max_departures,
                name("HubTakeoff", scenario_id, k),
            )

        # Realized flights reconcile with the first-stage commitment through
        # added and cancelled departures; a cancellation cannot exceed the
        # commitment for that destination-period.

        for destination in destinations:

            for k in periods:

                model += (
                    flights[destination, k]
                    == variables["n"][destination, k]
                    + added_departures[destination, k]
                    - cancelled_departures[destination, k],
                    name(
                        "CommitmentLink",
                        scenario_id,
                        destinations.index(destination),
                        k,
                    ),
                )

                model += (
                    cancelled_departures[destination, k]
                    <= variables["n"][destination, k],
                    name(
                        "CancellationLimit",
                        scenario_id,
                        destinations.index(destination),
                        k,
                    ),
                )

        # A request is either served in its admissible window or left unserved.

        for (r, destination), count in booked.items():

            served = [
                booked_passengers[r, destination, k]
                for k in structure["windows"].get(r, [])
                if (r, destination, k) in booked_passengers
            ]

            model += (
                pulp.lpSum(served) + unserved_booked[r, destination] == int(count),
                name("BookedBalance", scenario_id, r, destinations.index(destination)),
            )

        for (r, destination), count in demand.items():

            served = [
                ondemand_passengers[r, destination, k]
                for k in structure["windows"].get(r, [])
                if (r, destination, k) in ondemand_passengers
            ]

            model += (
                pulp.lpSum(served) + unserved_ondemand[r, destination] == int(count),
                name("OndemandBalance", scenario_id, r, destinations.index(destination)),
            )

        # Seats on a destination-departure are shared by booked and on-demand
        # flows; both are bucketed once by (destination, departure period).

        boarding = {}

        for (r, destination, k), variable in booked_passengers.items():
            boarding.setdefault((destination, k), []).append(variable)

        for (r, destination, k), variable in ondemand_passengers.items():
            boarding.setdefault((destination, k), []).append(variable)

        for (destination, k), variables_at_departure in boarding.items():

            model += (
                pulp.lpSum(variables_at_departure) <= capacity * flights[destination, k],
                name(
                    "SeatCapacity",
                    scenario_id,
                    destinations.index(destination),
                    k,
                ),
            )

        # Class-weighted count of unserved passengers, used by the CVaR term.

        loss_expression = (
            omega_booked
            * pulp.lpSum(
                unserved_booked[r, destination] for (r, destination) in booked
            )
            + omega_ondemand
            * pulp.lpSum(
                unserved_ondemand[r, destination] for (r, destination) in demand
            )
        )

        model += (
            variables["loss"][scenario_id] == loss_expression,
            name("ServiceLoss", scenario_id),
        )

        # Scenario profit: revenue from served passengers minus charging,
        # operating, emergency, schedule-adjustment and unserved penalties.

        revenue = pulp.lpSum(
            [
                fare[destination] * booked_passengers[r, destination, k]
                for (r, destination, k) in booked_passengers
            ]
            + [
                fare[destination] * ondemand_passengers[r, destination, k]
                for (r, destination, k) in ondemand_passengers
            ]
        )

        charging_cost = pulp.lpSum(
            groups[key]["energy"]
            * charging_cost_per_kwh(t, parameters)
            * group_charge[key, t]
            for (key, t) in group_charge
        )

        operating_cost = pulp.lpSum(
            flight_cost[destination] * flights[destination, k]
            for destination in destinations
            for k in periods
        )

        emergency_cost = pulp.lpSum(
            costs["emergency_capacity"] * emergency_capacity[t] for t in periods
        )

        adjustment_cost = pulp.lpSum(
            costs["added_departure"] * added_departures[destination, t]
            + costs["cancelled_departure"] * cancelled_departures[destination, t]
            for destination in destinations
            for t in periods
        )

        unserved_cost = (
            penalty_booked
            * pulp.lpSum(
                unserved_booked[r, destination] for (r, destination) in booked
            )
            + penalty_ondemand
            * pulp.lpSum(
                unserved_ondemand[r, destination] for (r, destination) in demand
            )
        )

        scenario_profit[scenario_id] = (
            revenue
            - charging_cost
            - operating_cost
            - emergency_cost
            - adjustment_cost
            - unserved_cost
        )

    # ========================================================
    # CVaR risk term
    # ========================================================

    zeta = pulp.LpVariable(name("CVaR_VaR"), lowBound=None, cat="Continuous")

    for scenario_id in scenarios:

        model += (
            variables["rho"][scenario_id]
            >= variables["loss"][scenario_id] - zeta,
            name("CVaRExcess", scenario_id),
        )

    probability = {
        scenario_id: float(scenario["probability"])
        for scenario_id, scenario in scenarios.items()
    }

    expected_profit = pulp.lpSum(
        probability[scenario_id] * scenario_profit[scenario_id]
        for scenario_id in scenarios
    )

    cvar = zeta + (1.0 / (1.0 - alpha)) * pulp.lpSum(
        probability[scenario_id] * variables["rho"][scenario_id]
        for scenario_id in scenarios
    )

    # ========================================================
    # First-stage cost terms
    # ========================================================

    fleet_cost = float(parameters["initial_aircraft_cost"]) * x0

    reservation_cost = pulp.lpSum(
        costs["charging_reservation"] * variables["b"][t] for t in periods
    )

    preparation_cost = pulp.lpSum(
        costs["commitment"] * variables["n"][destination, t]
        for destination in destinations
        for t in periods
    )

    # ========================================================
    # Objective: expected profit less first-stage costs and CVaR
    # ========================================================

    model += (
        expected_profit
        - fleet_cost
        - reservation_cost
        - preparation_cost
        - theta * cvar
    )

    model_data["zeta"] = zeta

    return model


__all__ = [
    "build_aggregated_model",
    "add_constraints_and_objective",
    "build_groups",
    "group_starts",
    "formulation_size",
    "GROUP_ENERGY_TOLERANCE",
]
