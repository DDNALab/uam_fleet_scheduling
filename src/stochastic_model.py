"""
Aircraft-indexed two-stage stochastic MILP for charging and departure planning.

Stage 1 commits the plan shared by every scenario: ``x0`` initially ready
aircraft, ``n[d,t]`` advance departures, ``b[t]`` reserved charging capacity
and the booked-seat protection flow ``qB[r,d,k]``. Stage 2 is the per-scenario
recourse, keeping aircraft identities: ``u[a]`` aircraft dispatched, ``x[j,t]``
charging start, ``z[a,d,k]`` flight, ``F[a]`` departure period, ``xiB``/``xiO``
passengers carried, ``etaB``/``etaO`` passengers unserved, ``bplus[t]``
emergency charging capacity, ``vplus``/``vminus`` added and cancelled
departures, ``loss[s]`` weighted service loss, and ``zeta``/``rho[s]`` the CVaR
threshold and excess. The objective maximizes expected profit minus first-stage
costs and a CVaR penalty on the service loss.

Aircraft eligibility for a demand group is projected out: an aircraft flies at
most once, so ``sum_{k in W_r} z_adk == 1`` already means "the flight to d
departs inside the admissible window W_r", and the windowed seat capacity
``xiB + xiO <= n_max * sum_{k in W_r} z_adk`` implies the capacity and
passenger timing rows together. This exact elimination removes one binary per
(aircraft, demand group) -- about 40% of the integer variables at the base case
-- and repairs a very weak relaxation.

This detailed formulation is the reference for validating the exact
reformulation in :mod:`aggregated_model` on small instances, not the
computational workhorse: retaining aircraft identities makes it far larger.
Both modules expose the same interface, so callers switch with
``formulation="indexed"``. Use ``python run_experiment.py --compare`` for
that validation: it solves the aggregated model, imposes its first-stage plan
here, and reports whether the two objectives agree.
"""

import math

import pulp

from economics import (
    charging_cost_per_kwh,
    charging_duration_periods,
    energy_price_per_kwh,
    period_hour,
)

# ``energy_price_per_kwh``, ``charging_duration_periods`` and ``period_hour``
# are re-exported deliberately: call sites import the tariff and duration
# helpers from this module, so the names are listed rather than left implicit.
__all__ = [
    "build_stochastic_model",
    "add_constraints_and_objective",
    "add_first_stage_constraints",
    "solve_model",
    "formulation_size",
    "admissible_departures",
    "feasible_starts",
    "occupancy_starts",
    "charging_duration_periods",
    "charging_periods_required",
    "energy_price_per_kwh",
    "period_hour",
]

#: Backwards-compatible alias kept for older call sites.
charging_periods_required = charging_duration_periods


# ============================================================
# Shared time-window helpers
# ============================================================
#
# Also imported by aggregated_model.py, so both formulations agree on the
# admissible service window and the feasible charging starts.

def admissible_departures(r, horizon, los):
    """Admissible departure periods ``W_r = {r, ..., min(L, r + LoS)}``."""
    r = int(r)

    return list(range(r, min(int(horizon), r + int(los)) + 1))


def feasible_starts(arrival_period, duration, horizon):
    """
    Feasible charging starts for an aircraft: ``r_js <= t`` and
    ``t + p_js <= L``.

    A start is excluded when the charging operation cannot complete before the
    final admissible departure period.
    """
    arrival_period = int(arrival_period)
    duration = int(duration)
    horizon = int(horizon)

    return [
        t
        for t in range(1, horizon + 1)
        if arrival_period <= t and t + duration <= horizon
    ]


def occupancy_starts(starts, duration, t):
    """Charging starts that keep an aircraft on a charger in period ``t``."""
    return [k for k in starts if k <= t <= k + int(duration) - 1]


def name(*parts):
    """Build a solver-safe variable/constraint name."""
    pieces = []

    for part in parts:
        text = str(part)

        cleaned = "".join(
            character if character.isalnum() or character in "._" else "_"
            for character in text
        )

        pieces.append(cleaned.strip("_") or "x")

    return "_".join(pieces)


# ============================================================
# First-stage constraints (shared by both formulations)
# ============================================================

def add_first_stage_constraints(model, model_data):
    """
    Add the scenario-independent (first-stage) constraints.

    Both formulations share this block: the projection to the aggregated model
    removes aircraft identities from the recourse stage only, so ``x0``, ``n``,
    ``b`` and the booking-protection flow ``qB`` keep exactly the same feasible
    set here. The bounds on ``x0`` and ``b`` come from the variable
    declarations, so only the rows coupling them are added here.
    """
    parameters = model_data["parameters"]
    booked_demand = model_data["booked_demand"]
    destinations = model_data["destinations"]
    periods = model_data["periods"]
    windows = model_data["structure"]["windows"]

    variables = model_data["variables"]

    n = variables["n"]
    protection = variables["protection"]

    capacity = int(parameters["evtol_capacity"])
    max_departures = int(parameters["max_departures"])

    # Hub takeoff throughput on the advance plan.
    for t in periods:

        model += (
            pulp.lpSum(n[destination, t] for destination in destinations)
            <= max_departures,
            name("PlannedDepartureCapacity", t),
        )

    # Every booked passenger is covered by protected seat capacity inside their
    # admissible window.
    for (r, destination), count in booked_demand.items():

        r = int(r)

        if int(count) <= 0 or r not in windows:
            continue

        terms = [
            protection[r, destination, k]
            for k in windows[r]
            if (r, destination, k) in protection
        ]

        model += (
            pulp.lpSum(terms) == int(count),
            name("BookingProtectionBalance", r, destinations.index(destination)),
        )

    # Protected seats never exceed the planned seat capacity.
    for destination in destinations:

        for k in periods:

            terms = [
                protection[r, destination, k]
                for r in periods
                if k in windows.get(r, []) and (r, destination, k) in protection
            ]

            if not terms:
                continue

            model += (
                pulp.lpSum(terms) <= capacity * n[destination, k],
                name("ProtectedSeatCapacity", destinations.index(destination), k),
            )


# ============================================================
# Model construction
# ============================================================

def build_stochastic_model(model_input):
    """
    Declare the first-stage and per-scenario variables and the structural index
    sets.

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
    capacity = int(parameters["evtol_capacity"])

    model = pulp.LpProblem(
        name("TwoStage_eVTOL_Indexed"),
        pulp.LpMaximize,
    )

    # ---- admissible service windows W_r -------------------------------
    windows = {r: admissible_departures(r, horizon, los) for r in periods}

    # ========================================================
    # FIRST STAGE
    # ========================================================

    x0 = pulp.LpVariable(
        name("InitialFleet"),
        lowBound=0,
        upBound=max_fleet,
        cat="Integer",
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

    # ---- booking protection flow qB -----------------------------------
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
        "u": {},
        "charging": {},
        "flights": {},
        "departure_period": {},
        "eligibility": {},
        "booked_passengers": {},
        "ondemand_passengers": {},
        "unserved_booked": {},
        "unserved_ondemand": {},
        "emergency_capacity": {},
        "added_departures": {},
        "cancelled_departures": {},
        "loss": {},
        "rho": {},
    }

    structure = {
        "windows": windows,
        "scenarios": {},
        "initial_pool": [name("init", i + 1) for i in range(max_fleet)],
    }

    for scenario_id, scenario in scenarios.items():

        aircraft = scenario["aircraft"]
        ondemand = scenario["ondemand_demand"]

        incoming = [name("inc", j) for j in sorted(aircraft)]

        initial = list(structure["initial_pool"])

        all_aircraft = initial + incoming

        # ---- per-aircraft data ------------------------------------
        ready = {a: 1 for a in initial}

        duration = {}

        energy = {}

        starts = {}

        for a, j in zip(incoming, sorted(aircraft)):

            info = aircraft[j]

            duration[a] = int(info["duration"])

            energy[a] = float(info["energy"])

            ready[a] = int(info["arrival_period"]) + duration[a]

            starts[a] = feasible_starts(
                info["arrival_period"], duration[a], horizon
            )

        # ---- demand groups with positive mass in this scenario -----
        groups = sorted(
            {
                (int(r), destination)
                for (r, destination), value in booked_demand.items()
                if int(value) > 0
            }
            | {
                (int(r), destination)
                for (r, destination), value in ondemand.items()
                if int(value) > 0
            }
        )

        # ---- variables --------------------------------------------
        u = {
            a: pulp.LpVariable(
                name("Dispatch", scenario_id, a), lowBound=0, upBound=1, cat="Binary"
            )
            for a in all_aircraft
        }

        # ---- flights ----------------------------------------------
        #
        # Flights start at ready[a]. Earlier columns are dropped, but a ready
        # aircraft may still fly empty to honour a planned departure.
        flights = {}

        for a in all_aircraft:
            for destination in destinations:
                for k in periods:
                    if k < ready[a]:
                        continue
                    flights[a, destination, k] = pulp.LpVariable(
                        name(
                            "Flight", scenario_id, a, destinations.index(destination), k
                        ),
                        lowBound=0,
                        upBound=1,
                        cat="Binary",
                    )

        charging = {
            (a, t): pulp.LpVariable(
                name("Charge", scenario_id, a, t), lowBound=0, upBound=1, cat="Binary"
            )
            for a in incoming
            for t in starts[a]
        }

        departure_period = {
            a: pulp.LpVariable(
                name("DeparturePeriod", scenario_id, a),
                lowBound=0,
                upBound=horizon,
                cat="Continuous",
            )
            for a in all_aircraft
        }

        # ---- passenger flows --------------------------------------
        #
        # Aircraft eligibility is projected out: an aircraft flies at most once,
        # so seats are committed exactly when sum_{k in W_r} z_adk == 1.
        booked_passengers = {}

        ondemand_passengers = {}

        has_booked = {
            (int(r), destination): int(value)
            for (r, destination), value in booked_demand.items()
            if int(value) > 0
        }

        has_ondemand = {
            (int(r), destination): int(value)
            for (r, destination), value in ondemand.items()
            if int(value) > 0
        }

        # ---- service feasibility: which columns can be non-zero ----
        #
        # Aircraft a serves group (r, d) only if some k in W_r has k >= ready[a].
        # Other columns are structurally zero and would only weaken the relaxation.
        service_slots = {}

        for a in all_aircraft:

            for (r, destination) in groups:

                slots = [k for k in windows.get(r, []) if k >= ready[a]]

                if slots:
                    service_slots[a, r, destination] = slots

        # ---- integer status of the passenger subsystem -------------
        #
        # With flights fixed, the flow system is a transportation problem with
        # integral data, so continuous passenger flows are integral without loss.
        for (a, r, destination) in service_slots:

            # Tight implied bounds: an aircraft carries at most ``capacity``
            # passengers, and each class is capped by its own request count.
            booked_count = has_booked.get((r, destination), 0)

            ondemand_count = has_ondemand.get((r, destination), 0)

            if booked_count > 0:
                booked_passengers[a, r, destination] = pulp.LpVariable(
                    name(
                        "BookedPax",
                        scenario_id,
                        a,
                        r,
                        destinations.index(destination),
                    ),
                    lowBound=0,
                    upBound=min(capacity, booked_count),
                    cat="Continuous",
                )

            if ondemand_count > 0:
                ondemand_passengers[a, r, destination] = pulp.LpVariable(
                    name(
                        "OndemandPax",
                        scenario_id,
                        a,
                        r,
                        destinations.index(destination),
                    ),
                    lowBound=0,
                    upBound=min(capacity, ondemand_count),
                    cat="Continuous",
                )

        unserved_booked = {
            (r, destination): pulp.LpVariable(
                name("UnservedBooked", scenario_id, r, destinations.index(destination)),
                lowBound=0,
                upBound=value,
                cat="Continuous",
            )
            for (r, destination), value in has_booked.items()
        }

        unserved_ondemand = {
            (r, destination): pulp.LpVariable(
                name(
                    "UnservedOndemand",
                    scenario_id,
                    r,
                    destinations.index(destination),
                ),
                lowBound=0,
                upBound=value,
                cat="Continuous",
            )
            for (r, destination), value in has_ondemand.items()
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
            name("ServiceLoss", scenario_id),
            lowBound=0,
            cat="Continuous",
        )

        rho = pulp.LpVariable(
            name("CVaRExcess", scenario_id),
            lowBound=0,
            cat="Continuous",
        )

        # ---- register ---------------------------------------------
        variables["u"][scenario_id] = u
        variables["charging"][scenario_id] = charging
        variables["flights"][scenario_id] = flights
        variables["departure_period"][scenario_id] = departure_period
        variables["booked_passengers"][scenario_id] = booked_passengers
        variables["ondemand_passengers"][scenario_id] = ondemand_passengers
        variables["unserved_booked"][scenario_id] = unserved_booked
        variables["unserved_ondemand"][scenario_id] = unserved_ondemand
        variables["emergency_capacity"][scenario_id] = emergency_capacity
        variables["added_departures"][scenario_id] = added_departures
        variables["cancelled_departures"][scenario_id] = cancelled_departures
        variables["loss"][scenario_id] = loss
        variables["rho"][scenario_id] = rho

        structure["scenarios"][scenario_id] = {
            "initial": initial,
            "incoming": incoming,
            "all": all_aircraft,
            "ready": ready,
            "duration": duration,
            "energy": energy,
            "starts": starts,
            "groups": groups,
            "booked": has_booked,
            "ondemand": has_ondemand,
            "service_slots": service_slots,
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
        "formulation": "indexed",
    }


# ============================================================
# Constraints and objective
# ============================================================

def add_constraints_and_objective(model_data):
    """
    Add all recourse constraints and the expected-profit / CVaR objective.

    Returns the populated :class:`pulp.LpProblem`.
    """
    model = model_data["model"]
    parameters = model_data["parameters"]
    scenarios = model_data["scenarios"]
    destinations = model_data["destinations"]
    periods = model_data["periods"]

    variables = model_data["variables"]
    structure = model_data["structure"]

    capacity = int(parameters["evtol_capacity"])
    facilities = int(parameters["charging_facilities"])
    max_departures = int(parameters["max_departures"])

    # Admissible service windows W_r, shared with the first-stage constraints
    # and with the windowed seat capacity below.
    windows = structure["windows"]

    x0 = variables["x0"]
    n = variables["n"]
    b = variables["b"]

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
    # FIRST-STAGE CONSTRAINTS
    # ========================================================

    add_first_stage_constraints(model, model_data)

    # ========================================================
    # SCENARIO RECOURSE
    # ========================================================

    scenario_profit = {}

    for scenario_id, scenario in scenarios.items():

        info = structure["scenarios"][scenario_id]

        u = variables["u"][scenario_id]
        charging = variables["charging"][scenario_id]
        flights = variables["flights"][scenario_id]
        departure_period = variables["departure_period"][scenario_id]
        booked_passengers = variables["booked_passengers"][scenario_id]
        ondemand_passengers = variables["ondemand_passengers"][scenario_id]
        unserved_booked = variables["unserved_booked"][scenario_id]
        unserved_ondemand = variables["unserved_ondemand"][scenario_id]
        emergency_capacity = variables["emergency_capacity"][scenario_id]
        added_departures = variables["added_departures"][scenario_id]
        cancelled_departures = variables["cancelled_departures"][scenario_id]

        incoming = info["incoming"]
        all_aircraft = info["all"]
        duration = info["duration"]
        energy = info["energy"]
        starts = info["starts"]
        booked = info["booked"]
        ondemand = info["ondemand"]

        # Initial pool deployment is limited by the planned fleet x0.

        if info["initial"]:
            model += (
                pulp.lpSum(u[a] for a in info["initial"]) <= x0,
                name("InitialFleetLimit", scenario_id),
            )

            # Symmetry breaking: the initial-ready aircraft are interchangeable, so
            # order their deployment indicators (and the departures by position) to
            # avoid enumerating permutations of the same plan.
            #
            position = {
                (destination, k): destinations.index(destination) * len(periods) + k
                for destination in destinations
                for k in periods
            }

            span = float(len(destinations) * len(periods))

            def departure_position(a):
                return pulp.lpSum(
                    weight * flights[a, destination, k]
                    for (destination, k), weight in position.items()
                    if (a, destination, k) in flights
                )

            for first, second in zip(info["initial"][:-1], info["initial"][1:]):

                model += (
                    u[first] >= u[second],
                    name("InitialFleetSymmetry", scenario_id, first),
                )

                model += (
                    departure_position(first)
                    <= departure_position(second)
                    + span * (1 - u[second]),
                    name("InitialDepartureSymmetry", scenario_id, first),
                )

        # One charging start per activated incoming aircraft.

        for a in incoming:

            model += (
                pulp.lpSum(charging[a, t] for t in starts[a]) == u[a],
                name("ChargingAssignment", scenario_id, a),
            )

        # Charger occupancy per period, and emergency capacity on top of b[t].

        for t in periods:

            occupying = []

            for a in incoming:

                for k in occupancy_starts(starts[a], duration[a], t):
                    occupying.append(charging[a, k])

            model += (
                pulp.lpSum(occupying) <= b[t] + emergency_capacity[t],
                name("ChargerOccupancy", scenario_id, t),
            )

            model += (
                emergency_capacity[t] <= facilities - b[t],
                name("EmergencyCapacityLimit", scenario_id, t),
            )

        # One departure per deployed aircraft, and its departure period F_a.

        for a in all_aircraft:

            model += (
                pulp.lpSum(
                    flights[a, destination, k]
                    for destination in destinations
                    for k in periods
                    if (a, destination, k) in flights
                )
                == u[a],
                name("OneDeparture", scenario_id, a),
            )

            model += (
                departure_period[a]
                == pulp.lpSum(
                    k * flights[a, destination, k]
                    for destination in destinations
                    for k in periods
                    if (a, destination, k) in flights
                ),
                name("DeparturePeriod", scenario_id, a),
            )

        # Departure is linked to charging completion: F_a >= t + p_a.

        for a in incoming:

            model += (
                departure_period[a]
                >= pulp.lpSum(
                    (t + duration[a]) * charging[a, t] for t in starts[a]
                ),
                name("ChargeCompletesBeforeDeparture", scenario_id, a),
            )

            # Per-start cut implied by the two rows above for integral schedules,
            # but tighter: without it the LP relaxation can depart before charging
            # finishes.
            for t in starts[a]:

                model += (
                    pulp.lpSum(
                        flights[a, destination, k]
                        for destination in destinations
                        for k in periods
                        if (a, destination, k) in flights and k < t + duration[a]
                    )
                    <= 1 - charging[a, t],
                    name("ChargeStartPrecedesDeparture", scenario_id, a, t),
                )

        # Hub takeoff throughput per period.

        for t in periods:

            model += (
                pulp.lpSum(
                    flights[a, destination, t]
                    for a in all_aircraft
                    for destination in destinations
                    if (a, destination, t) in flights
                )
                <= max_departures,
                name("HubTakeoff", scenario_id, t),
            )

        # Reconcile realized flights with the advance plan: added and
        # cancelled departures make up the difference.

        for destination in destinations:

            for t in periods:

                actual = pulp.lpSum(
                    flights[a, destination, t]
                    for a in all_aircraft
                    if (a, destination, t) in flights
                )

                model += (
                    actual
                    == n[destination, t]
                    + added_departures[destination, t]
                    - cancelled_departures[destination, t],
                    name("CommitmentLink", scenario_id, destinations.index(destination), t),
                )

                model += (
                    cancelled_departures[destination, t] <= n[destination, t],
                    name(
                        "CancellationLimit",
                        scenario_id,
                        destinations.index(destination),
                        t,
                    ),
                )

        # Booked and on-demand passengers share the seats of one aircraft but are
        # tracked separately; capacity is written against the flights inside W_r.

        passengers_by_aircraft = {a: [] for a in all_aircraft}

        passengers_by_group = {}

        for (a, r, destination), variable in booked_passengers.items():
            passengers_by_aircraft[a].append(variable)
            passengers_by_group.setdefault((a, r, destination), []).append(variable)

        for (a, r, destination), variable in ondemand_passengers.items():
            passengers_by_aircraft[a].append(variable)
            passengers_by_group.setdefault((a, r, destination), []).append(variable)

        for a in all_aircraft:

            model += (
                pulp.lpSum(passengers_by_aircraft[a]) <= capacity * u[a],
                name("AircraftSeatCapacity", scenario_id, a),
            )

        for (a, r, destination) in passengers_by_group:

            in_window = [
                flights[a, destination, k]
                for k in windows.get(r, [])
                if (a, destination, k) in flights
            ]

            model += (
                pulp.lpSum(passengers_by_group[a, r, destination])
                <= capacity * pulp.lpSum(in_window),
                name("GroupSeatCapacity", scenario_id, a, r, destination),
            )

        # Class demand conservation: served plus unserved equals the requests.

        for (r, destination), count in booked.items():

            model += (
                pulp.lpSum(
                    booked_passengers[a, r, destination]
                    for a in all_aircraft
                    if (a, r, destination) in booked_passengers
                )
                + unserved_booked[r, destination]
                == int(count),
                name("BookedBalance", scenario_id, r, destinations.index(destination)),
            )

        for (r, destination), count in ondemand.items():

            model += (
                pulp.lpSum(
                    ondemand_passengers[a, r, destination]
                    for a in all_aircraft
                    if (a, r, destination) in ondemand_passengers
                )
                + unserved_ondemand[r, destination]
                == int(count),
                name("OndemandBalance", scenario_id, r, destinations.index(destination)),
            )

        # Passenger timing is implied by the windowed seat capacity above:
        # F_a = sum_k k z_adk, and a positive flow requires a departure with k
        # in W_r = [r, r + LoS]. No row is added for it.

        # Class-weighted service loss.

        loss_expression = (
            omega_booked
            * pulp.lpSum(
                unserved_booked[r, destination]
                for (r, destination) in booked
            )
            + omega_ondemand
            * pulp.lpSum(
                unserved_ondemand[r, destination]
                for (r, destination) in ondemand
            )
        )

        model += (
            variables["loss"][scenario_id] == loss_expression,
            name("ServiceLoss", scenario_id),
        )

        # Scenario profit: revenue minus every scenario cost.

        revenue = pulp.lpSum(
            [
                fare[destination] * booked_passengers[a, r, destination]
                for (a, r, destination) in booked_passengers
            ]
            + [
                fare[destination] * ondemand_passengers[a, r, destination]
                for (a, r, destination) in ondemand_passengers
            ]
        )

        charging_cost = pulp.lpSum(
            energy[a] * charging_cost_per_kwh(t, parameters) * charging[a, t]
            for a in incoming
            for t in starts[a]
        )

        operating_cost = pulp.lpSum(
            flight_cost[destination] * flights[a, destination, k]
            for (a, destination, k) in flights
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
                unserved_ondemand[r, destination] for (r, destination) in ondemand
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
    # CVaR
    # ========================================================
    #
    # zeta is the VaR threshold and is free in sign; rho[s] is the excess loss
    # of scenario s above it, weighted by 1 / (1 - alpha).

    zeta = pulp.LpVariable(
        name("CVaR_VaR"), lowBound=None, cat="Continuous"
    )

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
        costs["charging_reservation"] * b[t] for t in periods
    )

    preparation_cost = pulp.lpSum(
        costs["commitment"] * n[destination, t]
        for destination in destinations
        for t in periods
    )

    # ========================================================
    # Objective
    # ========================================================
    #
    # Expected profit minus first-stage costs and the CVaR penalty.

    model += (
        expected_profit
        - fleet_cost
        - reservation_cost
        - preparation_cost
        - theta * cvar
    )

    model_data["zeta"] = zeta

    return model


# ============================================================
# Size diagnostics
# ============================================================

def formulation_size(model):
    """
    Pre-presolve variable, integer, binary and constraint counts.

    PuLP normalizes ``cat="Binary"`` to ``cat="Integer"`` with bounds [0, 1],
    so binaries are counted by their unit bounds rather than by ``.cat``.
    These absolute counts, paired with matched solver results, are what show a
    realized saving rather than an assumed one.
    """
    variables = model.variables()

    integer = 0
    binary = 0
    continuous = 0

    for variable in variables:

        is_integer = variable.cat == pulp.LpInteger

        unit_bounds = (
            variable.lowBound is not None
            and variable.upBound is not None
            and abs(float(variable.lowBound)) < 1e-12
            and abs(float(variable.upBound) - 1.0) < 1e-12
        )

        if is_integer and unit_bounds:
            binary += 1
            integer += 1
        elif is_integer:
            integer += 1
        else:
            continuous += 1

    constraints = sum(
        len(constraint) if isinstance(constraint, list) else 1
        for constraint in model.constraints.values()
    )

    return {
        "variables": len(variables),
        "integer_variables": integer,
        "binary_variables": binary,
        "continuous_variables": continuous,
        "constraints": constraints,
    }


# ============================================================
# Solver
# ============================================================

#: Solver preference order used when ``solver="auto"``.
SOLVER_PREFERENCE = ("GUROBI", "HIGHS", "CBC")

#: Above this many integer variables a 5%-gap solve in one hour is unlikely on
#: an open-source branch-and-bound solver, so the run is flagged before it
#: silently burns its budget and returns an incumbent.
LARGE_MODEL_INTEGER_THRESHOLD = 20_000


def size_advisory(size):
    """
    Pre-solve warning for models that will almost certainly time out.

    The aircraft-indexed formulation grows with
    ``|I0| * |D| * T + |J_s| * |D| * T`` binaries per scenario, so a generous
    fleet cap makes it intractable long before the aggregated formulation is
    stressed; the two are only comparable on small instances. Saying so up
    front is cheaper than discovering it from a 0%-gap incumbent.

    Returns ``None`` when the model is of a plausible size.
    """
    integer = int(size.get("integer_variables", 0))
    binary = int(size.get("binary_variables", 0))

    if integer <= LARGE_MODEL_INTEGER_THRESHOLD:
        return None

    return (
        "this model has %d integer variables (%d binary). That is far above the "
        "~%d at which a 5%%-gap solve in the Table 2 budget is realistic for an "
        "open-source solver, so expect a time-limited incumbent rather than a "
        "proved optimum. Its objective is then NOT comparable with a solved "
        "formulation -- use --compare to check equality properly, "
        "--formulation aggregated to get a policy, or shrink the instance with "
        "--max-initial-fleet / --n-periods / --n-destinations."
        % (integer, binary, LARGE_MODEL_INTEGER_THRESHOLD)
    )


def _make_solver(
    solver_name,
    model,
    time_limit,
    gap_rel,
    threads,
    warm_start,
    mip_focus,
    cuts,
    presolve,
    log_path,
    verbose,
):
    """
    Instantiate one PuLP solver, or return ``(None, reason)`` if unavailable.

    Gurobi's trial licence caps the model at 2000 variables and 2000
    constraints, so ``available()`` returning True is not enough: the model
    size is checked first, rather than letting the solve die inside gurobipy.
    """
    solver_name = solver_name.upper()

    if solver_name == "GUROBI":

        solver = pulp.GUROBI(
            msg=verbose,
            timeLimit=time_limit,
            gapRel=gap_rel,
            warmStart=warm_start,
            logPath=log_path,
            MIPFocus=mip_focus,
            Cuts=cuts,
            Presolve=presolve,
        )

        if not solver.available():
            return None, "gurobipy unavailable or not licensed"

        size = formulation_size(model)

        if size["variables"] > 2000 or size["constraints"] > 2000:
            return None, (
                "size-limited licence: model has %d variables / %d constraints, "
                "trial cap is 2000 each" % (size["variables"], size["constraints"])
            )

        if threads:
            solver.solver_params["Threads"] = threads

        return solver, None

    if solver_name == "HIGHS":

        solver = pulp.HiGHS(
            msg=verbose,
            timeLimit=time_limit,
            gapRel=gap_rel,
            threads=threads or None,
            logPath=log_path,
        )

        if not solver.available():
            return None, "highspy unavailable"

        return solver, None

    if solver_name == "CBC":

        solver = pulp.PULP_CBC_CMD(
            msg=verbose,
            timeLimit=time_limit,
            gapRel=gap_rel,
            threads=threads or None,
        )

        if not solver.available():
            return None, "cbc unavailable"

        return solver, None

    return None, "unknown solver %r" % solver_name


def solve_model(
    model,
    time_limit=600,
    gap_rel=0.02,
    threads=0,
    warm_start=False,
    mip_focus=1,
    cuts=2,
    presolve=2,
    log_path=None,
    verbose=True,
    solver="auto",
):
    """
    Solve the two-stage stochastic MILP.

    Parameters
    ----------
    model : pulp.LpProblem
    solver : str
        ``"auto"`` walks :data:`SOLVER_PREFERENCE`; pass ``"gurobi"``,
        ``"highs"`` or ``"cbc"`` to force one.
    time_limit : float
        Seconds before returning the best solution found.
    gap_rel : float
        Relative MIP gap tolerance the solver is asked to reach (default 2%).
    threads : int
        0 lets the solver choose.
    warm_start, mip_focus, cuts, presolve : Gurobi-only tuning.
    log_path : str, optional
    verbose : bool

    Returns
    -------
    dict
        ``status``, ``objective``, ``solver``, ``proven_optimal``, ``bound``,
        ``gap`` and ``size``.

        ``status`` is ``Optimal``, ``GapReached(...)``, ``TimeLimit``,
        ``NotProven(...)``, ``NotVerified``, ``NoSolution``, ``Infeasible`` or
        ``Unbounded``. Only ``Optimal`` carries a proved-optimality claim;
        every other status means the objective is an incumbent or undefined.
    """
    # Requested solver, normalized once. ``_make_solver`` upper-cases locally, so
    # the HiGHS status check below must not depend on the caller's casing.
    requested = (solver or "auto").strip().lower()

    candidates = (
        SOLVER_PREFERENCE if requested == "auto" else (requested.upper(),)
    )

    chosen = None

    skip_reasons = []

    for solver_name in candidates:

        instance, reason = _make_solver(
            solver_name,
            model,
            time_limit,
            gap_rel,
            threads,
            warm_start,
            mip_focus,
            cuts,
            presolve,
            log_path,
            verbose,
        )

        if instance is not None:
            chosen = (solver_name, instance)
            break

        skip_reasons.append("%s: %s" % (solver_name, reason))

        print("[info] skipping %s -- %s" % (solver_name, reason))

    if chosen is None:
        raise RuntimeError(
            "No usable MILP solver found. Tried: %s" % "; ".join(skip_reasons)
        )

    solver_name, solver_instance = chosen

    size = formulation_size(model)

    advisory = size_advisory(size)

    if advisory:
        print("[warning] %s" % advisory)

    print("[info] solving with %s" % solver_name)

    model.solve(solver_instance)

    status = pulp.LpStatus[model.status]

    proven_optimal = status == "Optimal"

    bound = None

    gap = None

    has_solution = True

    if solver_name == "HIGHS":

        try:

            import highspy

            highs = model.solverModel

            model_status = highs.getModelStatus()

            info = highs.getInfo()

            # HiGHS minimizes, so PuLP negates our objective; ``mip_dual_bound``
            # is the bound and must be negated back into our sign convention.
            try:
                dual = float(info.mip_dual_bound)
                bound = -dual if math.isfinite(dual) else None
            except Exception:
                bound = None

            try:
                reported_gap = float(info.mip_gap)
                gap = reported_gap if math.isfinite(reported_gap) else None
            except Exception:
                gap = None

            # ``primal_solution_status``: 0 = none, 1 = infeasible point, 2 =
            # feasible. With no incumbent there is no objective to report.
            try:
                has_solution = int(info.primal_solution_status) == 2
            except Exception:
                has_solution = True

            # PuLP/HiGHS report ``kOptimal`` whenever the solver stops *inside*
            # the requested relative gap, not only at a zero gap, so the two
            # cases are reported separately instead of both being "Optimal".
            tolerance = max(float(gap_rel), 1e-9)

            inside_gap = gap is not None and 1e-9 < gap <= tolerance

            if not has_solution:
                status = "NoSolution"
                proven_optimal = False
            elif model_status == highspy.HighsModelStatus.kTimeLimit:
                status = "TimeLimit"
                proven_optimal = False
            elif model_status == highspy.HighsModelStatus.kOptimal and not inside_gap:
                status = "Optimal"
                proven_optimal = True
            elif model_status == highspy.HighsModelStatus.kOptimal:
                status = "GapReached(%.2f%%)" % (100.0 * gap)
                proven_optimal = False
            else:
                status = "NotProven(%s)" % model_status.name
                proven_optimal = False

        except Exception as error:  # pragma: no cover - introspection is best effort

            # Never claim optimality that could not be verified.
            proven_optimal = False

            has_solution = True

            status = "NotVerified"

            print(
                "[warning] could not verify the HiGHS solve status (%s); "
                "reporting the run as NOT proven optimal." % error
            )

    elif solver_name == "CBC":

        # PuLP reports LpStatusOptimal as soon as CBC hands back a feasible
        # incumbent -- including when CBC stopped on its time limit or gap.
        # model.sol_status distinguishes a proved optimum from an incumbent.
        solution_status = getattr(model, "sol_status", None)

        if solution_status is None:

            proven_optimal = False

            status = "NotVerified"

            print(
                "[warning] CBC did not report a solution status; reporting the "
                "run as NOT proven optimal."
            )

        elif solution_status != pulp.LpSolutionOptimal:

            proven_optimal = False

            status = "NotProven(%s)" % pulp.LpSolution.get(
                solution_status, str(solution_status)
            )

    if has_solution:

        try:
            objective = pulp.value(model.objective)
        except Exception:  # pragma: no cover
            objective = None

    else:

        # No incumbent: there is no objective value. Reporting 0.0 (what
        # ``pulp.value`` yields with unset variables) would look like a very
        # poor policy rather than an unsolved model.
        objective = None

    if not proven_optimal:

        if status in {"Infeasible", "Unbounded"}:

            print(
                "\n[warning] %s reported the model as %s.\n"
                "          For the two-stage program this usually means the accepted "
                "booking pattern cannot be protected within the hub takeoff "
                "throughput -- a structural property of the booking pattern, not of "
                "the operating policy. Check it with "
                "feasibility.screen_booking_share before "
                "interpreting the run.\n"
                % (solver_name, status)
            )

        elif gap is not None and gap <= max(float(gap_rel), 1e-9):

            # Stopped inside the requested gap: that meets the budget, not a
            # failure, so say so plainly rather than warning.
            print(
                "[info] %s stopped inside the requested gap: achieved %.2f%% "
                "(target %.2f%%). The solution meets the budget but is not a "
                "proved optimum."
                % (solver_name, 100.0 * gap, 100.0 * gap_rel)
            )

        else:

            evidence = []

            if gap is not None:
                evidence.append("achieved MIP gap %.2f%% (target %.2f%%)"
                                % (100.0 * gap, 100.0 * gap_rel))

            if bound is not None:
                evidence.append("dual bound %.4f" % bound)

            print(
                "\n[warning] %s DID NOT SOLVE the model (%s%s).\n"
                "          The objective below is a time-limited incumbent and must "
                "NOT be compared head-to-head with a solved formulation. Obtain a "
                "proved optimum with a longer time_limit, or shrink the instance "
                "and/or use --formulation aggregated.\n"
                % (
                    solver_name,
                    status,
                    (": " + ", ".join(evidence)) if evidence else "",
                )
            )

            if not has_solution:
                print(
                    "          No feasible solution was found within the budget, so "
                    "the objective is undefined (None) rather than a poor value.\n"
                )

    return {
        "status": status,
        "objective": objective,
        "solver": solver_name,
        "proven_optimal": proven_optimal,
        "bound": bound,
        "gap": gap,
        "size": size,
    }
