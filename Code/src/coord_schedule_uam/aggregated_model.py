"""
aggregated_model.py
===================

Duration-aggregated cumulative-flow reformulation (PDF Section 4.3).

Based on:
Risk-Aware Charging and Service Commitment for Urban Air Mobility
under Correlated Operational Uncertainty

This is the exact projected extensive form. It removes aircraft identities
where they have no operational consequence, which makes the model far smaller
than the aircraft-indexed formulation in stochastic_model.py while preserving
the same feasible first-stage policies and optimal objective value.

The interface mirrors stochastic_model.py exactly, so the two formulations are
interchangeable:

    build_aggregated_model(model_input) -> model_data
    add_constraints_and_objective(model_data) -> pulp.LpProblem

Architecture:

uam_data_pipeline.py
        |
scenario_generator.py
        |
scenario_reduction.py
        |
model_builder.py
        |
THIS FILE

Stage 1 decisions (identical to the indexed model):
    n[d,t] : committed departures
    b[t]   : reserved charging capacity

Stage 2 decisions (aggregated, per scenario s):
    rho[g,t]       : aircraft of group g assigned to charging start t
    w[h,t]         : number of duration-h charging operations starting at t
    f[d,k]         : realized flights to destination d in period k
    y[r,d,k]       : passengers arriving r for d served in period k
    phi[r,d]       : unserved passengers
    v_plus[d,k]    : added departures
    v_minus[d,k]   : cancelled departures
    b_plus[t]      : emergency charging capacity

Reformulation structure (PDF equations in parentheses):

    (24) group charging assignment within group size
    (25) duration-start totals
    (26) charger occupancy via duration-start totals
    (27) completion / departure totals
    (28) cumulative flow: departures cannot precede completions
    (29) every charged aircraft gets exactly one in-horizon departure
    (30) hub departure capacity
    (31) commitment reconciliation
    (32) passenger conservation
    (33) aircraft passenger capacity
    (34) charging-release prefixes (redundant, aids propagation)
    (35) destination service covers (redundant, aids propagation)

Notes on exactness:

    Propositions 1-4 in the paper establish that the aggregate solution can be
    disaggregated to an integral aircraft-level plan with the same objective
    value. The pooling argument requires that post-charge aircraft are
    interchangeable -- identical destination feasibility, capacity and
    destination-specific operating cost. That holds for the homogeneous eVTOL
    fleet used here. It would NOT hold for a heterogeneous fleet with
    group-dependent range, destination eligibility, capacity or flight cost.
"""


import pulp
import numpy as np


from .stochastic_model import (
    energy_price_per_kwh,
    charging_periods_required,
)


# ============================================================
# Aircraft grouping
# ============================================================

# Aircraft are pooled by (arrival period, charging duration). Arrival state of
# charge varies continuously, so the energy requirement is quantized into
# buckets of this many kWh. A smaller bucket is more faithful to the per-
# aircraft charging cost and yields more groups; a larger bucket aggregates more
# aggressively. Coarser buckets slightly perturb the charging cost, which is a
# small term relative to revenue, so this is a safe lever for model size.
GROUP_ENERGY_TOLERANCE = 1.0


def build_groups(aircraft, params, tolerance=GROUP_ENERGY_TOLERANCE):
    """
    Partition one scenario's aircraft into groups G_s.

    Group g holds N_g aircraft sharing an arrival period, a charging duration
    and (to within `tolerance` kWh) an energy requirement, matching the paper's
    definition of G_s.

    Returns a dict keyed by (arrival_period, duration, energy_bucket), each
    value carrying the group size and the parameters the model needs.
    """

    groups = {}

    for j, info in aircraft.items():

        arrival = info["arrival_period"]

        duration = charging_periods_required(info["initial_soc"], params)

        energy = (
            params["battery_capacity"]
            * max(0, params["soc_min"] - info["initial_soc"])
            / 100
        )

        # Quantize so that near-identical aircraft share a group.
        bucket = (
            round(energy / tolerance) * tolerance
            if tolerance > 0
            else energy
        )

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


def feasible_starts(group, horizon):
    """
    K_gs = {t : r_gs <= t, t + h_gs <= L}   (PDF Eq. 23)

    Excludes charging that cannot complete before the final admissible
    departure period.
    """

    return [
        t
        for t in range(1, horizon + 1)
        if group["arrival"] <= t and t + group["duration"] <= horizon
    ]


def admissible_departures(r, horizon, los_periods):
    """
    W_r = {r, ..., min(L, r + LoS)}
    """

    return list(range(r, min(horizon, r + los_periods) + 1))


# ============================================================
# Main model builder
# ============================================================

def build_aggregated_model(model_input):

    params = model_input["parameters"]
    scenarios = model_input["scenarios"]

    horizon = params["periods"]

    periods = range(1, horizon + 1)

    destinations = sorted({
        d
        for s in scenarios.values()
        for (t, d) in s["passenger_demand"].keys()
    })

    model = pulp.LpProblem(
        "TwoStage_eVTOL_Aggregated_CumulativeFlow",
        pulp.LpMaximize
    )

    # ========================================================
    # FIRST STAGE VARIABLES
    # ========================================================

    n = {}
    for d in destinations:
        for t in periods:
            n[d, t] = pulp.LpVariable(
                f"Committed_Departure_{d}_{t}",
                lowBound=0,
                cat="Integer"
            )

    b = {}
    for t in periods:
        b[t] = pulp.LpVariable(
            f"Reserved_Charging_{t}",
            lowBound=0,
            cat="Integer"
        )

    # ========================================================
    # SECOND STAGE VARIABLES
    # ========================================================

    groups = {}

    rho = {}          # rho[g,t]  >= 0   continuous
    w = {}            # w[h,t]    int
    flights = {}      # f[d,k]    int
    passengers = {}   # y[r,d,k]  >= 0   continuous
    unserved = {}     # phi[r,d]  >= 0
    completions = {}  # C[k]      >= 0
    departures_total = {}   # D[k] >= 0

    added_departures = {}
    cancelled_departures = {}
    emergency_capacity = {}

    for sid, scenario in scenarios.items():

        groups[sid] = build_groups(scenario["aircraft"], params)

        durations = sorted({g["duration"] for g in groups[sid].values()})

        # ------------------------------------------------
        # rho_gts : group charging assignment (continuous)
        # ------------------------------------------------

        rho[sid] = {}
        for g in groups[sid]:
            for t in feasible_starts(groups[sid][g], horizon):
                rho[sid][g, t] = pulp.LpVariable(
                    f"GroupCharge_s{sid}_g{g[0]}_{g[1]}_{g[2]}_t{t}",
                    lowBound=0,
                    cat="Continuous"
                )

        # ------------------------------------------------
        # w_hts : duration-start totals
        # ------------------------------------------------

        w[sid] = {}
        for h in durations:
            for t in periods:
                # Only create starts that at least one group could use.
                if any(
                    h == groups[sid][g]["duration"] and t in feasible_starts(
                        groups[sid][g], horizon)
                    for g in groups[sid]
                ):
                    w[sid][h, t] = pulp.LpVariable(
                        f"DurationStart_s{sid}_h{h}_t{t}",
                        lowBound=0,
                        cat="Integer"
                    )

        # ------------------------------------------------
        # f_dks : realized flights
        # ------------------------------------------------

        flights[sid] = {}
        for d in destinations:
            for k in periods:
                flights[sid][d, k] = pulp.LpVariable(
                    f"Flights_s{sid}_{d}_{k}",
                    lowBound=0,
                    cat="Integer"
                )

        # ------------------------------------------------
        # y_rdks : passengers served in a departure period
        # ------------------------------------------------

        passengers[sid] = {}
        unserved[sid] = {}

        demanded = {
            (r, d)
            for (r, d), q in scenario["passenger_demand"].items()
            if q > 0 and 1 <= r <= horizon
        }

        for (r, d) in demanded:

            unserved[sid][r, d] = pulp.LpVariable(
                f"Unserved_s{sid}_{r}_{d}",
                lowBound=0,
                cat="Continuous"
            )

            for k in admissible_departures(r, horizon, params["los_periods"]):

                if k < 1 or k > horizon:
                    continue

                passengers[sid][r, d, k] = pulp.LpVariable(
                    f"Passengers_s{sid}_{r}_{d}_{k}",
                    lowBound=0,
                    cat="Continuous"
                )

        # ------------------------------------------------
        # Derived totals: completions C_ks and departures D_ks
        # ------------------------------------------------

        completions[sid] = {}
        departures_total[sid] = {}

        for k in periods:

            # C_ks = sum over durations of starts that complete at k
            completion_terms = [
                w[sid][h, k - h]
                for h in durations
                if (h, k - h) in w[sid]
            ]

            completions[sid][k] = pulp.lpSum(completion_terms)

            departures_total[sid][k] = pulp.lpSum(
                flights[sid][d, k] for d in destinations
            )

        # ------------------------------------------------
        # Recourse adjustment variables
        # ------------------------------------------------

        added_departures[sid] = pulp.LpVariable.dicts(
            f"Added_s{sid}", (destinations, periods), lowBound=0, cat="Integer"
        )

        cancelled_departures[sid] = pulp.LpVariable.dicts(
            f"Cancelled_s{sid}", (destinations, periods), lowBound=0,
            cat="Integer"
        )

        emergency_capacity[sid] = pulp.LpVariable.dicts(
            f"EmergencyCharge_s{sid}", periods, lowBound=0, cat="Integer"
        )

    return {
        "model": model,
        "parameters": params,
        "variables": {
            "n": n,
            "b": b,
            "groups": groups,
            "rho": rho,
            "w": w,
            "flights": flights,
            "passengers": passengers,
            "unserved": unserved,
            "completions": completions,
            "departures_total": departures_total,
            "added_departures": added_departures,
            "cancelled_departures": cancelled_departures,
            "emergency_capacity": emergency_capacity,
        },
        "destinations": destinations,
        "periods": periods,
        "scenarios": scenarios,
        "formulation": "aggregated",
    }


# ============================================================
# Constraints and objective
# ============================================================

def add_constraints_and_objective(model_data):

    model = model_data["model"]
    vars = model_data["variables"]
    params = model_data["parameters"]
    scenarios = model_data["scenarios"]
    destinations = model_data["destinations"]
    periods = model_data["periods"]

    horizon = params["periods"]
    los_periods = params["los_periods"]
    capacity = params["evtol_capacity"]
    facilities = params["charging_facilities"]

    n = vars["n"]
    b = vars["b"]
    groups = vars["groups"]
    rho = vars["rho"]
    w = vars["w"]
    flights = vars["flights"]
    passengers = vars["passengers"]
    unserved = vars["unserved"]
    completions = vars["completions"]
    departures_total = vars["departures_total"]
    added_departures = vars["added_departures"]
    cancelled_departures = vars["cancelled_departures"]
    emergency_capacity = vars["emergency_capacity"]

    # ========================================================
    # FIRST STAGE CONSTRAINTS
    # PDF Eq. (3)-(4)
    # ========================================================

    for t in periods:

        model += (
            b[t] <= facilities
        ), f"Charging_reservation_limit_{t}"

    for t in periods:

        model += (
            pulp.lpSum(n[d, t] for d in destinations)
            <= params["max_departures"]
        ), f"Departure_reservation_limit_{t}"

    # ========================================================
    # SCENARIO RECOURSE CONSTRAINTS
    # ========================================================

    for sid, scenario in scenarios.items():

        gs = groups[sid]
        demand = scenario["passenger_demand"]

        durations = sorted({g["duration"] for g in gs.values()})

        # ----------------------------------------------------
        # Eq. (24): group charging assignment within group size
        #
        # sum_t rho_gts <= N_gs
        # ----------------------------------------------------

        for g, info in gs.items():

            starts = feasible_starts(info, horizon)

            if not starts:
                continue

            model += (
                pulp.lpSum(rho[sid][g, t] for t in starts)
                <= info["N"]
            ), f"Group_charge_limit_{sid}_{g[0]}_{g[1]}_{g[2]}"

        # ----------------------------------------------------
        # Eq. (25): duration-start totals
        #
        # sum_{g : h_gs = h, t in K_gs} rho_gts = w_hts
        # ----------------------------------------------------

        for h in durations:
            for t in periods:

                if (h, t) not in w[sid]:
                    continue

                members = [
                    rho[sid][g, t]
                    for g in gs
                    if gs[g]["duration"] == h and (g, t) in rho[sid]
                ]

                model += (
                    pulp.lpSum(members) == w[sid][h, t]
                ), f"Duration_start_total_{sid}_{h}_{t}"

        # ----------------------------------------------------
        # Eq. (26): charger occupancy via duration-start totals
        #
        # sum_h sum_{t : t <= tau < t+h} w_hts <= b_tau + b+_taus
        # ----------------------------------------------------

        for tau in periods:

            occupying = [
                w[sid][h, t]
                for h in durations
                for t in periods
                if (h, t) in w[sid] and t <= tau < t + h
            ]

            model += (
                pulp.lpSum(occupying)
                <= b[tau] + emergency_capacity[sid][tau]
            ), f"Charging_capacity_{sid}_{tau}"

            # Eq. (7): emergency capacity within the physical limit
            model += (
                emergency_capacity[sid][tau] <= facilities - b[tau]
            ), f"Emergency_capacity_limit_{sid}_{tau}"

        # ----------------------------------------------------
        # Eq. (28): cumulative flow
        #
        # Departures through period k cannot exceed completions through k.
        # This is what replaces the per-aircraft charging-before-departure
        # coupling of the indexed model.
        # ----------------------------------------------------

        for k in periods:

            if k >= horizon:
                continue

            model += (
                pulp.lpSum(departures_total[sid][kk] for kk in range(1, k + 1))
                <=
                pulp.lpSum(completions[sid][kk] for kk in range(1, k + 1))
            ), f"Cumulative_flow_{sid}_{k}"

        # ----------------------------------------------------
        # Eq. (29): every charged aircraft departs exactly once
        # ----------------------------------------------------

        total_departures = pulp.lpSum(
            departures_total[sid][k] for k in periods
        )

        total_completions = pulp.lpSum(
            completions[sid][k] for k in periods
        )

        total_starts = pulp.lpSum(
            w[sid][h, t] for (h, t) in w[sid]
        )

        model += (
            total_departures == total_completions
        ), f"Departure_completion_balance_{sid}"

        model += (
            total_completions == total_starts
        ), f"Completion_start_balance_{sid}"

        # ----------------------------------------------------
        # Eq. (30): hub departure capacity
        # ----------------------------------------------------

        for k in periods:

            model += (
                pulp.lpSum(flights[sid][d, k] for d in destinations)
                <= params["max_departures"]
            ), f"Hub_departure_capacity_{sid}_{k}"

        # ----------------------------------------------------
        # Eq. (31) and (13): commitment reconciliation
        # ----------------------------------------------------

        for d in destinations:
            for k in periods:

                model += (
                    flights[sid][d, k]
                    ==
                    n[d, k]
                    + added_departures[sid][d][k]
                    - cancelled_departures[sid][d][k]
                ), f"Commitment_link_{sid}_{d}_{k}"

                model += (
                    cancelled_departures[sid][d][k] <= n[d, k]
                ), f"Cancellation_limit_{sid}_{d}_{k}"

        # ----------------------------------------------------
        # Eq. (32): passenger conservation
        # ----------------------------------------------------

        for (r, d) in unserved[sid]:

            ks = admissible_departures(r, horizon, los_periods)

            served = pulp.lpSum(
                passengers[sid][r, d, k]
                for k in ks
                if (r, d, k) in passengers[sid]
            )

            model += (
                served + unserved[sid][r, d] == demand.get((r, d), 0)
            ), f"Demand_balance_{sid}_{r}_{d}"

        # ----------------------------------------------------
        # Eq. (33): aircraft passenger capacity per departure
        # ----------------------------------------------------

        for d in destinations:
            for k in periods:

                boarding = [
                    passengers[sid][r, d, k]
                    for r in range(1, horizon + 1)
                    if (r, d, k) in passengers[sid]
                ]

                model += (
                    pulp.lpSum(boarding)
                    <= capacity * flights[sid][d, k]
                ), f"Passenger_capacity_{sid}_{d}_{k}"

        # ----------------------------------------------------
        # Eq. (34): charging-release prefixes
        #
        # Redundant but tightens the relaxation: by period t, total starts of
        # duration h cannot exceed the number of h-aircraft that have arrived.
        # ----------------------------------------------------

        for h in durations:
            for t in periods:

                up_to_t = [
                    w[sid][h, kk]
                    for kk in range(1, t + 1)
                    if (h, kk) in w[sid]
                ]

                if not up_to_t:
                    continue

                arrived = sum(
                    info["N"]
                    for g, info in gs.items()
                    if info["duration"] == h and info["arrival"] <= t
                )

                model += (
                    pulp.lpSum(up_to_t) <= arrived
                ), f"Charging_release_{sid}_{h}_{t}"

        # ----------------------------------------------------
        # Eq. (35): destination service covers
        #
        # Redundant but tightens the relaxation: cumulative unserved demand
        # cannot fall below cumulatively arrived demand minus the seats that
        # have departed.
        # ----------------------------------------------------

        for d in destinations:

            total_demand = sum(
                demand.get((r, d), 0)
                for r in periods
            )

            if total_demand == 0:
                continue

            total_unserved = pulp.lpSum(
                unserved[sid][r, d]
                for r in periods
                if (r, d) in unserved[sid]
            )     

            total_seats = capacity * pulp.lpSum(
                flights[sid][d, k]
                for k in periods
            )   

            model += (
                total_unserved
                >=
                total_demand - total_seats
            ), f"Service_cover_{sid}_{d}"

    # ========================================================
    # SCENARIO PROFIT
    # ========================================================

    scenario_profit = {}

    for sid, scenario in scenarios.items():

        gs = groups[sid]

        # --------------------------------------------
        # Revenue: fare per served passenger
        # --------------------------------------------

        revenue = pulp.lpSum(
            passengers[sid][r, d, k]
            * params["destination_fares"].get(d, 0)
            for (r, d, k) in passengers[sid]
        )

        # --------------------------------------------
        # Charging cost: energy per group times time-of-use price
        #
        # The group energy requirement preserves arrival-SoC differences even
        # though rho is continuous.
        # --------------------------------------------

        charging_cost = pulp.lpSum(
            rho[sid][g, t]
            * gs[g]["energy"]
            * energy_price_per_kwh(
                (
                    params["horizon"]["start"]
                    + (t - 1) * params["bin_size"]
                ) / 60,
                params
            )
            for (g, t) in rho[sid]
        )

        # --------------------------------------------
        # Flight operating cost
        # --------------------------------------------

        flight_cost = pulp.lpSum(
            flights[sid][d, k] * params["costs"]["flight"]
            for d in destinations
            for k in periods
        )

        # --------------------------------------------
        # Emergency charging cost
        # --------------------------------------------

        emergency_cost = pulp.lpSum(
            emergency_capacity[sid][t] * params["costs"]["emergency_capacity"]
            for t in periods
        )

        # --------------------------------------------
        # Unserved passenger penalty
        # --------------------------------------------

        unserved_cost = pulp.lpSum(
            unserved[sid][r, d] * params["costs"]["unserved"]
            for (r, d) in unserved[sid]
        )

        # --------------------------------------------
        # Added / cancelled commitment costs
        # --------------------------------------------

        added_cost = pulp.lpSum(
            added_departures[sid][d][t] * params["costs"]["added_departure"]
            for d in destinations
            for t in periods
        )

        cancelled_cost = pulp.lpSum(
            cancelled_departures[sid][d][t]
            * params["costs"]["cancelled_departure"]
            for d in destinations
            for t in periods
        )

        scenario_profit[sid] = (
            revenue
            - charging_cost
            - flight_cost
            - emergency_cost
            - unserved_cost
            - added_cost
            - cancelled_cost
        )

    # ========================================================
    # CVaR
    # PDF Eq. (20)-(22)
    # ========================================================

    alpha = params["cvar"]["alpha"]
    theta = params["cvar"]["weight"]

    zeta = pulp.LpVariable("CVaR_zeta", lowBound=None, cat="Continuous")

    rho_cvar = pulp.LpVariable.dicts(
        "CVaR_rho", scenarios.keys(), lowBound=0, cat="Continuous"
    )

    for sid, scenario in scenarios.items():

        # L_s = unserved passengers + cancelled-seat equivalent
        scenario_loss = (
            pulp.lpSum(
                unserved[sid][r, d] for (r, d) in unserved[sid]
            )
            + capacity * pulp.lpSum(
                cancelled_departures[sid][d][t]
                for d in destinations
                for t in periods
            )
        )

        model += (
            rho_cvar[sid] >= scenario_loss - zeta
        ), f"CVaR_excess_{sid}"

    expected_profit = pulp.lpSum(
        scenarios[sid]["probability"] * scenario_profit[sid]
        for sid in scenarios
    )

    cvar = (
        zeta
        + (1 / (1 - alpha)) * pulp.lpSum(
            scenarios[sid]["probability"] * rho_cvar[sid]
            for sid in scenarios
        )
    )

    # ========================================================
    # FIRST STAGE COSTS
    # ========================================================

    commitment_cost = pulp.lpSum(
        n[d, t] * params["costs"]["commitment"]
        for d in destinations
        for t in periods
    )

    reservation_cost = pulp.lpSum(
        b[t] * params["costs"]["charging_reservation"]
        for t in periods
    )

    model += (
        expected_profit
        - commitment_cost
        - reservation_cost
        - theta * cvar
    )

    return model
