"""Projected two-stage stochastic MILP for hybrid booked + on-demand UAM service.

Stage 1
-------
* x0      : integer initial ready fleet at the hub (first-stage allocation).
* n[d,t] : advance-planned departures protecting bookings and optionally hedging on-demand demand.
* b[t]   : charging/power capacity reserved in advance.

Stage 2 (scenario dependent)
----------------------------
* a0[s]     : initial ready aircraft actually deployed in scenario s.
* chi[g,t,s] : incoming group-to-charging-start transportation flow (continuous).
* w[h,t,s]   : integer number of duration-h charging jobs started at t.
* f[d,k,s]   : integer realized departures.
* y_book / y_on : booked and on-demand passenger flows to realized departures.
* v_plus / v_minus : added/cancelled departures relative to n.
* b_plus : emergency charging capacity beyond b.

The formulation projects aircraft identities out after charge completion.  Cumulative
completion-prefix constraints are sufficient because post-charge aircraft are assumed
interchangeable in capacity, destination feasibility and flight cost.
"""

from collections import defaultdict
import math

import numpy as np
import pulp


def energy_price_per_kwh(hour, params):
    """Time-of-use electricity price used by the case study."""
    base = params["base_price"]
    if 16 <= hour < 21:
        return base * params["peak_multiplier"]
    if 0 <= hour < 6 or 21 <= hour < 24:
        return base * params["off_multiplier"]
    return base * params["shoulder_multiplier"]


def charging_periods_required(soc, params):
    """15-min periods needed to reach the required departure SoC."""
    deficit = max(0.0, params["soc_min"] - soc)
    if deficit <= 0:
        return 0
    energy = params["battery_capacity"] * deficit / 100.0
    hours = energy / params["charging_rate"]
    return int(math.ceil(hours * 60.0 / params["bin_size"]))


def _energy_needed_kwh(soc, params):
    return params["battery_capacity"] * max(0.0, params["soc_min"] - soc) / 100.0


def _build_aircraft_groups(scenario, params):
    """Aggregate aircraft by arrival period, charge duration and charge energy."""
    counts = defaultdict(int)
    for info in scenario["aircraft"].values():
        arrival = int(info["arrival_period"])
        duration = charging_periods_required(info["initial_soc"], params)
        energy = round(_energy_needed_kwh(info["initial_soc"], params), 6)
        counts[(arrival, duration, energy)] += 1

    groups = {}
    for gid, ((arrival, duration, energy), number) in enumerate(sorted(counts.items())):
        groups[gid] = {
            "arrival_period": arrival,
            "duration": duration,
            "energy_kwh": energy,
            "count": int(number),
        }
    return groups


def _window(r, n_periods, los_periods):
    return range(r, min(n_periods, r + los_periods) + 1)


def build_stochastic_model(model_input):
    """Create variables for the projected extensive-form model."""
    params = model_input["parameters"]
    scenarios = model_input["scenarios"]
    periods = tuple(range(1, params["periods"] + 1))

    advance_bookings = model_input.get("advance_bookings", {})
    destinations = sorted(
        {d for (_, d) in advance_bookings}
        | {
            d
            for scenario in scenarios.values()
            for (_, d) in scenario["on_demand_demand"]
        }
    )

    model = pulp.LpProblem("Hybrid_Booked_OnDemand_UAM", pulp.LpMaximize)

    # ------------------------- first stage -------------------------
    x0 = pulp.LpVariable(
        "InitialReadyFleet", lowBound=0,
        upBound=params.get("initial_fleet_max", 0), cat="Integer"
    )
    n = {
        (d, t): pulp.LpVariable(f"PlannedDeparture_{d}_{t}", lowBound=0, cat="Integer")
        for d in destinations for t in periods
    }
    b = {
        t: pulp.LpVariable(f"ReservedCharging_{t}", lowBound=0, cat="Integer")
        for t in periods
    }

    # Integer first-stage transportation flow used only to certify that
    # planned flights contain enough seats for every accepted booking group.
    protected = {}
    for r in periods:
        for d in destinations:
            for k in _window(r, params["periods"], params["los_periods"]):
                protected[r, d, k] = pulp.LpVariable(
                    f"ProtectedBooking_{r}_{d}_{k}", lowBound=0, cat="Integer"
                )

    # ------------------------- scenario data -----------------------
    groups = {}
    feasible_starts = {}
    durations = {}
    for sid, scenario in scenarios.items():
        groups[sid] = _build_aircraft_groups(scenario, params)
        durations[sid] = sorted({g["duration"] for g in groups[sid].values()})
        feasible_starts[sid] = {}
        for gid, g in groups[sid].items():
            last = params["periods"] - g["duration"]
            feasible_starts[sid][gid] = tuple(
                t for t in periods if g["arrival_period"] <= t <= last
            )

    # ------------------------- recourse variables ------------------
    a0 = {sid: pulp.LpVariable(
        f"InitialFleetDeployed_s{sid}", lowBound=0, cat="Integer"
    ) for sid in scenarios}
    chi = {}
    w = {}
    flights = {}
    y_book = {}
    y_on = {}
    eta_book = {}
    eta_on = {}
    added = {}
    cancelled = {}
    emergency = {}

    for sid, scenario in scenarios.items():
        for gid in groups[sid]:
            for t in feasible_starts[sid][gid]:
                chi[sid, gid, t] = pulp.LpVariable(
                    f"ChargeFlow_s{sid}_g{gid}_t{t}", lowBound=0, cat="Continuous"
                )

        for h in durations[sid]:
            for t in periods:
                w[sid, h, t] = pulp.LpVariable(
                    f"DurationStart_s{sid}_h{h}_t{t}", lowBound=0, cat="Integer"
                )

        for d in destinations:
            for k in periods:
                flights[sid, d, k] = pulp.LpVariable(
                    f"Flight_s{sid}_{d}_{k}", lowBound=0, cat="Integer"
                )
                added[sid, d, k] = pulp.LpVariable(
                    f"Added_s{sid}_{d}_{k}", lowBound=0, cat="Integer"
                )
                cancelled[sid, d, k] = pulp.LpVariable(
                    f"Cancelled_s{sid}_{d}_{k}", lowBound=0, cat="Integer"
                )

        for t in periods:
            emergency[sid, t] = pulp.LpVariable(
                f"EmergencyCharge_s{sid}_{t}", lowBound=0, cat="Integer"
            )

        for r in periods:
            for d in destinations:
                eta_book[sid, r, d] = pulp.LpVariable(
                    f"UnservedBooked_s{sid}_{r}_{d}", lowBound=0, cat="Integer"
                )
                eta_on[sid, r, d] = pulp.LpVariable(
                    f"UnservedOnDemand_s{sid}_{r}_{d}", lowBound=0, cat="Integer"
                )
                for k in _window(r, params["periods"], params["los_periods"]):
                    y_book[sid, r, d, k] = pulp.LpVariable(
                        f"BookedFlow_s{sid}_{r}_{d}_{k}", lowBound=0, cat="Integer"
                    )
                    y_on[sid, r, d, k] = pulp.LpVariable(
                        f"OnDemandFlow_s{sid}_{r}_{d}_{k}", lowBound=0, cat="Integer"
                    )

    zeta = pulp.LpVariable("CVaR_zeta", lowBound=0, cat="Continuous")
    rho = {
        sid: pulp.LpVariable(f"CVaR_rho_{sid}", lowBound=0, cat="Continuous")
        for sid in scenarios
    }

    return {
        "model": model,
        "parameters": params,
        "advance_bookings": advance_bookings,
        "scenarios": scenarios,
        "periods": periods,
        "destinations": destinations,
        "groups": groups,
        "feasible_starts": feasible_starts,
        "durations": durations,
        "variables": {
            "initial_fleet": x0,
            "deployed_initial": a0,
            "n": n,
            "b": b,
            "protected_bookings": protected,
            "chi": chi,
            "w": w,
            "flights": flights,
            "booked_passengers": y_book,
            "on_demand_passengers": y_on,
            "unserved_booked": eta_book,
            "unserved_on_demand": eta_on,
            "added_departures": added,
            "cancelled_departures": cancelled,
            "emergency_capacity": emergency,
            "zeta": zeta,
            "rho": rho,
        },
    }


def add_constraints_and_objective(model_data):
    """Add the hybrid booking/on-demand constraints and risk-averse objective."""
    model = model_data["model"]
    params = model_data["parameters"]
    scenarios = model_data["scenarios"]
    bookings = model_data["advance_bookings"]
    periods = model_data["periods"]
    destinations = model_data["destinations"]
    groups = model_data["groups"]
    feasible = model_data["feasible_starts"]
    durations = model_data["durations"]
    v = model_data["variables"]

    x0 = v["initial_fleet"]
    a0 = v["deployed_initial"]
    n = v["n"]
    b = v["b"]
    protected = v["protected_bookings"]
    chi = v["chi"]
    w = v["w"]
    flights = v["flights"]
    y_book = v["booked_passengers"]
    y_on = v["on_demand_passengers"]
    eta_book = v["unserved_booked"]
    eta_on = v["unserved_on_demand"]
    added = v["added_departures"]
    cancelled = v["cancelled_departures"]
    emergency = v["emergency_capacity"]
    zeta = v["zeta"]
    rho = v["rho"]

    # ============================================================
    # Stage 1: charging readiness and service prepared for bookings
    # ============================================================
    for t in periods:
        model += b[t] <= params["charging_facilities"], f"ReserveLimit_{t}"
        model += (
            pulp.lpSum(n[d, t] for d in destinations) <= params["max_departures"]
        ), f"PlannedDepartureLimit_{t}"

    for r in periods:
        for d in destinations:
            window = list(_window(r, params["periods"], params["los_periods"]))
            model += (
                pulp.lpSum(protected[r, d, k] for k in window)
                == bookings.get((r, d), 0)
            ), f"BookingProtection_{r}_{d}"

    for d in destinations:
        for k in periods:
            eligible_r = [
                r for r in periods
                if k in _window(r, params["periods"], params["los_periods"])
            ]
            model += (
                pulp.lpSum(protected[r, d, k] for r in eligible_r)
                <= params["evtol_capacity"] * n[d, k]
            ), f"ProtectedSeatCapacity_{d}_{k}"

    # ============================================================
    # Stage 2: projected charging, cumulative ready-aircraft flow
    # ============================================================
    scenario_profit = {}
    scenario_loss = {}

    for sid, scenario in scenarios.items():
        model += a0[sid] <= x0, f"InitialFleetDeployment_s{sid}"
        # group supply and duration-start margins
        for gid, g in groups[sid].items():
            model += (
                pulp.lpSum(chi[sid, gid, t] for t in feasible[sid][gid])
                <= g["count"]
            ), f"GroupCapacity_s{sid}_g{gid}"

        for h in durations[sid]:
            for t in periods:
                eligible_g = [
                    gid for gid, g in groups[sid].items()
                    if g["duration"] == h and t in feasible[sid][gid]
                ]
                model += (
                    pulp.lpSum(chi[sid, gid, t] for gid in eligible_g)
                    == w[sid, h, t]
                ), f"DurationStartMargin_s{sid}_h{h}_t{t}"

        # charging occupancy and emergency activation
        for ell in periods:
            occupying = []
            for h in durations[sid]:
                if h == 0:
                    continue
                for t in periods:
                    if t <= ell < t + h:
                        occupying.append(w[sid, h, t])
            model += (
                pulp.lpSum(occupying) <= b[ell] + emergency[sid, ell]
            ), f"ChargingOccupancy_s{sid}_t{ell}"
            model += (
                emergency[sid, ell] <= params["charging_facilities"] - b[ell]
            ), f"EmergencyLimit_s{sid}_t{ell}"

        # ready-aircraft cumulative precedence
        completion = {}
        departure_total = {}
        for k in periods:
            terms = []
            for h in durations[sid]:
                start = k - h
                if start in periods:
                    terms.append(w[sid, h, start])
            completion[k] = pulp.lpSum(terms)
            departure_total[k] = pulp.lpSum(flights[sid, d, k] for d in destinations)
            model += departure_total[k] <= params["max_departures"], f"TakeoffCap_s{sid}_t{k}"

        for k in periods:
            model += (
                pulp.lpSum(departure_total[ell] for ell in periods if ell <= k)
                <= a0[sid] + pulp.lpSum(completion[ell] for ell in periods if ell <= k)
            ), f"CompletionPrefix_s{sid}_t{k}"

        model += (
            pulp.lpSum(departure_total[k] for k in periods)
            == a0[sid] + pulp.lpSum(completion[k] for k in periods)
        ), f"CompletionTotalBalance_s{sid}"

        # planned versus realized service
        for d in destinations:
            for k in periods:
                model += (
                    flights[sid, d, k]
                    == n[d, k] + added[sid, d, k] - cancelled[sid, d, k]
                ), f"ServiceReconciliation_s{sid}_{d}_{k}"
                model += (
                    cancelled[sid, d, k] <= n[d, k]
                ), f"CancellationBound_s{sid}_{d}_{k}"

        # passenger balances: advance bookings are shared across scenarios;
        # on-demand requests are scenario dependent.
        on_demand = scenario["on_demand_demand"]
        for r in periods:
            for d in destinations:
                window = list(_window(r, params["periods"], params["los_periods"]))
                model += (
                    pulp.lpSum(y_book[sid, r, d, k] for k in window)
                    + eta_book[sid, r, d]
                    == bookings.get((r, d), 0)
                ), f"BookedDemandBalance_s{sid}_{r}_{d}"
                model += (
                    pulp.lpSum(y_on[sid, r, d, k] for k in window)
                    + eta_on[sid, r, d]
                    == on_demand.get((r, d), 0)
                ), f"OnDemandBalance_s{sid}_{r}_{d}"

        for d in destinations:
            for k in periods:
                eligible_r = [
                    r for r in periods
                    if k in _window(r, params["periods"], params["los_periods"])
                ]
                model += (
                    pulp.lpSum(
                        y_book[sid, r, d, k] + y_on[sid, r, d, k]
                        for r in eligible_r
                    )
                    <= params["evtol_capacity"] * flights[sid, d, k]
                ), f"RealizedSeatCapacity_s{sid}_{d}_{k}"

        # ---------------- objective components ----------------
        revenue = pulp.lpSum(
            params["destination_fares"].get(d, 0.0)
            * (y_book[sid, r, d, k] + y_on[sid, r, d, k])
            for r in periods for d in destinations
            for k in _window(r, params["periods"], params["los_periods"])
        )

        charging_cost = pulp.lpSum(
            chi[sid, gid, t]
            * groups[sid][gid]["energy_kwh"]
            * energy_price_per_kwh(
                (params["horizon"]["start"] + (t - 1) * params["bin_size"]) / 60.0,
                params,
            )
            for gid in groups[sid] for t in feasible[sid][gid]
        )

        flight_cost = pulp.lpSum(
            params["costs"]["flight"] * flights[sid, d, k]
            for d in destinations for k in periods
        )
        emergency_cost = pulp.lpSum(
            params["costs"]["emergency_capacity"] * emergency[sid, t]
            for t in periods
        )
        added_cost = pulp.lpSum(
            params["costs"]["added_departure"] * added[sid, d, k]
            for d in destinations for k in periods
        )
        cancelled_cost = pulp.lpSum(
            params["costs"]["cancelled_departure"] * cancelled[sid, d, k]
            for d in destinations for k in periods
        )
        booked_unserved_cost = pulp.lpSum(
            params["costs"]["unserved_booked"] * eta_book[sid, r, d]
            for r in periods for d in destinations
        )
        on_unserved_cost = pulp.lpSum(
            params["costs"]["unserved_on_demand"] * eta_on[sid, r, d]
            for r in periods for d in destinations
        )

        scenario_profit[sid] = (
            revenue - charging_cost - flight_cost - emergency_cost
            - added_cost - cancelled_cost - booked_unserved_cost - on_unserved_cost
        )

        scenario_loss[sid] = (
            params["cvar"]["booked_weight"]
            * pulp.lpSum(eta_book[sid, r, d] for r in periods for d in destinations)
            + params["cvar"]["on_demand_weight"]
            * pulp.lpSum(eta_on[sid, r, d] for r in periods for d in destinations)
        )
        model += rho[sid] >= scenario_loss[sid] - zeta, f"CVaRExcess_s{sid}"

    expected_profit = pulp.lpSum(
        scenarios[sid]["probability"] * scenario_profit[sid] for sid in scenarios
    )
    alpha = params["cvar"]["alpha"]
    cvar = zeta + (1.0 / (1.0 - alpha)) * pulp.lpSum(
        scenarios[sid]["probability"] * rho[sid] for sid in scenarios
    )
    commitment_cost = pulp.lpSum(
        params["costs"]["commitment"] * n[d, t]
        for d in destinations for t in periods
    )
    reservation_cost = pulp.lpSum(
        params["costs"]["charging_reservation"] * b[t] for t in periods
    )

    fleet_cost = params["costs"].get("initial_fleet", 0.0) * x0
    model += (
        expected_profit - fleet_cost - commitment_cost - reservation_cost
        - params["cvar"]["weight"] * cvar
    )

    model_data["expressions"] = {
        "scenario_profit": scenario_profit,
        "scenario_loss": scenario_loss,
        "expected_profit": expected_profit,
        "cvar": cvar,
        "commitment_cost": commitment_cost,
        "reservation_cost": reservation_cost,
        "initial_fleet_cost": fleet_cost,
    }
    return model


SOLVER_PREFERENCE = ("GUROBI", "HIGHS", "CBC")


def _make_solver(name, time_limit, gap_rel, threads, verbose, log_path=None):
    name = name.upper()
    if name == "GUROBI":
        solver = pulp.GUROBI(msg=verbose, timeLimit=time_limit, gapRel=gap_rel)
        if not solver.available():
            return None, "gurobipy unavailable or not licensed"
        if threads:
            solver.solver_params["Threads"] = threads
        if log_path:
            solver.solver_params["LogFile"] = log_path
        return solver, None
    if name == "HIGHS":
        solver = pulp.HiGHS(
            msg=verbose, timeLimit=time_limit, gapRel=gap_rel,
            threads=threads or None,
        )
        return (solver, None) if solver.available() else (None, "highspy unavailable")
    if name == "CBC":
        solver = pulp.PULP_CBC_CMD(
            msg=verbose, timeLimit=time_limit, gapRel=gap_rel,
            threads=threads or None,
        )
        return (solver, None) if solver.available() else (None, "cbc unavailable")
    return None, f"unknown solver {name!r}"


def solve_model(
    model,
    time_limit=3600,
    gap_rel=0.05,
    threads=0,
    log_path=None,
    verbose=True,
    solver="auto",
    **_ignored,
):
    """Solve with Gurobi/HiGHS/CBC. Auto mode falls back if a solver fails."""
    candidates = SOLVER_PREFERENCE if solver == "auto" else (solver.upper(),)
    reasons = []

    for name in candidates:
        inst, reason = _make_solver(name, time_limit, gap_rel, threads, verbose, log_path)
        if inst is None:
            reasons.append(f"{name}: {reason}")
            continue
        try:
            print(f"[info] solving with {name}")
            model.solve(inst)
            status = pulp.LpStatus.get(model.status, str(model.status))
            objective = pulp.value(model.objective) if model.objective is not None else None
            raw_gap = None
            backend = getattr(model, "solverModel", None)
            if backend is not None and hasattr(backend, "getInfo"):
                try:
                    candidate = float(backend.getInfo().mip_gap)
                    if math.isfinite(candidate):
                        raw_gap = candidate
                except (AttributeError, RuntimeError, ValueError):
                    pass
            return {
                "status": status,
                "objective": objective,
                "solver": name,
                "proven_optimal": status == "Optimal" and raw_gap is not None and raw_gap <= 1e-8,
                "solver_reported_gap": raw_gap,
                "n_variables": len(model.variables()),
                "n_constraints": len(model.constraints),
            }
        except Exception as exc:  # license-size failures should fall through in auto mode
            reasons.append(f"{name}: solve failed ({exc})")
            if solver != "auto":
                raise
            print(f"[info] {name} failed; trying next solver: {exc}")

    raise RuntimeError("No usable MILP solver found. " + "; ".join(reasons))
