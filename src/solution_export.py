"""
Reporting tables for a solved two-stage stochastic model.

Reports the objective decomposition (expected profit, first-stage costs and
the CVaR penalty) reconciled against the solver objective, the booked and
on-demand service rates, the per-scenario served and unserved passengers,
cancellations and charger utilization, and the scenario-level CVaR (``L_s``,
``VaR``, ``CVaR`` and its penalty).

The two formulations store the same economic quantities under different
variable families, so every accessor below is formulation aware.

Usage::

    python src/solution_export.py

Outputs land in ``output/results/solution_dump/`` by default.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pulp

import config
from economics import charging_cost_per_kwh


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _val(variable):
    """Solved value of a PuLP variable; 0.0 when unset or None."""
    if variable is None:
        return 0.0
    value = variable.value()
    return 0.0 if value is None else float(value)


def _write_csv(path, header, rows):
    """
    Write one CSV.

    A locked file (typically the CSV being held open in Excel) must not abort
    the run -- the console report already carries these numbers, so warn and
    carry on.
    """
    path = Path(path)

    try:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    except OSError as error:
        print(
            "[warning] could not write %s (%s); the console report already "
            "carries these numbers, skipping the file." % (path, error)
        )

    return path


def _formulation(model_data):
    return model_data.get("formulation", "indexed")


def _periods(model_data):
    return [int(t) for t in model_data["periods"]]


# ------------------------------------------------------------
# service outcomes (formulation agnostic)
# ------------------------------------------------------------

def served_by_destination(sid, model_data):
    """
    ``{destination: {"booked": x, "ondemand": y}}`` served passengers.

    The two classes are tracked separately so their service rates stay
    distinguishable.
    """
    variables = model_data["variables"]
    destinations = model_data["destinations"]

    out = {d: {"booked": 0.0, "ondemand": 0.0} for d in destinations}

    if _formulation(model_data) == "aggregated":

        for (r, d, k), variable in variables["booked_passengers"][sid].items():
            out[d]["booked"] += _val(variable)

        for (r, d, k), variable in variables["ondemand_passengers"][sid].items():
            out[d]["ondemand"] += _val(variable)

        return out

    for (a, r, d), variable in variables["booked_passengers"][sid].items():
        out[d]["booked"] += _val(variable)

    for (a, r, d), variable in variables["ondemand_passengers"][sid].items():
        out[d]["ondemand"] += _val(variable)

    return out


def unserved_by_destination(sid, model_data):
    """``{destination: {"booked": x, "ondemand": y}}`` unserved passengers."""
    variables = model_data["variables"]
    destinations = model_data["destinations"]

    out = {d: {"booked": 0.0, "ondemand": 0.0} for d in destinations}

    for (r, d), variable in variables["unserved_booked"][sid].items():
        out[d]["booked"] += _val(variable)

    for (r, d), variable in variables["unserved_ondemand"][sid].items():
        out[d]["ondemand"] += _val(variable)

    return out


def service_rates(model_data):
    """
    Booked and on-demand service rates over the whole ensemble.

    Both numerator and denominator are probability weighted, so the on-demand
    rate is ``E[served] / E[requests]`` under the reduced scenario distribution
    rather than a weighted numerator over an unweighted total. Booked demand is
    scenario independent, so its denominator is simply the booking total.

    The classes are reported separately because a single pooled rate can hide
    bookings being protected at the expense of on-demand requests.
    """
    destinations = model_data["destinations"]
    scenarios = model_data["scenarios"]

    booked_demand = {d: 0.0 for d in destinations}

    for (t, d), value in model_data["booked_demand"].items():
        booked_demand[d] += float(value)

    booked_total = sum(booked_demand.values())

    ondemand_total = 0.0

    booked_served = 0.0
    ondemand_served = 0.0

    for sid, scenario in scenarios.items():

        probability = float(scenario["probability"])

        ondemand_total += probability * sum(
            float(v) for v in scenario["ondemand_demand"].values()
        )

        served = served_by_destination(sid, model_data)

        booked_served += probability * sum(s["booked"] for s in served.values())

        ondemand_served += probability * sum(s["ondemand"] for s in served.values())

    return {
        "booked_demand": booked_total,
        "ondemand_demand": ondemand_total,
        "booked_served": booked_served,
        "ondemand_served": ondemand_served,
        "booked_service_rate": (
            booked_served / booked_total if booked_total > 0 else float("nan")
        ),
        "ondemand_service_rate": (
            ondemand_served / ondemand_total if ondemand_total > 0 else float("nan")
        ),
    }


# ------------------------------------------------------------
# per-scenario service outcomes
# ------------------------------------------------------------

def scenario_service(model_data):
    """
    Served and unserved passenger counts for every retained scenario.

    :func:`service_rates` gives the ensemble rates; this table gives the batch
    behind each scenario, which is what a per-scenario service report needs
    (:func:`print_scenario_service`). Both classes stay separate for the same
    reason as above: booked protection and on-demand acceptance are different
    decisions.

    Returns one dict per retained scenario, in model order::

        {"scenario", "probability",
         "booked_demand", "booked_served", "booked_unserved",
         "ondemand_demand", "ondemand_served", "ondemand_unserved",
         "demand_total", "served_total", "unserved_total", "served_share"}

    Booked demand is known before the first stage, so the same
    ``booked_demand`` appears in every row; a row where
    ``booked_served + booked_unserved`` misses it means the served/unserved
    variables were not read correctly. ``served_share`` is the pooled service
    rate of the scenario, ``served_total / demand_total``.

    Read the values from a solved model only -- on an unsolved model every
    variable is ``None`` and the table silently reads as zeros.
    """
    scenarios = model_data["scenarios"]

    booked_demand = float(
        sum(float(value) for value in model_data["booked_demand"].values())
    )

    rows = []

    for sid, scenario in scenarios.items():

        served = served_by_destination(sid, model_data)

        unserved = unserved_by_destination(sid, model_data)

        booked_served = float(sum(entry["booked"] for entry in served.values()))

        ondemand_served = float(sum(entry["ondemand"] for entry in served.values()))

        booked_unserved = float(sum(entry["booked"] for entry in unserved.values()))

        ondemand_unserved = float(
            sum(entry["ondemand"] for entry in unserved.values())
        )

        ondemand_demand = float(
            sum(float(value) for value in scenario["ondemand_demand"].values())
        )

        served_total = booked_served + ondemand_served

        unserved_total = booked_unserved + ondemand_unserved

        demand_total = booked_demand + ondemand_demand

        rows.append(
            {
                "scenario": sid,
                "probability": float(scenario["probability"]),
                "booked_demand": booked_demand,
                "booked_served": booked_served,
                "booked_unserved": booked_unserved,
                "ondemand_demand": ondemand_demand,
                "ondemand_served": ondemand_served,
                "ondemand_unserved": ondemand_unserved,
                "demand_total": demand_total,
                "served_total": served_total,
                "unserved_total": unserved_total,
                "served_share": (
                    served_total / demand_total if demand_total > 0 else float("nan")
                ),
            }
        )

    return rows


# ------------------------------------------------------------
# cost components (formulation agnostic)
# ------------------------------------------------------------

def charging_cost(sid, model_data):
    """Energy cost of charging, priced by the time-of-use tariff."""
    variables = model_data["variables"]
    parameters = model_data["parameters"]

    total = 0.0

    if _formulation(model_data) == "aggregated":

        groups = variables["groups"][sid]

        for (key, t), variable in variables["group_charge"][sid].items():
            total += (
                _val(variable)
                * groups[key]["energy"]
                * charging_cost_per_kwh(t, parameters)
            )

        return total

    info = model_data["structure"]["scenarios"][sid]

    for a in info["incoming"]:
        for t in info["starts"][a]:
            total += (
                _val(variables["charging"][sid][a, t])
                * info["energy"][a]
                * charging_cost_per_kwh(t, parameters)
            )

    return total


def flight_cost(sid, model_data):
    """Operating cost of the realized flights."""
    variables = model_data["variables"]
    parameters = model_data["parameters"]

    total = 0.0

    if _formulation(model_data) == "aggregated":

        for (d, k), variable in variables["flights"][sid].items():
            total += _val(variable) * parameters["destination_flight_cost"][d]

        return total

    for (a, d, k), variable in variables["flights"][sid].items():
        total += _val(variable) * parameters["destination_flight_cost"][d]

    return total


def deployed_aircraft(sid, model_data):
    """Number of aircraft actually dispatched in the scenario."""
    variables = model_data["variables"]

    if _formulation(model_data) == "aggregated":
        return _val(variables["deployed_initial"][sid]) + _charged_aircraft(
            sid, model_data
        )

    return sum(_val(variable) for variable in variables["u"][sid].values())


def _charged_aircraft(sid, model_data):
    """Number of incoming aircraft that completed a charging operation."""
    variables = model_data["variables"]

    if _formulation(model_data) == "aggregated":
        return sum(_val(v) for v in variables["duration_start"][sid].values())

    info = model_data["structure"]["scenarios"][sid]

    return sum(
        _val(variables["charging"][sid][a, t])
        for a in info["incoming"]
        for t in info["starts"][a]
    )


def departures(sid, model_data):
    """Total realized departures in the scenario."""
    variables = model_data["variables"]

    if _formulation(model_data) == "aggregated":
        return sum(_val(v) for v in variables["departure_totals"][sid].values())

    return sum(_val(v) for v in variables["flights"][sid].values())


def charger_occupancy(sid, model_data):
    """
    ``{period: occupied charger units}``.

    Both formulations read the same physical quantity from their own charger
    variables, so they report identically.
    """
    variables = model_data["variables"]
    periods = _periods(model_data)

    occupancy = {t: 0.0 for t in periods}

    if _formulation(model_data) == "aggregated":

        for (duration, start), variable in variables["duration_start"][sid].items():
            value = _val(variable)
            for t in periods:
                if start <= t < start + duration:
                    occupancy[t] += value

        return occupancy

    info = model_data["structure"]["scenarios"][sid]

    for a in info["incoming"]:
        for start in info["starts"][a]:
            value = _val(variables["charging"][sid][a, start])
            for t in periods:
                if start <= t < start + info["duration"][a]:
                    occupancy[t] += value

    return occupancy


# ------------------------------------------------------------
# objective decomposition
# ------------------------------------------------------------

def objective_breakdown(model_data):
    """
    Recompute every objective term from the solved variables.

    Returns ``(rows, totals, expected_profit)`` where ``totals`` holds
    probability-weighted components and ``expected_profit`` is the
    probability-weighted expected profit before first-stage and CVaR costs.
    """
    parameters = model_data["parameters"]
    variables = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = _periods(model_data)
    destinations = model_data["destinations"]

    costs = parameters["costs"]
    fares = parameters["destination_fares"]

    penalty_booked = float(parameters["unserved_booked"])
    penalty_ondemand = float(parameters["unserved_ondemand"])

    totals = {
        "revenue": 0.0,
        "charging": 0.0,
        "flight": 0.0,
        "emergency": 0.0,
        "added": 0.0,
        "cancelled": 0.0,
        "unserved_booked": 0.0,
        "unserved_ondemand": 0.0,
    }

    expected_profit = 0.0

    rows = []

    for sid, scenario in scenarios.items():

        probability = float(scenario["probability"])

        served = served_by_destination(sid, model_data)

        unserved = unserved_by_destination(sid, model_data)

        revenue = sum(
            fares.get(d, 0.0) * (served[d]["booked"] + served[d]["ondemand"])
            for d in destinations
        )

        charging = charging_cost(sid, model_data)

        operating = flight_cost(sid, model_data)

        emergency_units = sum(
            _val(variables["emergency_capacity"][sid][t]) for t in periods
        )

        added_units = sum(
            _val(variables["added_departures"][sid][d, t])
            for d in destinations
            for t in periods
        )

        cancelled_units = sum(
            _val(variables["cancelled_departures"][sid][d, t])
            for d in destinations
            for t in periods
        )

        unserved_booked_units = sum(u["booked"] for u in unserved.values())

        unserved_ondemand_units = sum(u["ondemand"] for u in unserved.values())

        emergency_cost = emergency_units * costs["emergency_capacity"]

        added_cost = added_units * costs["added_departure"]

        cancelled_cost = cancelled_units * costs["cancelled_departure"]

        unserved_booked_cost = unserved_booked_units * penalty_booked

        unserved_ondemand_cost = unserved_ondemand_units * penalty_ondemand

        profit = (
            revenue
            - charging
            - operating
            - emergency_cost
            - added_cost
            - cancelled_cost
            - unserved_booked_cost
            - unserved_ondemand_cost
        )

        expected_profit += probability * profit

        totals["revenue"] += probability * revenue
        totals["charging"] += probability * charging
        totals["flight"] += probability * operating
        totals["emergency"] += probability * emergency_cost
        totals["added"] += probability * added_cost
        totals["cancelled"] += probability * cancelled_cost
        totals["unserved_booked"] += probability * unserved_booked_cost
        totals["unserved_ondemand"] += probability * unserved_ondemand_cost

        rows.append(
            {
                "scenario": sid,
                "probability": round(probability, 6),
                "departures": round(departures(sid, model_data), 4),
                "aircraft_dispatched": round(deployed_aircraft(sid, model_data), 4),
                "served_booked": round(sum(s["booked"] for s in served.values()), 4),
                "served_ondemand": round(sum(s["ondemand"] for s in served.values()), 4),
                "unserved_booked": round(unserved_booked_units, 4),
                "unserved_ondemand": round(unserved_ondemand_units, 4),
                "revenue": round(revenue, 4),
                "charging_cost": round(charging, 4),
                "flight_cost": round(operating, 4),
                "emergency_cost": round(emergency_cost, 4),
                "added_cost": round(added_cost, 4),
                "cancelled_cost": round(cancelled_cost, 4),
                "unserved_booked_cost": round(unserved_booked_cost, 4),
                "unserved_ondemand_cost": round(unserved_ondemand_cost, 4),
                "scenario_profit": round(profit, 4),
            }
        )

    return rows, totals, expected_profit


# ------------------------------------------------------------
# CVaR
# ------------------------------------------------------------

def cvar_summary(model_data):
    """
    Recompute ``L_s``, ``VaR``, ``CVaR`` and the penalized CVaR term.

    ``L_s`` is read from the model's own loss variable, so it is the
    class-weighted loss rather than a passenger-equivalent proxy.
    """
    parameters = model_data["parameters"]
    variables = model_data["variables"]
    scenarios = model_data["scenarios"]

    alpha = float(parameters["cvar"]["alpha"])
    theta = float(parameters["cvar"]["weight"])

    losses = {sid: _val(variables["loss"][sid]) for sid in scenarios}

    probabilities = np.array(
        [float(scenarios[sid]["probability"]) for sid in scenarios], dtype=float
    )

    loss_vector = np.array([losses[sid] for sid in scenarios], dtype=float)

    if loss_vector.size == 0:
        return {
            "losses": {},
            "var": 0.0,
            "cvar": 0.0,
            "alpha": alpha,
            "theta": theta,
            "multiplier": 0.0,
            "penalty": 0.0,
        }

    order = np.argsort(-loss_vector)

    cumulative = 0.0

    var = float(loss_vector[order[0]])

    for index in order:
        cumulative += probabilities[index]
        if cumulative >= (1.0 - alpha):
            var = float(loss_vector[index])
            break

    excess = float(
        sum(p * max(0.0, loss - var) for p, loss in zip(probabilities, loss_vector))
    )

    cvar = var + (1.0 / (1.0 - alpha)) * excess

    # Effective marginal penalty of one more unserved passenger of each class,
    # conditional on being strictly inside the tail.
    marginal_booked = float(parameters["unserved_booked"]) + (
        theta * float(parameters["cvar"]["weight_booked"]) / (1.0 - alpha)
    )

    marginal_ondemand = float(parameters["unserved_ondemand"]) + (
        theta * float(parameters["cvar"]["weight_ondemand"]) / (1.0 - alpha)
    )

    return {
        "losses": losses,
        "var": var,
        "cvar": cvar,
        "alpha": alpha,
        "theta": theta,
        "multiplier": theta / (1.0 - alpha),
        "penalty": theta * cvar,
        "marginal_penalty_booked": marginal_booked,
        "marginal_penalty_ondemand": marginal_ondemand,
    }


# ------------------------------------------------------------
# First-stage cost accounting
# ------------------------------------------------------------

def first_stage_costs(model_data):
    """
    First-stage cost read from the solved variables and the parameter
    dictionary the model actually received.

    The model prices the first stage as

        c_fleet * x0 + c_res * sum_t b_t + c_com * sum_{d,t} n_dt

    so the report must read the same ``parameters["costs"]`` and the same
    variables. Pricing from a module-level constant is what makes a report
    disagree with the objective after a cost-profile change or a parameter
    override.

    Returns
    -------
    dict
        ``x0``, ``reserved_units``, ``planned_units``, ``fleet_cost``,
        ``reservation_cost``, ``commitment_cost`` and their ``total``.
    """
    parameters = model_data["parameters"]

    variables = model_data["variables"]

    costs = parameters["costs"]

    # Read the solved decision variables at the exact parameters passed to the
    # model.
    x0_value = pulp.value(variables["x0"]) or 0.0

    reserved_total = sum(
        pulp.value(variable) or 0.0 for variable in variables["b"].values()
    )

    planned_total = sum(
        pulp.value(variable) or 0.0 for variable in variables["n"].values()
    )

    fleet_cost = float(parameters["initial_aircraft_cost"]) * x0_value

    reservation_cost = float(costs["charging_reservation"]) * reserved_total

    commitment_cost = float(costs["commitment"]) * planned_total

    return {
        "x0": float(x0_value),
        "reserved_units": float(reserved_total),
        "planned_units": float(planned_total),
        "fleet_cost": float(fleet_cost),
        "reservation_cost": float(reservation_cost),
        "commitment_cost": float(commitment_cost),
        "total": float(fleet_cost + reservation_cost + commitment_cost),
    }


# ------------------------------------------------------------
# summary
# ------------------------------------------------------------

def build_summary(model_data):
    """
    One dictionary with every reported quantity.

    ``objective`` is recomputed from the components, so comparing it with the
    solver objective is a genuine reconciliation check rather than an echo.
    """
    _, totals, expected_profit = objective_breakdown(model_data)

    risk = cvar_summary(model_data)

    first = first_stage_costs(model_data)

    fleet_units = first["x0"]

    reservation_units = first["reserved_units"]

    preparation_units = first["planned_units"]

    fleet_cost = first["fleet_cost"]

    reservation_cost = first["reservation_cost"]

    preparation_cost = first["commitment_cost"]

    objective = (
        expected_profit
        - fleet_cost
        - reservation_cost
        - preparation_cost
        - risk["penalty"]
    )

    rates = service_rates(model_data)

    return {
        "formulation": _formulation(model_data),
        "scenarios": len(model_data["scenarios"]),
        "x0": fleet_units,
        "fleet_cost": fleet_cost,
        "charging_reservation_units": reservation_units,
        "charging_reservation_cost": reservation_cost,
        "planned_departures": preparation_units,
        "preparation_cost": preparation_cost,
        "expected_profit": expected_profit,
        "cvar": risk["cvar"],
        "var": risk["var"],
        "cvar_penalty": risk["penalty"],
        "objective": objective,
        "components": totals,
        "service": rates,
    }


# ------------------------------------------------------------
# CSV export
# ------------------------------------------------------------

def export_solution(model_data, out_dir=None):
    """
    Write every decision-variable family to CSV.

    Returns a dict ``{name: path}``.
    """
    out = Path(out_dir) if out_dir is not None else config.RESULTS_DIR / "solution_dump"
    out.mkdir(parents=True, exist_ok=True)

    parameters = model_data["parameters"]
    variables = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = _periods(model_data)
    destinations = model_data["destinations"]

    aggregated = _formulation(model_data) == "aggregated"

    written = {}

    # ---- first stage ---------------------------------------
    written["initial_fleet"] = _write_csv(
        out / "initial_fleet.csv",
        ["x0", "initial_aircraft_cost"],
        [[_val(variables["x0"]), parameters["initial_aircraft_cost"]]],
    )

    rows = [
        [d, t, _val(variables["n"][d, t])]
        for d in destinations
        for t in periods
        if _val(variables["n"][d, t])
    ]
    written["planned_departures"] = _write_csv(
        out / "planned_departures.csv", ["destination", "period", "n"], rows
    )

    rows = [[t, _val(variables["b"][t])] for t in periods]
    written["reserved_capacity"] = _write_csv(
        out / "reserved_capacity.csv", ["period", "b_reserved"], rows
    )

    rows = [
        [r, d, k, _val(variable)]
        for (r, d, k), variable in variables["protection"].items()
        if _val(variable)
    ]
    written["booking_protection"] = _write_csv(
        out / "booking_protection.csv",
        ["request_period", "destination", "protected_period", "qB"],
        rows,
    )

    # ---- flights -------------------------------------------
    if aggregated:

        rows = [
            [sid, d, k, _val(variables["flights"][sid][d, k])]
            for sid in scenarios
            for d in destinations
            for k in periods
            if _val(variables["flights"][sid][d, k])
        ]

    else:

        rows = [
            [sid, a, d, k, _val(variable)]
            for sid in scenarios
            for (a, d, k), variable in variables["flights"][sid].items()
            if _val(variable)
        ]

    written["realized_flights"] = _write_csv(
        out / "realized_flights.csv",
        (
            ["scenario", "destination", "period", "flights"]
            if aggregated
            else ["scenario", "aircraft", "destination", "period", "flight"]
        ),
        rows,
    )

    # ---- charging ------------------------------------------
    if aggregated:

        rows = [
            [sid, key[0], key[1], round(key[2], 4), t, _val(variables["group_charge"][sid][key, t])]
            for sid in scenarios
            for (key, t) in variables["group_charge"][sid]
            if _val(variables["group_charge"][sid][key, t])
        ]
        written["group_charging"] = _write_csv(
            out / "group_charging.csv",
            ["scenario", "arrival_period", "duration", "energy_kwh", "start", "chi"],
            rows,
        )

        rows = [
            [sid, h, t, _val(variable)]
            for sid in scenarios
            for (h, t), variable in variables["duration_start"][sid].items()
            if _val(variable)
        ]
        written["duration_starts"] = _write_csv(
            out / "duration_starts.csv",
            ["scenario", "duration", "start", "w"],
            rows,
        )

        rows = [
            [sid, k, _val(variables["completions"][sid][k]),
             _val(variables["departure_totals"][sid][k])]
            for sid in scenarios
            for k in periods
        ]
        written["cumulative_flow"] = _write_csv(
            out / "cumulative_flow.csv",
            ["scenario", "period", "completions", "departures"],
            rows,
        )

        rows = [
            [sid, k, _val(variables["deployed_initial"][sid])]
            for sid in scenarios
            for k in [periods[0]]
        ]
        written["deployed_initial"] = _write_csv(
            out / "deployed_initial.csv", ["scenario", "period", "A0"], rows
        )

    else:

        rows = [
            [sid, a, t, _val(variable)]
            for sid in scenarios
            for (a, t), variable in variables["charging"][sid].items()
            if _val(variable)
        ]
        written["charging_starts"] = _write_csv(
            out / "charging_starts.csv",
            ["scenario", "aircraft", "start", "x"],
            rows,
        )

        rows = [
            [sid, a, _val(variable)]
            for sid in scenarios
            for a, variable in variables["u"][sid].items()
            if _val(variable)
        ]
        written["dispatching"] = _write_csv(
            out / "dispatching.csv", ["scenario", "aircraft", "u"], rows
        )

    # ---- passengers ----------------------------------------
    if aggregated:

        rows = [
            [sid, r, d, k, "booked", _val(variable)]
            for sid in scenarios
            for (r, d, k), variable in variables["booked_passengers"][sid].items()
            if _val(variable)
        ]
        rows += [
            [sid, r, d, k, "ondemand", _val(variable)]
            for sid in scenarios
            for (r, d, k), variable in variables["ondemand_passengers"][sid].items()
            if _val(variable)
        ]

    else:

        rows = [
            [sid, r, d, "booked", _val(variable)]
            for sid in scenarios
            for (a, r, d), variable in variables["booked_passengers"][sid].items()
            if _val(variable)
        ]
        rows += [
            [sid, r, d, "ondemand", _val(variable)]
            for sid in scenarios
            for (a, r, d), variable in variables["ondemand_passengers"][sid].items()
            if _val(variable)
        ]

    written["served_passengers"] = _write_csv(
        out / "served_passengers.csv",
        (
            ["scenario", "request_period", "destination", "departure_period", "class", "passengers"]
            if aggregated
            else ["scenario", "request_period", "destination", "class", "passengers"]
        ),
        rows,
    )

    # ---- unserved ------------------------------------------
    rows = [
        [sid, r, d, "booked", _val(variable)]
        for sid in scenarios
        for (r, d), variable in variables["unserved_booked"][sid].items()
        if _val(variable)
    ]
    rows += [
        [sid, r, d, "ondemand", _val(variable)]
        for sid in scenarios
        for (r, d), variable in variables["unserved_ondemand"][sid].items()
        if _val(variable)
    ]
    written["unserved"] = _write_csv(
        out / "unserved.csv",
        ["scenario", "request_period", "destination", "class", "passengers"],
        rows,
    )

    # ---- adjustments and emergency capacity ----------------
    rows = [
        [sid, d, t, _val(variables["added_departures"][sid][d, t]),
         _val(variables["cancelled_departures"][sid][d, t]),
         _val(variables["emergency_capacity"][sid][t])]
        for sid in scenarios
        for d in destinations
        for t in periods
        if _val(variables["added_departures"][sid][d, t])
        or _val(variables["cancelled_departures"][sid][d, t])
        or _val(variables["emergency_capacity"][sid][t])
    ]
    written["recourse_adjustments"] = _write_csv(
        out / "recourse_adjustments.csv",
        ["scenario", "destination", "period", "added", "cancelled", "emergency"],
        rows,
    )

    # ---- charger utilization -------------------------------
    rows = [
        [sid, t, occupancy, occupancy / max(1, int(parameters["charging_facilities"]))]
        for sid in scenarios
        for t, occupancy in charger_occupancy(sid, model_data).items()
    ]
    written["charger_utilization"] = _write_csv(
        out / "charger_utilization.csv",
        ["scenario", "period", "occupied_units", "utilization"],
        rows,
    )

    # ---- summaries -----------------------------------------
    rows, totals, expected_profit = objective_breakdown(model_data)
    written["scenario_summary"] = _write_csv(
        out / "scenario_summary.csv", list(rows[0].keys()), [list(r.values()) for r in rows]
    ) if rows else None

    risk = cvar_summary(model_data)
    written["cvar"] = _write_csv(
        out / "cvar.csv",
        ["scenario", "probability", "loss", "alpha", "var", "cvar", "penalty"],
        [
            [sid, scenarios[sid]["probability"], risk["losses"][sid],
             risk["alpha"], risk["var"], risk["cvar"], risk["penalty"]]
            for sid in scenarios
        ],
    )

    summary = build_summary(model_data)
    written["summary"] = _write_csv(
        out / "summary.csv",
        ["key", "value"],
        [[key, value] for key, value in summary.items() if not isinstance(value, dict)],
    )

    service_rows = scenario_service(model_data)
    written["scenario_service"] = _write_csv(
        out / "scenario_service.csv",
        [
            "scenario",
            "probability",
            "booked_demand",
            "booked_served",
            "booked_unserved",
            "ondemand_demand",
            "ondemand_served",
            "ondemand_unserved",
            "demand_total",
            "served_total",
            "unserved_total",
            "served_share",
        ],
        [
            [
                row["scenario"],
                round(row["probability"], 6),
                round(row["booked_demand"], 4),
                round(row["booked_served"], 4),
                round(row["booked_unserved"], 4),
                round(row["ondemand_demand"], 4),
                round(row["ondemand_served"], 4),
                round(row["ondemand_unserved"], 4),
                round(row["demand_total"], 4),
                round(row["served_total"], 4),
                round(row["unserved_total"], 4),
                round(row["served_share"], 6),
            ]
            for row in service_rows
        ],
    )

    return written


# ------------------------------------------------------------
# console report
# ------------------------------------------------------------

def print_scenario_service(model_data):
    """
    Print the per-scenario served / unserved passenger table.

    Columns are batch counts read from the solved recourse variables::

        scenario  prob   booked  bk_srvd  bk_unsv   ondemd  od_srvd  od_unsv
                  srvd   unsv   rate%

    ``served + unserved = demand`` holds row by row and ``rate%`` is the pooled
    service rate of the scenario. The closing line is the probability-weighted
    mean over the retained scenarios.

    A scenario whose booked served plus booked unserved departs from the
    booking total is flagged: booked demand is scenario independent, so that can
    only be a reporting error.
    """
    rows = scenario_service(model_data)

    if not rows:
        return

    format_row = "  %-8s %6.3f %9.1f %8.1f %8.1f %8.1f %8.1f %8.1f %7.1f %7.1f %7.1f"

    print("\n-- served / unserved passengers by scenario " + "-" * 33)

    print(
        "  %-8s %6s %9s %8s %8s %8s %8s %8s %7s %7s %7s"
        % (
            "scenario",
            "prob",
            "booked",
            "bk_srvd",
            "bk_unsv",
            "ondemd",
            "od_srvd",
            "od_unsv",
            "srvd",
            "unsv",
            "rate%",
        )
    )

    for row in rows:

        print(
            format_row
            % (
                str(row["scenario"]),
                row["probability"],
                row["booked_demand"],
                row["booked_served"],
                row["booked_unserved"],
                row["ondemand_demand"],
                row["ondemand_served"],
                row["ondemand_unserved"],
                row["served_total"],
                row["unserved_total"],
                100.0 * row["served_share"],
            )
        )

    total_probability = sum(row["probability"] for row in rows)

    expected_served = sum(row["probability"] * row["served_total"] for row in rows)

    expected_unserved = sum(row["probability"] * row["unserved_total"] for row in rows)

    if total_probability > 0:
        expected_served /= total_probability
        expected_unserved /= total_probability

    print(
        "  %-8s %6.3f %9s %8s %8s %8s %8s %8s %7.1f %7.1f"
        % (
            "E[row]",
            total_probability,
            "",
            "",
            "",
            "",
            "",
            "",
            expected_served,
            expected_unserved,
        )
    )

    # Booked demand is a first-stage parameter: it must be reproduced in every
    # scenario. Anything else is a reporting bug, not a policy outcome.
    tolerance = 1e-6 * max(1.0, max(row["booked_demand"] for row in rows))

    mismatched = [
        str(row["scenario"])
        for row in rows
        if abs(row["booked_served"] + row["booked_unserved"] - row["booked_demand"])
        > tolerance
    ]

    if mismatched:
        print(
            "  [warning] booked served + unserved does not reproduce the booking "
            "total for scenario(s) %s" % ", ".join(mismatched)
        )

    print(
        "  totals are passengers per scenario; E[row] is the probability-weighted "
        "mean over the %d retained scenarios."
        % len(rows)
    )


def print_solution_report(model_data, objective=None, solver=None, status=None):
    """
    Print the full solution report.

    Sections: first stage, expected recourse components, CVaR, ensemble service
    rates, objective reconciliation, and finally the per-scenario served /
    unserved passenger table (:func:`print_scenario_service`).

    The reconciliation line compares the recomputed objective with the solver
    objective; a mismatch signals a term that is missing from the decomposition.
    """
    summary = build_summary(model_data)
    risk = cvar_summary(model_data)
    service = summary["service"]

    line = "=" * 78

    print("\n" + line)
    print(" SOLUTION REPORT  (%s formulation)" % summary["formulation"])
    print(line)

    if solver or status:
        print("Solver            : %s (%s)" % (solver, status))

    periods = _periods(model_data)

    print("Scenarios         : %d (retained)" % summary["scenarios"])
    print(
        "Horizon           : %d periods x %d min"
        % (len(periods), model_data["parameters"]["bin_size"])
    )

    print("\n-- first stage " + "-" * 62)
    first = first_stage_costs(model_data)
    print(
        "  Initial fleet: %.2f, cost $%.2f"
        % (first["x0"], first["fleet_cost"])
    )
    print(
        "  Reserved charger units (total): %.2f, cost $%.2f"
        % (first["reserved_units"], first["reservation_cost"])
    )
    print(
        "  Planned departures (total): %.2f, cost $%.2f"
        % (first["planned_units"], first["commitment_cost"])
    )

    print("\n-- expected recourse components " + "-" * 46)
    components = summary["components"]
    for label, key in (
        ("Revenue", "revenue"),
        ("Charging energy", "charging"),
        ("Flight operating", "flight"),
        ("Emergency charging", "emergency"),
        ("Added departures", "added"),
        ("Cancelled departures", "cancelled"),
        ("Unserved booked", "unserved_booked"),
        ("Unserved on-demand", "unserved_ondemand"),
    ):
        print("  %-22s %14.2f" % (label, components[key]))

    print("  %-22s %14.2f" % ("Expected profit", summary["expected_profit"]))

    print("\n-- risk " + "-" * 69)
    print("  alpha = %.2f, theta = %.2f" % (risk["alpha"], risk["theta"]))
    print("  VaR(zeta)                    %14.2f" % risk["var"])
    print("  CVaR_alpha(L)                %14.2f" % risk["cvar"])
    print("  theta * CVaR penalty         %14.2f" % risk["penalty"])
    print("  marginal penalty (booked)    %14.2f" % risk["marginal_penalty_booked"])
    print("  marginal penalty (on-demand) %14.2f" % risk["marginal_penalty_ondemand"])

    print("\n-- service " + "-" * 66)
    print("  Booked demand            %10.1f" % service["booked_demand"])
    print("  Booked served            %10.1f   rate %6.2f%%"
          % (service["booked_served"], 100.0 * service["booked_service_rate"]))
    print("  On-demand requests       %10.1f" % service["ondemand_demand"])
    print("  On-demand served         %10.1f   rate %6.2f%%"
          % (service["ondemand_served"], 100.0 * service["ondemand_service_rate"]))

    print("\n-- reconciliation " + "-" * 59)
    print("  recomputed objective         %14.2f" % summary["objective"])
    if objective is not None:
        print("  solver objective             %14.2f" % float(objective))
        difference = abs(float(objective) - float(summary["objective"]))
        print("  |difference|                 %14.2e" % difference)
        if difference > 1e-4 * max(1.0, abs(float(objective))):
            print("  [warning] the decomposition does not reconcile with the solver")

    # Per-scenario served / unserved counts close the report: the block above
    # gives the ensemble rates, this one the batch behind each scenario.
    print_scenario_service(model_data)

    print(line)


# ------------------------------------------------------------
# script entry point
# ------------------------------------------------------------

def _main():  # pragma: no cover

    from experiment_runner import get_formulation
    from model_builder import build_small_instance
    from stochastic_model import solve_model

    model_input = build_small_instance(
        n_destinations=2,
        n_periods=6,
        n_scenarios=3,
        total_passengers=40,
        expected_incoming_aircraft=8,
        max_initial_fleet=3,
        charging_facilities=3,
        max_departures=4,
        booking_share=0.4,
    )

    _, builder, adder = get_formulation("aggregated")

    model_data = builder(model_input)
    problem = adder(model_data)

    results = solve_model(problem, solver="auto", verbose=False)

    print_solution_report(
        model_data,
        objective=results.get("objective"),
        solver=results.get("solver"),
        status=results.get("status"),
    )

    written = export_solution(model_data)

    for name, path in written.items():
        print("  >> %s" % path)


if __name__ == "__main__":
    _main()
