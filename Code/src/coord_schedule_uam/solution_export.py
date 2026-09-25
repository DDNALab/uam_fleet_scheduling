"""
solution_export.py
==================

Turn a solved two-stage stochastic model into inspectable tables.

After a successful solve, `export_solution(model_data, out_dir)` writes one CSV
per decision-variable family, plus a per-scenario summary, an objective
breakdown and a CVaR summary. Everything is recomputed from the solved
variable values, so the breakdown can be checked against `model.objective`.

Run as a script to build, solve and export in one go:

    python -m coord_schedule_uam.solution_export

Outputs land in outputs/results/solution_dump/ by default.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .stochastic_model import energy_price_per_kwh


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _val(var):
    """Solved value of a PuLP variable; 0.0 when unset or None."""
    if var is None:
        return 0.0
    x = var.value()
    return 0.0 if x is None else float(x)


def _write_csv(path, header, rows):
    """
    Write one CSV. A locked file (typically the CSV being held open in Excel)
    must not abort the run -- the console report already carries these numbers,
    so warn and carry on.
    """
    path = Path(path)
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
    except PermissionError:
        print("[warning] could not write %s (file is open in another program, "
              "e.g. Excel); skipping." % path)
    return path


def _energy_needed(info, params):
    """kWh required to bring this aircraft from its arrival SoC to soc_min."""
    return (
        params["battery_capacity"]
        * max(0.0, params["soc_min"] - info["initial_soc"])
        / 100.0
    )


# ------------------------------------------------------------
# formulation-aware accessors
#
# The indexed model (stochastic_model.py) and the aggregated model
# (aggregated_model.py) store the same economic quantities under different
# variable families. These adapters expose one interface to both, so every
# reporting function below stays formulation-agnostic.
# ------------------------------------------------------------

def _formulation(model_data):
    return model_data.get("formulation", "indexed")


def _served(sid, model_data, params, durations=None):
    """
    Passenger-equivalent flow per destination, for revenue and service totals.

    Returns a dict destination -> served passengers.
    """

    vars_ = model_data["variables"]
    destinations = model_data["destinations"]
    periods = list(model_data["periods"])

    out = {d: 0.0 for d in destinations}

    if _formulation(model_data) == "aggregated":

        for (r, d, k) in vars_["passengers"][sid]:
            out[d] += _val(vars_["passengers"][sid][r, d, k])

        return out

    aircraft = model_data["scenarios"][sid]["aircraft"]

    for j in aircraft:
        for t in periods:
            for d in destinations:
                out[d] += _val(vars_["passengers"][sid][j][t][d])

    return out


def _charging_cost(sid, model_data, params):
    """Energy cost of charging, priced at the time-of-use rate."""

    vars_ = model_data["variables"]
    periods = list(model_data["periods"])

    def price_at(t):
        return energy_price_per_kwh(
            (params["horizon"]["start"]
             + (t - 1) * params["bin_size"]) / 60.0,
            params,
        )

    if _formulation(model_data) == "aggregated":

        groups = vars_["groups"][sid]

        total = 0.0
        for (g, t) in vars_["rho"][sid]:
            total += _val(vars_["rho"][sid][g, t]) * groups[g]["energy"] * price_at(t)

        return total

    aircraft = model_data["scenarios"][sid]["aircraft"]

    total = 0.0
    for j, info in aircraft.items():
        energy = _energy_needed(info, params)
        for t in periods:
            if _val(vars_["charging"][sid][j][t]) > 0.5:
                total += energy * price_at(t)

    return total


def _unserved(sid, model_data):
    """Total unserved passengers in a scenario."""

    vars_ = model_data["variables"]
    destinations = model_data["destinations"]
    periods = list(model_data["periods"])

    if _formulation(model_data) == "aggregated":
        return sum(
            _val(vars_["unserved"][sid][r, d])
            for (r, d) in vars_["unserved"][sid]
        )

    return sum(
        _val(vars_["unserved"][sid][t][d])
        for t in periods
        for d in destinations
    )


def _activated_aircraft(sid, model_data):
    """
    Number of activated aircraft.

    The aggregated model projects aircraft identities out, so the closest
    equivalent is the number of aircraft that start charging.
    """

    vars_ = model_data["variables"]

    if _formulation(model_data) == "aggregated":

        # Every charged aircraft completes and departs exactly once (Eq. 29),
        # so total completions over the horizon is the activated count.
        return sum(
            _val(vars_["completions"][sid][k])
            for k in list(model_data["periods"])
        )

    aircraft = model_data["scenarios"][sid]["aircraft"]

    return sum(_val(vars_["activation"][sid][j]) for j in aircraft)


# ------------------------------------------------------------
# objective decomposition
# ------------------------------------------------------------

def objective_breakdown(model_data):
    """
    Recompute every objective term from the solved variables.

    Returns (per_scenario_rows, totals, expected_profit).
    """

    params = model_data["parameters"]
    vars_ = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = list(model_data["periods"])
    destinations = model_data["destinations"]

    costs = params["costs"]
    fares = params["destination_fares"]

    rows = []
    totals = {
        "revenue": 0.0,
        "charging": 0.0,
        "unserved_penalty": 0.0,
        "added": 0.0,
        "cancelled": 0.0,
        "emergency": 0.0,
    }
    expected_profit = 0.0

    for sid, scenario in scenarios.items():
        prob = scenario["probability"]

        by_destination = _served(sid, model_data, params)

        revenue = sum(
            by_destination[d] * fares.get(d, 0.0) for d in destinations
        )

        served = sum(by_destination.values())

        charging = _charging_cost(sid, model_data, params)

        unserved = _unserved(sid, model_data)

        added = sum(
            _val(vars_["added_departures"][sid][d][t])
            for d in destinations
            for t in periods
        )
        cancelled = sum(
            _val(vars_["cancelled_departures"][sid][d][t])
            for d in destinations
            for t in periods
        )
        emergency = sum(
            _val(vars_["emergency_capacity"][sid][t])
            for t in periods
        )

        activated = _activated_aircraft(sid, model_data)

        demand = sum(scenario["passenger_demand"].values())

        profit = (
            revenue
            - charging
            - unserved * costs["unserved"]
            - added * costs["added_departure"]
            - cancelled * costs["cancelled_departure"]
            - emergency * costs["emergency_capacity"]
        )

        expected_profit += prob * profit

        totals["revenue"] += prob * revenue
        totals["charging"] += prob * charging
        totals["unserved_penalty"] += prob * unserved * costs["unserved"]
        totals["added"] += prob * added * costs["added_departure"]
        totals["cancelled"] += prob * cancelled * costs["cancelled_departure"]
        totals["emergency"] += prob * emergency * costs["emergency_capacity"]

        rows.append({
            "scenario": sid,
            "probability": round(prob, 6),
            "n_aircraft": len(scenario["aircraft"]),
            "n_activated": int(round(activated)),
            "demand": demand,
            "served": round(served, 4),
            "unserved": round(unserved, 4),
            "revenue": round(revenue, 4),
            "charging_cost": round(charging, 4),
            "unserved_penalty": round(unserved * costs["unserved"], 4),
            "added_cost": round(added * costs["added_departure"], 4),
            "cancelled_cost": round(cancelled * costs["cancelled_departure"], 4),
            "emergency_cost": round(emergency * costs["emergency_capacity"], 4),
            "scenario_profit": round(profit, 4),
        })

    return rows, totals, expected_profit


# ------------------------------------------------------------
# CVaR
# ------------------------------------------------------------

def cvar_summary(model_data):
    """Recompute L_s, VaR, CVaR and the penalised CVaR term."""

    params = model_data["parameters"]
    vars_ = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = list(model_data["periods"])
    destinations = model_data["destinations"]

    alpha = params["cvar"]["alpha"]
    theta = params["cvar"]["weight"]
    capacity = params["evtol_capacity"]

    losses = {}
    for sid in scenarios:
        unserved = _unserved(sid, model_data)
        cancelled = sum(
            _val(vars_["cancelled_departures"][sid][d][t])
            for d in destinations
            for t in periods
        )
        losses[sid] = unserved + capacity * cancelled

    probs = np.array([scenarios[s]["probability"] for s in scenarios], dtype=float)
    loss_vec = np.array([losses[s] for s in scenarios], dtype=float)

    order = np.argsort(-loss_vec)
    cumulative = 0.0
    var = float(loss_vec[order[0]]) if len(order) else 0.0
    for idx in order:
        cumulative += probs[idx]
        if cumulative >= (1.0 - alpha):
            var = float(loss_vec[idx])
            break

    excess = float(
        sum(p * max(0.0, l - var) for p, l in zip(probs, loss_vec))
    )
    cvar = var + (1.0 / (1.0 - alpha)) * excess

    return {
        "losses": losses,
        "var": var,
        "cvar": cvar,
        "alpha": alpha,
        "theta": theta,
        "multiplier": theta / (1.0 - alpha),
        "penalty": theta * cvar,
    }


# ------------------------------------------------------------
# main export
# ------------------------------------------------------------

def export_solution(model_data, out_dir="outputs/results/solution_dump"):
    """
    Write every decision-variable family to CSV.

    Returns a dict of {name: path}.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = model_data["parameters"]
    vars_ = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = list(model_data["periods"])
    destinations = model_data["destinations"]

    written = {}

    # ---- first stage: committed departures -------------------
    rows = []
    for d in destinations:
        for t in periods:
            v = _val(vars_["n"][(d, t)])
            if v:
                rows.append([d, t, v])
    written["committed_departures"] = _write_csv(
        out / "committed_departures.csv",
        ["destination", "period", "n_committed"], rows)

    # ---- first stage: reserved charging capacity -------------
    rows = [[t, _val(vars_["b"][t])] for t in periods]
    written["reserved_capacity"] = _write_csv(
        out / "reserved_capacity.csv",
        ["period", "b_reserved"], rows)

    # ---- scenario-level variable families -------------------
    flight_rows, pax_rows = [], []
    unserved_rows, adj_rows, emg_rows = [], [], []

    aggregated = _formulation(model_data) == "aggregated"

    if aggregated:

        # The aggregated model has no aircraft identities: report group
        # charging, duration-start totals and the cumulative flows instead.
        group_rows, w_rows = [], []
        flow_rows, comp_rows = [], []

        for sid, scenario in scenarios.items():

            groups = vars_["groups"][sid]

            for (g, t) in vars_["rho"][sid]:
                v = _val(vars_["rho"][sid][g, t])
                if v:
                    group_rows.append([
                        sid, g[0], g[1], round(g[2], 4), t, v
                    ])

            for (h, t), var in vars_["w"][sid].items():
                v = _val(var)
                if v:
                    w_rows.append([sid, h, t, v])

            for k in periods:
                c = _val(vars_["completions"][sid][k])
                d_ = _val(vars_["departures_total"][sid][k])
                if c or d_:
                    flow_rows.append([sid, k, c, d_])

            for d in destinations:
                for k in periods:
                    f = _val(vars_["flights"][sid][d, k])
                    if f:
                        flight_rows.append([sid, d, k, f])

            for (r, d, k) in vars_["passengers"][sid]:
                p = _val(vars_["passengers"][sid][r, d, k])
                if p:
                    pax_rows.append([sid, r, d, k, p])

            for (r, d) in vars_["unserved"][sid]:
                u = _val(vars_["unserved"][sid][r, d])
                if u:
                    unserved_rows.append([sid, r, d, u])

            for d in destinations:
                for t in periods:
                    ap = _val(vars_["added_departures"][sid][d][t])
                    cp = _val(vars_["cancelled_departures"][sid][d][t])
                    if ap or cp:
                        adj_rows.append([sid, d, t, ap, cp])

            for t in periods:
                e = _val(vars_["emergency_capacity"][sid][t])
                if e:
                    emg_rows.append([sid, t, e])

        written["group_charging"] = _write_csv(
            out / "group_charging.csv",
            ["scenario", "arrival_period", "duration", "energy_kwh",
             "start_period", "rho_assigned"], group_rows)

        written["duration_starts"] = _write_csv(
            out / "duration_starts.csv",
            ["scenario", "duration", "start_period", "w_operations"], w_rows)

        written["cumulative_flow"] = _write_csv(
            out / "cumulative_flow.csv",
            ["scenario", "period", "C_completions", "D_departures"], flow_rows)

        written["flights"] = _write_csv(
            out / "flights.csv",
            ["scenario", "destination", "period", "f_realized"], flight_rows)

        written["passengers"] = _write_csv(
            out / "passengers.csv",
            ["scenario", "arrival_period", "destination", "departure_period",
             "y_served"], pax_rows)

        written["unserved"] = _write_csv(
            out / "unserved.csv",
            ["scenario", "arrival_period", "destination", "phi_unserved"],
            unserved_rows)

        written["adjustments"] = _write_csv(
            out / "adjustments.csv",
            ["scenario", "destination", "period", "v_plus_added",
             "v_minus_cancelled"], adj_rows)

        written["emergency_capacity"] = _write_csv(
            out / "emergency_capacity.csv",
            ["scenario", "period", "b_plus_emergency"], emg_rows)

    else:

        charge_rows, act_rows, elig_rows = [], [], []

        for sid, scenario in scenarios.items():
            aircraft = scenario["aircraft"]

            for j, info in aircraft.items():
                act_rows.append([
                    sid, j, info["arrival_period"],
                    round(info["initial_soc"], 4),
                    _val(vars_["activation"][sid][j]),
                    _val(vars_["departure_time"][sid][j]),
                    round(_energy_needed(info, params), 4),
                ])
                for t in periods:
                    v = _val(vars_["charging"][sid][j][t])
                    if v:
                        charge_rows.append([sid, j, t, v])

            for j in aircraft:
                for d in destinations:
                    for t in periods:
                        z = _val(vars_["departures"][sid][j][d][t])
                        if z:
                            flight_rows.append([sid, j, d, t, z])
                        y = _val(vars_["eligibility"][sid][j][t][d])
                        if y:
                            elig_rows.append([sid, j, t, d, y])
                        p = _val(vars_["passengers"][sid][j][t][d])
                        if p:
                            pax_rows.append([sid, j, t, d, p])

            for t in periods:
                for d in destinations:
                    u = _val(vars_["unserved"][sid][t][d])
                    if u:
                        unserved_rows.append([sid, t, d, u])

            for d in destinations:
                for t in periods:
                    ap = _val(vars_["added_departures"][sid][d][t])
                    cp = _val(vars_["cancelled_departures"][sid][d][t])
                    if ap or cp:
                        adj_rows.append([sid, d, t, ap, cp])

            for t in periods:
                e = _val(vars_["emergency_capacity"][sid][t])
                if e:
                    emg_rows.append([sid, t, e])

        written["activation"] = _write_csv(
            out / "activation.csv",
            ["scenario", "aircraft", "arrival_period", "initial_soc",
             "u_activated", "F_departure_time", "energy_needed_kwh"], act_rows)

        written["charging"] = _write_csv(
            out / "charging.csv",
            ["scenario", "aircraft", "period", "x_charge_start"], charge_rows)

        written["departures"] = _write_csv(
            out / "departures.csv",
            ["scenario", "aircraft", "destination", "period", "z_departure"],
            flight_rows)

        written["eligibility"] = _write_csv(
            out / "eligibility.csv",
            ["scenario", "aircraft", "period", "destination", "Y_eligible"],
            elig_rows)

        written["passengers"] = _write_csv(
            out / "passengers.csv",
            ["scenario", "aircraft", "period", "destination", "xi_assigned"],
            pax_rows)

        written["unserved"] = _write_csv(
            out / "unserved.csv",
            ["scenario", "period", "destination", "phi_unserved"],
            unserved_rows)

        written["adjustments"] = _write_csv(
            out / "adjustments.csv",
            ["scenario", "destination", "period", "v_plus_added",
             "v_minus_cancelled"], adj_rows)

        written["emergency_capacity"] = _write_csv(
            out / "emergency_capacity.csv",
            ["scenario", "period", "b_plus_emergency"], emg_rows)

    # ---- summaries -----------------------------------------
    rows, totals, expected_profit = objective_breakdown(model_data)

    written["scenario_summary"] = _write_csv(
        out / "scenario_summary.csv",
        ["scenario", "probability", "n_aircraft", "n_activated", "demand",
         "served", "unserved", "revenue", "charging_cost", "unserved_penalty",
         "added_cost", "cancelled_cost", "emergency_cost", "scenario_profit"],
        [list(r.values()) for r in rows])

    # Reuse the shared summary so the CSV and the console report can never
    # drift apart.
    summary = build_summary(model_data)
    cv = summary["cvar"]
    pax = summary["passengers"]

    breakdown = [
        ["expected_revenue", round(summary["totals"]["revenue"], 4)],
        ["expected_charging_cost", round(-summary["totals"]["charging"], 4)],
        ["expected_unserved_penalty",
         round(-summary["totals"]["unserved_penalty"], 4)],
        ["expected_added_cost", round(-summary["totals"]["added"], 4)],
        ["expected_cancelled_cost", round(-summary["totals"]["cancelled"], 4)],
        ["expected_emergency_cost", round(-summary["totals"]["emergency"], 4)],
        ["expected_profit", round(summary["expected_profit"], 4)],
        ["first_stage_commitment_cost",
         round(-summary["commitment_cost"], 4)],
        ["first_stage_reservation_cost",
         round(-summary["reservation_cost"], 4)],
        ["cvar_var", round(cv["var"], 4)],
        ["cvar_value", round(cv["cvar"], 4)],
        ["cvar_multiplier", round(cv["multiplier"], 4)],
        ["cvar_penalty", round(-cv["penalty"], 4)],
        ["objective_total", round(summary["objective_total"], 4)],
        ["passengers_demand", pax["demand"]],
        ["passengers_served", round(pax["served"], 4)],
        ["passengers_unserved", round(pax["unserved"], 4)],
        ["passengers_weighted_unserved", round(pax["weighted_unserved"], 4)],
        ["passengers_service_rate", round(pax["service_rate"], 6)],
        ["committed_departures", int(round(summary["committed_departures"]))],
        ["reserved_capacity", int(round(summary["reserved_capacity"]))],
    ]

    written["objective_breakdown"] = _write_csv(
        out / "objective_breakdown.csv", ["term", "value"], breakdown)

    summary["out_dir"] = str(out)
    written["summary"] = summary

    return written


# ------------------------------------------------------------
# console report (no files written)
# ------------------------------------------------------------

def build_summary(model_data):
    """
    Assemble the objective breakdown, CVaR block and passenger service totals
    from a solved model. Pure computation -- writes nothing.

    Returns a dict used by both `print_solution_report` and `export_solution`.
    """

    params = model_data["parameters"]
    vars_ = model_data["variables"]
    scenarios = model_data["scenarios"]
    periods = list(model_data["periods"])
    destinations = model_data["destinations"]

    rows, totals, expected_profit = objective_breakdown(model_data)
    cv = cvar_summary(model_data)

    committed = sum(
        _val(vars_["n"][(d, t)]) for d in destinations for t in periods
    )
    reserved = sum(_val(vars_["b"][t]) for t in periods)

    commitment_cost = committed * params["costs"]["commitment"]
    reservation_cost = reserved * params["costs"]["charging_reservation"]

    served_total = sum(r["served"] for r in rows)
    unserved_total = sum(r["unserved"] for r in rows)
    demand_total = sum(r["demand"] for r in rows)
    weighted_unserved = sum(
        r["probability"] * r["unserved"] for r in rows
    )

    return {
        "rows": rows,
        "totals": totals,
        "expected_profit": expected_profit,
        "commitment_cost": commitment_cost,
        "reservation_cost": reservation_cost,
        "committed_departures": committed,
        "reserved_capacity": reserved,
        "cvar": cv,
        "objective_total": (expected_profit - commitment_cost
                            - reservation_cost - cv["penalty"]),
        "passengers": {
            "demand": demand_total,
            "served": served_total,
            "unserved": unserved_total,
            "weighted_unserved": weighted_unserved,
            "service_rate": (served_total / demand_total) if demand_total else 1.0,
        },
    }


def print_solution_report(model_data, objective=None, solver=None,
                          status=None):
    """
    Print the objective breakdown, service totals and CVaR block to stdout.

    Mirrors what `export_solution` writes to CSV, so the same numbers are
    visible without opening any file.
    """

    s = build_summary(model_data)
    pax = s["passengers"]
    cv = s["cvar"]

    line = "=" * 64

    print("\n" + line)
    print("OBJECTIVE BREAKDOWN")
    print(line)

    for term, value in [
        ("Expected revenue", s["totals"]["revenue"]),
        ("Expected charging cost", -s["totals"]["charging"]),
        ("Expected unserved penalty", -s["totals"]["unserved_penalty"]),
        ("Expected added-departure cost", -s["totals"]["added"]),
        ("Expected cancelled-departure cost", -s["totals"]["cancelled"]),
        ("Expected emergency-capacity cost", -s["totals"]["emergency"]),
    ]:
        print("  %-34s %12.2f" % (term, value))

    print("  %-34s %12s" % ("", "-" * 12))
    print("  %-34s %12.2f" % ("EXPECTED PROFIT", s["expected_profit"]))

    print("  %-34s %12.2f" % ("First-stage commitment cost",
                              -s["commitment_cost"]))
    print("  %-34s %12.2f" % ("First-stage reservation cost",
                              -s["reservation_cost"]))
    print("  %-34s %12.2f" % ("CVaR risk penalty", -cv["penalty"]))

    print("  %-34s %12s" % ("", "-" * 12))
    print("  %-34s %12.2f" % ("OBJECTIVE (model)", s["objective_total"]))

    if objective is not None:
        print("  %-34s %12.2f" % ("OBJECTIVE (solver)", objective))

    print()
    print(line)
    print("PASSENGER SERVICE")
    print(line)
    print("  %-34s %12d" % ("Total demand (sum over scenarios)",
                            int(pax["demand"])))
    print("  %-34s %12.2f" % ("Served", pax["served"]))
    print("  %-34s %12.2f" % ("Unserved", pax["unserved"]))
    print("  %-34s %11.1f%%" % ("Service rate", 100.0 * pax["service_rate"]))
    print("  %-34s %12.2f" % ("Probability-weighted unserved",
                              pax["weighted_unserved"]))

    print()
    print("  per scenario:")
    print("    %-6s %-8s %-8s %-8s %-8s %-10s"
          % ("scn", "prob", "demand", "served", "unserved", "profit"))
    for r in s["rows"]:
        print("    %-6s %-8.3f %-8d %-8.2f %-8.2f %-10.2f"
              % (r["scenario"], r["probability"], r["demand"],
                 r["served"], r["unserved"], r["scenario_profit"]))

    print()
    print(line)
    print("CVaR")
    print(line)
    print("  %-34s %s" % ("L_s per scenario",
                          {k: round(v, 4) for k, v in cv["losses"].items()}))
    print("  %-34s %12.4f" % ("VaR at alpha=%.2f" % cv["alpha"], cv["var"]))
    print("  %-34s %12.4f" % ("CVaR", cv["cvar"]))
    print("  %-34s %12.1f" % ("lambda/(1-alpha) multiplier",
                              cv["multiplier"]))
    print("  %-34s %12.4f" % ("CVaR penalty", cv["penalty"]))

    print()
    print(line)
    print("FIRST-STAGE DECISIONS")
    print(line)
    print("  %-34s %12d" % ("Committed departures (sum n_dt)",
                            int(round(s["committed_departures"]))))
    print("  %-34s %12d" % ("Reserved charging units (sum b_t)",
                            int(round(s["reserved_capacity"]))))

    if status is not None or solver is not None:
        print()
        print("  solver=%s  status=%s" % (solver, status))

    return s


# ------------------------------------------------------------
# run standalone: build -> solve -> export
# ------------------------------------------------------------

def _main():

    import pulp

    from .config import RANDOM_SEED, RAW_SCENARIOS, TARGET_SCENARIOS

    from .uam_data_pipeline import build_uam_data
    from .scenario_generator import generate_scenarios
    from .scenario_reduction import reduce_scenarios
    from .model_builder import build_model_input
    from .stochastic_model import (
        build_stochastic_model,
        add_constraints_and_objective,
        solve_model,
    )

    data = build_uam_data()
    expected = {
        row["destination"]: row["expected_trips"]
        for _, row in data["od_demand"].iterrows()
    }
    profile = data["time_profile"]["weight"].values

    scenarios = generate_scenarios(
        expected, profile, n_scenarios=RAW_SCENARIOS, seed=RANDOM_SEED)
    reduced = reduce_scenarios(scenarios, target_size=TARGET_SCENARIOS)

    model_data = build_stochastic_model(build_model_input(reduced))
    model = add_constraints_and_objective(model_data)

    result = solve_model(model)
    print_solution_report(
        model_data,
        objective=result.get("objective"),
        solver=result.get("solver"),
        status=result.get("status"),
    )

    written = export_solution(model_data)
    print()
    print("written to", written["summary"]["out_dir"])


if __name__ == "__main__":
    _main()
