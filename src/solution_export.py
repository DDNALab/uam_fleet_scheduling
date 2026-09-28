"""Reporting utilities for the projected hybrid UAM formulation."""

import csv
from pathlib import Path

import pulp


def _val(x):
    value = pulp.value(x)
    return 0.0 if value is None else float(value)


def _write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    except PermissionError:
        print(f"[warning] could not write {path}; skipping")
    return path


def build_summary(model_data):
    params = model_data["parameters"]
    scenarios = model_data["scenarios"]
    bookings = model_data["advance_bookings"]
    periods = model_data["periods"]
    destinations = model_data["destinations"]
    v = model_data["variables"]
    expr = model_data.get("expressions", {})

    rows = []
    weighted_booked_unserved = 0.0
    weighted_on_unserved = 0.0
    weighted_added = 0.0
    weighted_cancelled = 0.0
    weighted_emergency = 0.0
    weighted_initial_deployed = 0.0

    for sid, scenario in scenarios.items():
        prob = scenario["probability"]
        booked_demand = sum(bookings.values())
        on_demand = sum(scenario["on_demand_demand"].values())
        ub = sum(_val(v["unserved_booked"][sid, r, d]) for r in periods for d in destinations)
        uo = sum(_val(v["unserved_on_demand"][sid, r, d]) for r in periods for d in destinations)
        added = sum(_val(v["added_departures"][sid, d, k]) for d in destinations for k in periods)
        cancelled = sum(_val(v["cancelled_departures"][sid, d, k]) for d in destinations for k in periods)
        emergency = sum(_val(v["emergency_capacity"][sid, t]) for t in periods)
        flights = sum(_val(v["flights"][sid, d, k]) for d in destinations for k in periods)
        loss = _val(expr["scenario_loss"][sid]) if "scenario_loss" in expr else 0.0
        profit = _val(expr["scenario_profit"][sid]) if "scenario_profit" in expr else 0.0

        weighted_booked_unserved += prob * ub
        weighted_on_unserved += prob * uo
        weighted_added += prob * added
        weighted_cancelled += prob * cancelled
        weighted_emergency += prob * emergency
        weighted_initial_deployed += prob * _val(v["deployed_initial"][sid])
        rows.append({
            "scenario": sid,
            "initial_deployed": _val(v["deployed_initial"][sid]),
            "probability": prob,
            "booked_demand": booked_demand,
            "on_demand_demand": on_demand,
            "unserved_booked": ub,
            "unserved_on_demand": uo,
            "flights": flights,
            "added": added,
            "cancelled": cancelled,
            "emergency_charger_periods": emergency,
            "service_loss": loss,
            "scenario_profit": profit,
        })

    initial_allocated = _val(v["initial_fleet"])
    planned = sum(_val(v["n"][d, t]) for d in destinations for t in periods)
    reserved = sum(_val(v["b"][t]) for t in periods)
    cvar = _val(expr.get("cvar", 0.0))
    expected_profit = _val(expr.get("expected_profit", 0.0))
    commitment_cost = _val(expr.get("commitment_cost", 0.0))
    reservation_cost = _val(expr.get("reservation_cost", 0.0))
    fleet_cost = _val(expr.get("initial_fleet_cost", 0.0))
    objective = _val(model_data["model"].objective)

    return {
        "rows": rows,
        "known_advance_bookings": sum(bookings.values()),
        "planned_departures": planned,
        "initial_allocated": initial_allocated,
        "expected_initial_deployed": weighted_initial_deployed,
        "initial_fleet_cost": fleet_cost,
        "reserved_capacity": reserved,
        "weighted_unserved_booked": weighted_booked_unserved,
        "weighted_unserved_on_demand": weighted_on_unserved,
        "weighted_added_departures": weighted_added,
        "weighted_cancelled_departures": weighted_cancelled,
        "weighted_emergency_capacity": weighted_emergency,
        "expected_profit": expected_profit,
        "commitment_cost": commitment_cost,
        "reservation_cost": reservation_cost,
        "cvar": cvar,
        "objective_total": objective,
    }


def print_solution_report(model_data, objective=None, solver=None, status=None):
    summary = build_summary(model_data)
    line = "=" * 68
    print("\n" + line)
    print("HYBRID ADVANCE-BOOKED + ON-DEMAND POLICY")
    print(line)
    print(f"  Known advance bookings             {summary['known_advance_bookings']:12.0f}")
    print(f"  Initial ready aircraft allocated   {summary['initial_allocated']:12.0f}")
    print(f"  E[initial aircraft deployed]       {summary['expected_initial_deployed']:12.2f}")
    print(f"  Initial fleet allocation cost      {summary['initial_fleet_cost']:12.2f}")
    print(f"  Planned departures (sum n_dt)      {summary['planned_departures']:12.0f}")
    print(f"  Reserved charger-periods           {summary['reserved_capacity']:12.0f}")
    print(f"  E[unserved booked]                  {summary['weighted_unserved_booked']:12.2f}")
    print(f"  E[unserved on-demand]               {summary['weighted_unserved_on_demand']:12.2f}")
    print(f"  E[added departures]                 {summary['weighted_added_departures']:12.2f}")
    print(f"  E[cancelled departures]             {summary['weighted_cancelled_departures']:12.2f}")
    print(f"  E[emergency charger-periods]        {summary['weighted_emergency_capacity']:12.2f}")
    print(f"  Service-loss CVaR                   {summary['cvar']:12.2f}")
    print(f"  Model objective                     {summary['objective_total']:12.2f}")
    if objective is not None:
        print(f"  Solver objective                    {objective:12.2f}")
    if status is not None or solver is not None:
        print(f"  solver={solver} status={status}")

    print("\n  per scenario:")
    print("    scn   prob   booked  on-dem  unB   unO   flights  add  can   loss")
    for row in summary["rows"]:
        print(
            "    %-5s %.3f  %6.0f  %6.0f  %4.1f  %4.1f  %7.1f  %4.1f %4.1f  %5.1f"
            % (
                row["scenario"], row["probability"], row["booked_demand"],
                row["on_demand_demand"], row["unserved_booked"],
                row["unserved_on_demand"], row["flights"], row["added"],
                row["cancelled"], row["service_loss"],
            )
        )
    return summary


def export_solution(model_data, out_dir="outputs/results/solution_dump"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    periods = model_data["periods"]
    destinations = model_data["destinations"]
    scenarios = model_data["scenarios"]
    groups = model_data["groups"]
    feasible = model_data["feasible_starts"]
    bookings = model_data["advance_bookings"]
    v = model_data["variables"]

    written = {}
    written["planned_departures"] = _write_csv(
        out / "planned_departures.csv",
        ["destination", "period", "n_planned"],
        [[d, t, _val(v["n"][d, t])] for d in destinations for t in periods if _val(v["n"][d, t])],
    )
    written["reserved_capacity"] = _write_csv(
        out / "reserved_capacity.csv",
        ["period", "b_reserved"],
        [[t, _val(v["b"][t])] for t in periods],
    )
    written["advance_bookings"] = _write_csv(
        out / "advance_bookings.csv",
        ["request_period", "destination", "booked_passengers"],
        [[r, d, value] for (r, d), value in sorted(bookings.items()) if value],
    )

    charge_rows = []
    flight_rows = []
    service_rows = []
    for sid in scenarios:
        for gid, g in groups[sid].items():
            for t in feasible[sid][gid]:
                value = _val(v["chi"][sid, gid, t])
                if value:
                    charge_rows.append([
                        sid, gid, g["arrival_period"], g["duration"], g["energy_kwh"], t, value
                    ])
        for d in destinations:
            for k in periods:
                f = _val(v["flights"][sid, d, k])
                if f:
                    flight_rows.append([sid, d, k, f])
        for r in periods:
            for d in destinations:
                service_rows.append([
                    sid, r, d,
                    bookings.get((r, d), 0),
                    scenarios[sid]["on_demand_demand"].get((r, d), 0),
                    _val(v["unserved_booked"][sid, r, d]),
                    _val(v["unserved_on_demand"][sid, r, d]),
                ])

    written["charging_flow"] = _write_csv(
        out / "charging_flow.csv",
        ["scenario", "group", "arrival_period", "duration", "energy_kwh", "start", "chi"],
        charge_rows,
    )
    written["flights"] = _write_csv(
        out / "flights.csv", ["scenario", "destination", "period", "flights"], flight_rows
    )
    written["service"] = _write_csv(
        out / "service.csv",
        ["scenario", "request_period", "destination", "booked", "on_demand", "unserved_booked", "unserved_on_demand"],
        service_rows,
    )
    written["summary"] = build_summary(model_data)
    return written
