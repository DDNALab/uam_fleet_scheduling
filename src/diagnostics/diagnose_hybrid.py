"""Reproducible diagnostics for the hybrid booked/on-demand projected UAM MILP.

Run from the directory that contains the project's src/ folder:
    python -m src.diagnostics.diagnose_hybrid --seed 60 --solver highs --test all

The script does not modify the project's source files or the saved random data.
It reuses one reduced scenario set for all controlled sensitivity cases.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
from pathlib import Path
import sys


def import_project():
    # Current layout: src/diagnostics/diagnose_hybrid.py
    project_dir = Path(__file__).resolve().parents[2]
    if not (project_dir / 'src' / 'experiment_runner.py').is_file():
        raise SystemExit('Expected src/experiment_runner.py; run from project root.')
    sys.path.insert(0, str(project_dir))
    try:
        import pulp
        runner = importlib.import_module('src.experiment_runner')
        core = importlib.import_module('src.stochastic_model')
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {exc}. Activate uam_net_opt and install project dependencies.') from exc
    return pulp, runner, core


def build_fixed_input(runner, seed):
    """Generate ONE sample; keep it fixed for the comparisons below."""
    data = runner.build_uam_data()
    full_total = float(data["od_demand"]["expected_trips"].sum())
    expected = {
        row["destination"]: float(row["expected_trips"])
        for _, row in data["od_demand"].iterrows()
        if row["destination"] in runner.ACTIVE_DESTINATIONS
    }
    if not expected:
        raise ValueError("No active destinations were found in the OD input")
    profile = data["time_profile"]["weight"].to_numpy()
    scenarios = runner.generate_scenarios(
        expected, profile, n_scenarios=runner.RAW_SCENARIOS, seed=seed
    )
    reduced = runner.reduce_scenarios(
        scenarios, target_size=runner.OPTIMIZATION_SCENARIOS
    )
    model_input = runner.build_model_input(reduced)
    assert len(profile) == model_input["parameters"]["periods"], (
        "The temporal profile length does not match the optimization horizon."
    )
    assert math.isclose(
        sum(s["probability"] for s in model_input["scenarios"].values()),
        1.0, abs_tol=1e-9
    ), "Reduced-scenario probabilities must sum to one."
    print("\nINPUT CHECK")
    print(f"  Total gravity-model expected passengers (all destinations): {full_total:.2f}")
    print(f"  Expected passengers in ACTIVE destinations: {sum(expected.values()):.2f}")
    print(f"  Advance bookings: {sum(model_input['advance_bookings'].values())}")
    print(f"  Time periods: {model_input['parameters']['periods']}")
    print(f"  Reduced scenarios: {len(model_input['scenarios'])}")
    for sid, scenario in model_input["scenarios"].items():
        print(
            f"    S{sid}: probability={scenario['probability']:.3f}; "
            f"aircraft={len(scenario['aircraft'])}; "
            f"on-demand={sum(scenario['on_demand_demand'].values())}"
        )
    return model_input


def make_instance(pulp, model_module, model_input, service_first=False):
    md = model_module.build_stochastic_model(model_input)
    model = model_module.add_constraints_and_objective(md)
    if service_first:
        v = md["variables"]
        booked = pulp.lpSum(
            sc["probability"] * v["unserved_booked"][sid, r, d]
            for sid, sc in md["scenarios"].items()
            for r in md["periods"] for d in md["destinations"]
        )
        ondemand = pulp.lpSum(
            sc["probability"] * v["unserved_on_demand"][sid, r, d]
            for sid, sc in md["scenarios"].items()
            for r in md["periods"] for d in md["destinations"]
        )
        # An upper bound on physical service: ignore monetary/CVaR terms and
        # minimize passenger failures. Tiny secondary tie-break favors bookings
        # without treating 2 booked passengers as 4 ordinary passengers.
        model.setObjective(-(booked + ondemand + 1e-6 * booked))
    return md, model


def solver_info(model):
    """Read the actual MILP gap rather than trusting PuLP's 'Optimal' label."""
    backend = getattr(model, "solverModel", None)
    if backend is None or not hasattr(backend, "getInfo"):
        return {"solver_reported_gap": None, "backend_status": None}
    try:
        info = backend.getInfo()
        gap = float(info.mip_gap) if hasattr(info, "mip_gap") else None
        if gap is not None and not math.isfinite(gap):
            gap = None
        backend_status = None
        if hasattr(backend, "getModelStatus"):
            raw_status = backend.getModelStatus()
            if hasattr(backend, "modelStatusToString"):
                backend_status = backend.modelStatusToString(raw_status)
            else:
                backend_status = str(raw_status)
        return {"solver_reported_gap": gap, "backend_status": backend_status}
    except (AttributeError, RuntimeError, ValueError):
        return {"solver_reported_gap": None, "backend_status": None}


def metrics(pulp, md):
    v = md["variables"]
    scenarios = md["scenarios"]
    periods, destinations = md["periods"], md["destinations"]
    val = pulp.value
    bookings = sum(md["advance_bookings"].values())
    expected_ondemand = sum(
        sc["probability"] * sum(sc["on_demand_demand"].values())
        for sc in scenarios.values()
    )
    expected_unb = sum(
        sc["probability"] * sum(val(v["unserved_booked"][sid, r, d]) for r in periods for d in destinations)
        for sid, sc in scenarios.items()
    )
    expected_uno = sum(
        sc["probability"] * sum(val(v["unserved_on_demand"][sid, r, d]) for r in periods for d in destinations)
        for sid, sc in scenarios.items()
    )
    exp_flights = sum(
        sc["probability"] * sum(val(v["flights"][sid, d, k]) for d in destinations for k in periods)
        for sid, sc in scenarios.items()
    )
    return {
        "booked": float(bookings),
        "expected_on_demand": float(expected_ondemand),
        "unserved_booked": float(expected_unb),
        "unserved_on_demand": float(expected_uno),
        "total_unserved": float(expected_unb + expected_uno),
        "booked_service_rate": float((bookings - expected_unb) / bookings) if bookings else None,
        "on_demand_service_rate": float((expected_ondemand - expected_uno) / expected_ondemand) if expected_ondemand else None,
        "expected_flights": float(exp_flights),
        "planned_flights": float(sum(val(v["n"][d, k]) for d in destinations for k in periods)),
        "reserved_charger_periods": float(sum(val(v["b"][k]) for k in periods)),
        "initial_fleet_allocated": float(val(v["initial_fleet"])),
        "expected_initial_deployed": float(sum(
            sc["probability"] * val(v["deployed_initial"][sid])
            for sid, sc in scenarios.items()
        )),
    }


def audit(pulp, md, model):
    """Audit EVERY explicit MILP constraint, all integrality, and flow prefixes."""
    by_group = {}
    violations = []
    max_v = 0.0
    for name, con in model.constraints.items():
        lhs = con.value()  # PuLP constraints are normalized to lhs sense 0.
        if lhs is None:
            continue
        if con.sense == pulp.LpConstraintEQ:
            violation = abs(lhs)
        elif con.sense == pulp.LpConstraintLE:
            violation = max(0.0, lhs)
        elif con.sense == pulp.LpConstraintGE:
            violation = max(0.0, -lhs)
        else:
            raise ValueError(f"Unknown PuLP constraint sense: {con.sense}")
        violation = float(violation)
        max_v = max(max_v, violation)
        group = name.split("_")[0]
        by_group[group] = max(violation, by_group.get(group, 0.0))
        if violation > 1e-5:
            violations.append((name, violation))

    max_integrality = 0.0
    max_bound_violation = 0.0
    for var in model.variables():
        x = pulp.value(var)
        if x is None:
            continue
        if var.cat == pulp.LpInteger:
            max_integrality = max(max_integrality, abs(x - round(x)))
        if var.lowBound is not None:
            max_bound_violation = max(max_bound_violation, var.lowBound - x)
        if var.upBound is not None:
            max_bound_violation = max(max_bound_violation, x - var.upBound)

    v = md["variables"]
    maximum_prefix_excess = 0.0
    for sid in md["scenarios"]:
        initial_deployed = pulp.value(v["deployed_initial"][sid])
        for k in md["periods"]:
            cumulative_departures = sum(
                pulp.value(v["flights"][sid, d, t])
                for d in md["destinations"] for t in md["periods"] if t <= k
            )
            cumulative_completions = sum(
                pulp.value(v["w"][sid, h, t])
                for h in md["durations"][sid] for t in md["periods"]
                if t + h <= k
            )
            maximum_prefix_excess = max(
                maximum_prefix_excess,
                float(cumulative_departures - initial_deployed - cumulative_completions)
            )
    # Compare the actual profit objective to its component expressions.
    expr = md["expressions"]
    p = md["parameters"]
    objective_reconciliation = (
        sum(sc["probability"] * pulp.value(expr["scenario_profit"][sid])
            for sid, sc in md["scenarios"].items())
        - pulp.value(expr["initial_fleet_cost"])
        - pulp.value(expr["commitment_cost"])
        - pulp.value(expr["reservation_cost"])
        - p["cvar"]["weight"] * pulp.value(expr["cvar"])
    )
    # For service-first optimization, objective is intentionally different; return
    # original economics as a diagnostic only, not an equality against model.objective.
    return {
        "max_constraint_violation": float(max_v),
        "max_integrality_error": float(max_integrality),
        "max_variable_bound_violation": float(max(0.0, max_bound_violation)),
        "max_completion_prefix_excess": float(maximum_prefix_excess),
        "original_economic_objective_from_components": float(objective_reconciliation),
        "violated_constraints_above_1e-5": sorted(violations, key=lambda t: -t[1])[:10],
        "largest_group_violations": sorted(by_group.items(), key=lambda t: -t[1])[:10],
    }


def solve_case(label, pulp, model_module, model_input, args, service_first=False, do_audit=False):
    md, model = make_instance(pulp, model_module, model_input, service_first)
    result = model_module.solve_model(
        model, solver=args.solver, time_limit=args.time_limit,
        gap_rel=args.gap, threads=args.threads, verbose=args.verbose,
    )
    info = solver_info(model)
    if (info["solver_reported_gap"] is not None
            and info["solver_reported_gap"] > args.gap + 1e-8):
        print(
            f"WARNING: {label}: solver reports gap "
            f"{100 * info['solver_reported_gap']:.5f}% despite PuLP status "
            f"{result['status']!r}; requested {100 * args.gap:.5f}%. "
            "Do not call this run proven optimal."
        )
    if result["status"] != "Optimal" or result.get("objective") is None:
        print(f"\n{label}: NO ACCEPTED SOLUTION: {result}")
        return {"case": label, "status": result["status"], **info}
    data = metrics(pulp, md)
    out = {
        "case": label,
        "status": result["status"],
        "solver_objective": float(result["objective"]),
        **info,
        **data,
    }
    if do_audit:
        detail = audit(pulp, md, model)
        out["audit"] = detail
        if not service_first:
            out["audit"]["objective_reconciliation_error"] = abs(
                float(result["objective"]) - detail["original_economic_objective_from_components"]
            )
    gap_text = "unknown" if info["solver_reported_gap"] is None else f"{100 * info['solver_reported_gap']:.6f}%"
    print(
        f"{label:21s} | gap={gap_text:>11s} | "
        f"unserved B/O={data['unserved_booked']:.3f}/{data['unserved_on_demand']:.3f} | "
        f"total={data['total_unserved']:.3f} | "
        f"booked service={100 * data['booked_service_rate']:.1f}%"
        if data["booked_service_rate"] is not None else
        f"{label:21s} | total unserved={data['total_unserved']:.3f}"
    )
    if do_audit:
        print(
            f"  AUDIT: max row violation={out['audit']['max_constraint_violation']:.2e}, "
            f"integer error={out['audit']['max_integrality_error']:.2e}, "
            f"cumulative excess={out['audit']['max_completion_prefix_excess']:.2e}"
        )
        if out["audit"]["violated_constraints_above_1e-5"]:
            print("  WARNING: violated rows:", out["audit"]["violated_constraints_above_1e-5"])
    return out


def additional_aircraft(model_input, factor=2, early_shift=0):
    """Counterfactual: clone *existing* arrivals in each scenario.

    Retains the original demand, scenario probabilities and initial aircraft;
    cloned arrivals share the source arrival time and SoC, so this is a controlled
    capacity diagnostic, NOT a fresh calibrated supply simulation.
    """
    x = copy.deepcopy(model_input)
    for sc in x["scenarios"].values():
        aircraft = sc["aircraft"]
        original = [copy.deepcopy(v) for v in aircraft.values()]
        if early_shift:
            for a in aircraft.values():
                a["arrival_period"] = max(1, a["arrival_period"] - early_shift)
        next_id = max(aircraft.keys(), default=-1) + 1
        for _ in range(factor - 1):
            for a in original:
                clone = copy.deepcopy(a)
                clone["arrival_period"] = max(1, a["arrival_period"] - early_shift)
                aircraft[next_id] = clone
                next_id += 1
    return x


def ideal_fleet(model_input):
    """Artificial upper bound: pre-position one *initial ready* aircraft per passenger.

    All incoming aircraft are removed for this diagnostic; it is not a real-world
    forecast. All passenger windows and hub takeoff capacities are retained.
    """
    x = copy.deepcopy(model_input)
    bookings = sum(x["advance_bookings"].values())
    maximum_need = max((bookings + sum(sc["on_demand_demand"].values())
                       for sc in x["scenarios"].values()), default=bookings)
    x["parameters"]["initial_fleet_max"] = max(x["parameters"]["initial_fleet_max"],
                                                 int(maximum_need) + 2)
    for sc in x["scenarios"].values():
        sc["aircraft"] = {}
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=60)
    p.add_argument("--solver", choices=("highs", "gurobi", "cbc"), default="highs")
    p.add_argument("--test", choices=("all", "baseline", "max-service", "sensitivity", "ideal"), default="all")
    p.add_argument("--gap", type=float, default=1e-6, help="MIP gap tolerance (default 1e-6)")
    p.add_argument("--time-limit", type=int, default=120, help="Seconds per solve")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="Show full MILP solver logs")
    from src.config import DIAGNOSTICS_DIR
    p.add_argument("--out", type=Path, default=DIAGNOSTICS_DIR)
    args = p.parse_args()
    pulp, runner, model_module = import_project()
    fixed = build_fixed_input(runner, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    print("\nRESULTS (the same reduced scenarios are reused in all cases)")

    def run(label, inp, service_first=False, do_audit=False):
        out = solve_case(label, pulp, model_module, inp, args, service_first, do_audit)
        outputs.append(out)
        return out

    if args.test in ("all", "baseline", "sensitivity"):
        baseline = run("baseline: full objective", fixed, do_audit=True)
    else:
        baseline = None

    if args.test in ("all", "max-service"):
        run("physical max-service", fixed, service_first=True, do_audit=True)

    if args.test in ("all", "sensitivity"):
        x = copy.deepcopy(fixed)
        x["parameters"]["initial_fleet_max"] = 0
        run("no initial-ready fleet", x)

        x = copy.deepcopy(fixed)
        x["parameters"]["initial_fleet_max"] *= 2
        run("2x initial-fleet limit", x)

        x = copy.deepcopy(fixed)
        x["parameters"]["costs"]["initial_fleet"] *= 2
        run("2x initial-fleet cost", x)

        x = copy.deepcopy(fixed)
        x["parameters"]["charging_facilities"] *= 2
        run("2x physical chargers", x)

        run("2x aircraft same times", additional_aircraft(fixed, factor=2))
        run("4x aircraft same times", additional_aircraft(fixed, factor=4))
        run("2x aircraft 30m earlier", additional_aircraft(fixed, factor=2, early_shift=2))

    if args.test in ("all", "ideal"):
        run("ample initial-ready fleet", ideal_fleet(fixed), service_first=True, do_audit=True)

    json_path = args.out / f"hybrid_diagnostics_seed{args.seed}.json"
    csv_path = args.out / f"hybrid_diagnostics_seed{args.seed}.csv"
    json_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    columns = [
        "case", "status", "solver_reported_gap", "solver_objective",
        "booked", "expected_on_demand", "unserved_booked", "unserved_on_demand",
        "total_unserved", "booked_service_rate", "on_demand_service_rate",
        "expected_flights", "planned_flights", "reserved_charger_periods",
        "initial_fleet_allocated", "expected_initial_deployed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outputs)
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")
    if baseline and "audit" in baseline:
        a = baseline["audit"]
        if max(a["max_constraint_violation"], a["max_integrality_error"],
               a["max_variable_bound_violation"], a["max_completion_prefix_excess"]) > 1e-5:
            print("WARNING: baseline audit exceeded 1e-5; inspect JSON details.")
        if a.get("objective_reconciliation_error", 0.0) > 1e-5:
            print("WARNING: expected objective components do not reconcile.")


if __name__ == "__main__":
    main()
