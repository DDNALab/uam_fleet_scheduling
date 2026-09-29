"""Run the hybrid advance-booked + on-demand UAM stochastic workflow."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pulp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    RANDOM_SEED,
    RESULTS_DIR,
    RAW_SCENARIOS,
    OPTIMIZATION_SCENARIOS,
    ACTIVE_DESTINATIONS,
)
from .uam_data_pipeline import build_uam_data
from .scenario_generator import generate_scenarios
from .scenario_reduction import reduce_scenarios
from .model_builder import build_model_input
from .stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model
from .solution_export import print_solution_report


def run_experiment(seed=RANDOM_SEED, solve=True, solver="auto", gap=0.05, time_limit=3600):
    np.random.seed(seed)

    print("\n[1] Building UAM data pipeline...")
    data = build_uam_data()

    # Restrict demand before scenario generation so aircraft supply and passenger
    # demand are based on the same active destination set.
    expected_demand = {
        row["destination"]: row["expected_trips"]
        for _, row in data["od_demand"].iterrows()
        if row["destination"] in ACTIVE_DESTINATIONS
    }
    if not expected_demand:
        raise ValueError("No ACTIVE_DESTINATIONS are present in the OD demand table")

    print("\nExpected OD demand used by optimization:")
    for d, value in expected_demand.items():
        print(f"  {d}: {value:.2f}")

    time_profile = data["time_profile"]["weight"].values

    print("\n[2] Generating hybrid-service scenarios...")
    scenarios = generate_scenarios(
        expected_demand,
        time_profile,
        n_scenarios=RAW_SCENARIOS,
        seed=seed,
    )
    booked_total = sum(scenarios[0].advance_bookings.values())
    print(f"Known advance bookings: {booked_total}")
    print(f"Generated scenarios: {len(scenarios)}")
    for s in scenarios[:3]:
        print(
            f"  S{s.id}: on-demand={sum(s.on_demand_demand.values())}, "
            f"total demand={sum(s.passenger_demand.values())}, aircraft={len(s.aircraft)}"
        )

    print("\n[3] Tail-aware scenario reduction...")
    reduced = reduce_scenarios(scenarios, target_size=OPTIMIZATION_SCENARIOS)
    print(f"Reduced scenarios: {len(reduced)}")
    for s in reduced:
        print(
            f"  S{s.id}: prob={s.probability:.3f}, stress={s.stress_score:.2f}, "
            f"aircraft={len(s.aircraft)}"
        )

    print("\n[4] Building optimization input...")
    model_input = build_model_input(reduced)

    if solve:
        print("\n[5] Building projected stochastic MILP...")
        model_data = build_stochastic_model(model_input)
        model = add_constraints_and_objective(model_data)
        print(
            f"  variables={len(model.variables()):,}, "
            f"constraints={len(model.constraints):,}"
        )

        print("\n[6] Solving...")
        results = solve_model(model, solver=solver, gap_rel=gap, time_limit=time_limit)
        print("\nRESULT")
        print(results)
        print_solution_report(
            model_data,
            objective=results.get("objective"),
            solver=results.get("solver"),
            status=results.get("status"),
        )
    else:
        print("\nOptimization skipped")
        results = {"status": "Skipped", "objective": None}
        model_data = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "known_advance_bookings": booked_total,
        "initial_fleet_allocated": (
            None if model_data is None else int(round(
                float(pulp.value(model_data["variables"]["initial_fleet"]) or 0.0)
            ))
        ),
        "generated_scenarios": len(scenarios),
        "reduced_scenarios": len(reduced),
        "status": results["status"],
        "objective": results["objective"],
    }
    (RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n====== DONE ======")
    return {
        "data": data,
        "scenarios": scenarios,
        "reduced": reduced,
        "model_input": model_input,
        "model_data": model_data,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "gurobi", "highs", "cbc"],
        help="MILP solver. Auto tries Gurobi, HiGHS, then CBC.",
    )
    parser.add_argument("--gap", type=float, default=0.05,
                        help="Relative MILP gap, e.g. 1e-6 for validation")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Solve time limit in seconds")
    args = parser.parse_args()
    run_experiment(seed=args.seed, solve=not args.skip_solve,
                   solver=args.solver, gap=args.gap, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
