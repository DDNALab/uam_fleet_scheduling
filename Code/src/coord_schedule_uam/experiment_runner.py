"""Run the complete data, scenario, optimization, and reporting workflow."""

import argparse
import json
from pathlib import Path
import sys

# Support both recommended package execution and IDE "Run Python File" actions.
# When this file is launched directly, Python does not assign it a package, so
# relative imports such as ``from .config`` would otherwise fail.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "coord_schedule_uam"

import numpy as np

from .config import (
    N_SCENARIOS,
    RESULTS_DIR,
    SEED,
    SHAPEFILE_PATH,
    TARGET_SCENARIOS,
    TRIPS_BY_DEST,
)
from .data_loading import load_all_data
from .demand_model import build_demand_model
from .model_construction import (
    build_stochastic_model_input,
    conditional_soc_sampling_per_evtol,
)
from .scenario_generation import (
    build_bin_destination_pmfs,
    generate_joint_demand_scenarios,
)
from .scenario_reduction import scenario_reduction_backward_with_evtol_filter
from .solution_algorithm import build_and_solve_stochastic_model_aggregated
from .visualization import plot_reduced_scenario_analysis, plot_scenario_arrivals

def run_experiment(seed=SEED, create_plots=True, solve=True):
    np.random.seed(seed)

    # 1. DATA LOADING
    print("\n[1] Loading data...")
    data = load_all_data(SHAPEFILE_PATH)
    verts, uam_by_vert = data["vertiports"], data["uam_by_vertiport"]

    # 2. DEMAND MODEL
    print("\n[2] Building demand model...")
    demand = build_demand_model(verts, uam_by_vert, vertiports=data["vertiports_dict"])
    print(demand["flows"][["origin", "destination", "distance_km", "trips"]].to_string(index=False))
    print(f"\nTotal Trips: {demand['total_trips']:.2f} | CBD Boost: {demand['cbd_boost']:.2f}x")

    # 3. SCENARIO GENERATION
    print("\n[3] Generating scenarios...")
    passenger_arrivals = demand["passenger_arrivals"]
    evtol_arrivals     = demand["evtol_arrivals"]
    passenger_dest     = demand["passenger_dest_names"]

    bin_pmfs, _ = build_bin_destination_pmfs(
        passenger_arrivals=passenger_arrivals,
        passenger_dest=passenger_dest,
        target_shares=TRIPS_BY_DEST
    )

    destination_data = {
        row["destination"]: {
            "fare":     round(20 + 3 * row["distance_km"] * 0.621371, 2),
            "distance": row["distance_km"] * 0.621371
        }
        for _, row in demand["flows"].iterrows()
    }

    scenarios = generate_joint_demand_scenarios(
        passenger_arrivals=passenger_arrivals,
        evtol_arrivals=evtol_arrivals,
        destination_data=destination_data,
        bin_pmfs=bin_pmfs,
        n_scenarios=N_SCENARIOS
    )

    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios[:3]:
        top_dest = max(s['destination_counts'], key=s['destination_counts'].get) if s['destination_counts'] else "N/A"
        print(f"  S{s['id']} | passengers={s['total_passengers']} | evtols={s['total_evtols']} | top dest={top_dest}")

    # 4. SCENARIO REDUCTION
    print("\n[4] Reducing scenarios...")
    reduced = scenario_reduction_backward_with_evtol_filter(
        scenarios,
        target_size=TARGET_SCENARIOS,
        plot=False,
    )

    print(f"Reduced to {len(reduced)} scenarios:")
    for s in reduced:
        print(f"  S{s['id']} | prob={s['probability']:.3f} | passengers={s['total_passengers']} | evtols={s['total_evtols']}")

    # 5. SOC SAMPLING
    print("\n[5] Conditional SoC sampling...")
    reduced = conditional_soc_sampling_per_evtol(reduced, assignment_case=1, plot=False)

    for s in reduced:
        avg_soc = np.mean(list(s["evtol_socs"].values())) if s["evtol_socs"] else 0
        print(f"  S{s['id']} | avg_soc={avg_soc:.1f}% | load_ratio={s['avg_load_ratio']:.2f}")

    # 6. MODEL CONSTRUCTION
    print("\n[6] Building stochastic model input...")

    dest_names = list(destination_data.keys())
    destination_mapping = {
        "id_to_name": {i: n for i, n in enumerate(dest_names)},
        "name_to_id": {n: i for i, n in enumerate(dest_names)}
    }
    destination_fares = {i: destination_data[n]["fare"] for i, n in enumerate(dest_names)}

    stochastic_input = build_stochastic_model_input(
        reduced_scenarios=reduced,
        bin_pmfs=bin_pmfs,
        destination_fares=destination_fares,
        destination_mapping=destination_mapping,
        verbose=True
    )

    for sid, s in stochastic_input["scenarios"].items():
        print(f"  S{sid} | prob={s['probability']:.3f} | evtols={s['n_evtols']} | passengers={s['w_passengers']}")

    # 7. OPTIMIZATION
    if solve:
        print("\n[7] Solving optimization model...")
        model, results = build_and_solve_stochastic_model_aggregated(
            stochastic_input,
            verbose=True,
        )
        print(f"  Status    : {results['status']}")
        objective = results["objective"]
        print(
            f"  Objective : {objective:.4f}"
            if objective is not None
            else "  Objective : unavailable"
        )
    else:
        print("\n[7] Optimization skipped.")
        model = None
        objective = None
        results = {"status": "Skipped", "status_code": None, "objective": None}

    # 8. VISUALIZATION
    if create_plots:
        print("\n[8] Visualization...")
        plot_scenario_arrivals(reduced)
        plot_reduced_scenario_analysis(reduced, scenarios)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "generated_scenarios": len(scenarios),
        "reduced_scenarios": len(reduced),
        "solver_status": results["status"],
        "objective": objective,
    }
    (RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


    print("\n====== DONE ======")
    return {
        "data": data,
        "demand": demand,
        "scenarios": scenarios,
        "reduced": reduced,
        "stochastic_input": stochastic_input,
        "model": model,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run the workflow without generating interactive HTML plots.",
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Build solver inputs but skip the computationally expensive MILP solve.",
    )
    args = parser.parse_args()
    run_experiment(
        seed=args.seed,
        create_plots=not args.no_plots,
        solve=not args.skip_solve,
    )


if __name__ == "__main__":
    main()
