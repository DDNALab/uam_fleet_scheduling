"""
Run complete UAM stochastic optimization workflow.

Pipeline:

1. Data preparation
2. Scenario generation
3. Scenario reduction
4. Model input construction
5. Two-stage stochastic MILP
6. Reporting
"""


import argparse
import json
from pathlib import Path
import sys

import numpy as np


# Support direct execution
if __package__ in {None, ""}:

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1])
    )

    __package__ = "coord_schedule_uam"



from .config import (
    RANDOM_SEED,
    RESULTS_DIR,
    RAW_SCENARIOS,
    OPTIMIZATION_SCENARIOS
)


from .uam_data_pipeline import (
    build_uam_data
)


from .scenario_generator import (
    generate_scenarios
)


from .scenario_reduction import (
    reduce_scenarios
)


from .model_builder import (
    build_model_input
)


from .stochastic_model import (
    build_stochastic_model,
    add_constraints_and_objective,
    solve_model
)

from .solution_export import (
    print_solution_report
)




# ============================================================
# Main workflow
# ============================================================


def run_experiment(
        seed=RANDOM_SEED,
        solve=True,
        solver="auto"
):


    np.random.seed(seed)



    # --------------------------------------------------------
    # 1. DATA PIPELINE
    # --------------------------------------------------------

    print("\n[1] Building UAM data pipeline...")


    data = build_uam_data()



    print("\nExpected OD demand:")

    print(
        data["od_demand"]
        .to_string(index=False)
    )



    # --------------------------------------------------------
    # 2. SCENARIO GENERATION
    # --------------------------------------------------------

    print("\n[2] Generating scenarios...")


    expected_demand = {

        row["destination"]:
        row["expected_trips"]

        for _, row
        in data["od_demand"].iterrows()

    }



    time_profile = (

        data["time_profile"]
        ["weight"]
        .values

    )



    scenarios = generate_scenarios(

        expected_demand,

        time_profile,

        n_scenarios=RAW_SCENARIOS,

        seed=seed

    )



    print(
        f"Generated scenarios: {len(scenarios)}"
    )



    for s in scenarios[:3]:

        print(
            f"""
Scenario {s.id}

Passengers:
{sum(s.passenger_demand.values())}

Aircraft:
{len(s.aircraft)}
"""
        )



    # --------------------------------------------------------
    # 3. SCENARIO REDUCTION
    # --------------------------------------------------------

    print("\n[3] Scenario reduction...")


    reduced = reduce_scenarios(

        scenarios,

        target_size=OPTIMIZATION_SCENARIOS

    )


    print(
        f"Reduced scenarios: {len(reduced)}"
    )


    for s in reduced:

        print(
            f"S{s.id} "
            f"prob={s.probability:.3f} "
            f"aircraft={len(s.aircraft)}"
        )



    # --------------------------------------------------------
    # 4. MODEL INPUT
    # --------------------------------------------------------

    print("\n[4] Building optimization input...")


    model_input = build_model_input(

        reduced

    )



    print(
        "Model input created"
    )



    # --------------------------------------------------------
    # 5. STOCHASTIC MILP
    # --------------------------------------------------------

    if solve:


        print(
            "\n[5] Building stochastic MILP..."
        )


        model_data = build_stochastic_model(

            model_input

        )


        model = add_constraints_and_objective(

            model_data

        )


        print(
            "\n[6] Solving..."
        )


        results = solve_model(

            model,

            solver=solver

        )


        print(
            "\nRESULT"
        )

        print(
            results
        )


        # ------------------------------------------------
        # Objective breakdown + passenger service, on screen
        # ------------------------------------------------

        print_solution_report(

            model_data,

            objective=results.get("objective"),

            solver=results.get("solver"),

            status=results.get("status")

        )



    else:

        print(
            "\nOptimization skipped"
        )

        results={

            "status":
            "Skipped",

            "objective":
            None

        }



    # --------------------------------------------------------
    # 6. SAVE REPORT
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    summary={

        "seed":
        seed,

        "generated_scenarios":
        len(scenarios),

        "reduced_scenarios":
        len(reduced),

        "status":
        results["status"],

        "objective":
        results["objective"]

    }


    (
        RESULTS_DIR /
        "run_summary.json"

    ).write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"

    )


    print(
        "\n====== DONE ======"
    )



    return {

        "data":
        data,

        "scenarios":
        scenarios,

        "reduced":
        reduced,

        "model_input":
        model_input,

        "results":
        results

    }




# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )


    parser.add_argument(
        "--skip-solve",
        action="store_true"
    )


    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "gurobi", "highs", "cbc"],
        help=(
            "MILP solver. 'auto' (default) prefers Gurobi, then HiGHS, "
            "then CBC, skipping any solver that cannot hold the model."
        )
    )


    args = parser.parse_args()



    run_experiment(

        seed=args.seed,

        solve=not args.skip_solve,

        solver=args.solver

    )



if __name__ == "__main__":

    main()