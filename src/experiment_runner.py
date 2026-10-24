"""
End-to-end workflow for the two-stage risk-averse UAM scheduling model.

Pipeline: demand -> scenarios -> reduction -> model input -> solve -> report.
Every intermediate artefact is returned, and the formulation is selectable.

Usage::

    python run_experiment.py
    python run_experiment.py --formulation aggregated --theta 0

The modules in ``src/`` are plain top-level modules, not a package, so this
file is also runnable on its own (``python src/experiment_runner.py``): the
script directory is already first on ``sys.path``.
"""

import argparse
import json
from pprint import pprint

import numpy as np

import config
from config import RESULTS_DIR, build_parameters

from demand_model import build_baseline_demand

from scenario_generator import generate_scenarios, scenario_diagnostics

from scenario_reduction import reduction_diagnostics, reduce_scenarios

from model_builder import build_model_input

from solution_export import export_solution, print_solution_report, scenario_service

# solve_model is formulation independent: both models are ordinary PuLP
# problems. The build/add functions are imported per run by get_formulation.
from stochastic_model import solve_model


# ============================================================
# Formulation selection
# ============================================================

# "indexed" is the aircraft-indexed model, "aggregated" its exact projected
# reformulation; both expose the same build/add interface.
FORMULATIONS = ("indexed", "aggregated")

DEFAULT_FORMULATION = "aggregated"


def get_formulation(name):
    """Return ``(name, build, add)`` for a formulation key."""
    name = (name or DEFAULT_FORMULATION).lower()

    if name not in FORMULATIONS:
        raise ValueError(
            "Unknown formulation %r. Choose one of: %s"
            % (name, ", ".join(FORMULATIONS))
        )

    if name == "aggregated":

        from aggregated_model import (
            add_constraints_and_objective as add,
            build_aggregated_model as build,
        )

    else:

        from stochastic_model import (
            add_constraints_and_objective as add,
            build_stochastic_model as build,
        )

    return name, build, add


# ============================================================
# Spatial demand with a graceful fallback
# ============================================================

def build_spatial_demand():
    """
    Build spatial demand from the ACS/gravity pipeline.

    Returns ``(od_table, time_profile)``. If the census shapefile, the ACS
    extract or geopandas is missing, fall back to equal destination shares so
    the pipeline stays runnable; the fallback is announced because it is not
    the case-study data.
    """
    try:

        from uam_data_pipeline import build_uam_data

        data = build_uam_data()

        return data["od_demand"], data["time_profile"]["weight"].values

    except Exception as error:  # pragma: no cover - environment dependent

        print(
            "[warning] spatial demand pipeline unavailable (%s: %s). "
            "Falling back to equal destination shares; this is not the "
            "case-study demand." % (type(error).__name__, error)
        )

        return None, None


# ============================================================
# Main workflow
# ============================================================

def run_experiment(
    seed=None,
    solve=True,
    solver="auto",
    formulation=DEFAULT_FORMULATION,
    theta=None,
    alpha=None,
    booking_share=None,
    raw_scenarios=None,
    optimization_scenarios=None,
    out_of_sample_scenarios=None,
    export=False,
    n_destinations=None,
    n_periods=None,
    max_initial_fleet=None,
    charging_facilities=None,
    max_departures=None,
    initial_aircraft_cost=None,
    cost_profile=None,
    time_limit=None,
    gap=None,
    verbose=True,
):
    """
    Run the full workflow and return every intermediate artefact.

    Keyword arguments override the configuration factors; ``None`` keeps the
    configured value.
    """
    overrides = {}

    if seed is not None:
        overrides["seed"] = int(seed)

    if raw_scenarios is not None:
        overrides["raw_scenarios"] = int(raw_scenarios)

    if optimization_scenarios is not None:
        overrides["optimization_scenarios"] = int(optimization_scenarios)

    if out_of_sample_scenarios is not None:
        overrides["out_of_sample_scenarios"] = int(out_of_sample_scenarios)

    if booking_share is not None:
        overrides["booking_share"] = float(booking_share)

    # ---- hub and fleet overrides ---------------------------------------
    if max_initial_fleet is not None:
        overrides["max_initial_fleet"] = int(max_initial_fleet)

    if charging_facilities is not None:
        overrides["charging_facilities"] = int(charging_facilities)

    if max_departures is not None:
        overrides["max_departures"] = int(max_departures)

    if initial_aircraft_cost is not None:
        overrides["initial_aircraft_cost"] = float(initial_aircraft_cost)

    if cost_profile is not None:
        import config as _config

        if cost_profile not in _config.COST_PROFILES:
            raise ValueError(
                "unknown cost profile %r; choose one of %s"
                % (cost_profile, ", ".join(_config.COST_PROFILES))
            )
        overrides["costs"] = dict(_config.COST_PROFILES[cost_profile])

    if theta is not None or alpha is not None:

        cvar = {}

        if theta is not None:
            cvar["weight"] = float(theta)

        if alpha is not None:
            cvar["alpha"] = float(alpha)

        overrides["cvar"] = cvar

    if time_limit is not None or gap is not None:

        solver_overrides = {}

        if time_limit is not None:
            solver_overrides["time_limit"] = float(time_limit)

        if gap is not None:
            solver_overrides["gap_rel"] = float(gap)

        overrides["solver"] = solver_overrides

    parameters = build_parameters(**overrides)

    # Cost-profile diagnostic: the costs the model receives must be exactly
    # the costs of the selected profile, not a stale copy.

    # ``--cost-profile`` may select a profile other than the module default, so
    # the check is against the *effective* profile.
    selected_profile = (cost_profile or config.COST_PROFILE).lower()

    print("CONFIG FILE:", config.__file__)
    print("ACTIVE PROFILE:", config.COST_PROFILE)

    if selected_profile != str(config.COST_PROFILE).lower():
        print("EFFECTIVE PROFILE (override):", selected_profile)

    print("CONFIGURED COSTS:")
    pprint(config.COST_PROFILES[selected_profile])

    print("COSTS PASSED TO MODEL:")
    pprint(parameters["costs"])

    assert parameters["costs"] == config.COST_PROFILES[selected_profile], (
        "Cost mismatch: parameters contain different costs "
        "from the selected configuration profile."
    )

    destinations = None

    if n_destinations is not None:
        destinations = list(config.ACTIVE_DESTINATIONS[: int(n_destinations)])

    np.random.seed(int(parameters["seed"]))

    # Demand: spatial OD shares, then the temporal period profile.

    if verbose:
        print("\n[1] Building spatial and temporal demand...")

    od_table, profile = build_spatial_demand()

    baseline = build_baseline_demand(
        od_table=od_table,
        profile=profile,
        total_passengers=parameters["total_passengers"],
        booking_share=parameters["booking_share"],
        destinations=destinations,
        n_periods=n_periods,
        bin_size=parameters["bin_size"],
    )

    if verbose:
        print(
            "    destinations=%d periods=%d booked=%d on-demand=%.1f"
            % (
                len(baseline["destinations"]),
                len(baseline["periods"]),
                baseline["booked_total"],
                baseline["ondemand_total"],
            )
        )

    # Booking feasibility: accepted bookings must be protectable by some
    # first-stage departure plan within the hub takeoff limit.

    from feasibility import screen_booking_share

    screen = screen_booking_share(
        baseline["booked"],
        baseline["destinations"],
        baseline["periods"],
        parameters,
    )

    if verbose:
        print(
            "    Eq. (10) data cut : %s"
            % ("passes" if screen["cut_ok"] else "FAILS")
        )
        if screen["cut_violations"]:
            worst = screen["cut_violations"][0]
            print(
                "      first violation: request periods %d..%d need %d departures, "
                "%d slots available"
                % (
                    worst["start"],
                    worst["end"],
                    worst["required"],
                    worst["available"],
                )
            )
        print(
            "    exact first-stage MILP : %s"
            % (
                "feasible"
                if screen["exact_ok"]
                else ("INFEASIBLE" if screen["exact_ok"] is False else "not run")
            )
        )

    if screen["structurally_feasible"] is False:
        print(
            "\n[warning] the accepted booking pattern for phi_B = %.2f is "
            "structurally infeasible: no advance departure plan can protect it "
            "within the hub takeoff throughput. This is a booking-pattern "
            "property, not an operating-policy result (Section 4.2, Remark)."
            % parameters["booking_share"]
        )

    # Correlated scenario generation.

    if verbose:
        print("\n[2] Generating correlated scenarios (Section 5.1)...")

    raw = generate_scenarios(
        baseline,
        n_scenarios=parameters["raw_scenarios"],
        seed=parameters["seed"],
        parameters=parameters,
    )

    raw_diagnostics = scenario_diagnostics(raw)

    if verbose:
        print("    %d raw scenarios" % len(raw))
        print("    on-demand mean %.2f (sd %.2f)"
              % (raw_diagnostics["ondemand_mean"], raw_diagnostics["ondemand_std"]))
        print("    aircraft   mean %.2f (sd %.2f)"
              % (raw_diagnostics["aircraft_mean"], raw_diagnostics["aircraft_std"]))
        print("    arrival SoC mean %.2f, duration counts %s"
              % (raw_diagnostics["soc_mean"], raw_diagnostics["duration_counts"]))

    # Tail-preserving scenario reduction.

    if verbose:
        print("\n[3] Tail-preserving scenario reduction (Section 5.2)...")

    reduced = reduce_scenarios(
        raw,
        target_size=parameters["optimization_scenarios"],
        parameters=parameters,
        booked=baseline["booked"],
    )

    reduced_diagnostics = reduction_diagnostics(reduced)

    if verbose:
        for scenario in reduced:
            print(
                "    S%-3d prob=%.3f stratum=%-7s aircraft=%d"
                % (
                    scenario.id,
                    scenario.probability,
                    scenario.stratum,
                    len(scenario.aircraft),
                )
            )
        print(
            "    N_eff=%.2f  max weight=%.3f  tail=%d central=%d"
            % (
                reduced_diagnostics["effective_scenarios"],
                reduced_diagnostics["max_probability"],
                reduced_diagnostics["tail"],
                reduced_diagnostics["central"],
            )
        )

    # Model-input assembly.

    if verbose:
        print("\n[4] Assembling optimization input...")

    model_input = build_model_input(
        reduced,
        baseline=baseline,
        parameters=parameters,
        destinations=destinations,
    )

    # Solve. Results stay unset when ``solve`` is False.

    results = {"status": "Skipped", "objective": None}

    model_data = None

    if solve:

        formulation_name, build_model, add_constraints = get_formulation(
            formulation
        )

        if verbose:
            print("\n[5] Building the %s formulation..." % formulation_name)

        model_data = build_model(model_input)

        model = add_constraints(model_data)

        from stochastic_model import formulation_size

        size = formulation_size(model)

        if verbose:
            print(
                "    variables=%d (integer=%d, binary=%d) constraints=%d"
                % (
                    size["variables"],
                    size["integer_variables"],
                    size["binary_variables"],
                    size["constraints"],
                )
            )
            print("\n[6] Solving...")

        results = solve_model(
            model,
            solver=solver,
            time_limit=parameters["solver"]["time_limit"],
            gap_rel=parameters["solver"]["gap_rel"],
            verbose=False,
        )

        if verbose:

            print("    %s  objective=%s" % (results["status"], results["objective"]))

            if results.get("gap") is not None:
                print(
                    "    achieved gap %.2f%% (target %.2f%%), dual bound %s"
                    % (
                        100.0 * results["gap"],
                        100.0 * parameters["solver"]["gap_rel"],
                        results.get("bound"),
                    )
                )

            if not results.get("proven_optimal"):
                print(
                    "    [note] incumbent, not a proved optimum -- do not compare "
                    "this objective against a solved formulation."
                )

        if results.get("objective") is None:

            print(
                "\n[warning] the solver returned no feasible solution, so there is "
                "no policy to report. The instance is either structurally "
                "infeasible or too large for the budget."
            )

        else:

            print_solution_report(
                model_data,
                objective=results.get("objective"),
                solver=results.get("solver"),
                status=results.get("status"),
            )

        if export:

            written = export_solution(model_data)

            if verbose:
                print("\n[7] Exported %d CSV files" % len(written))

    # --------------------------------------------------------
    # Save run summary
    # --------------------------------------------------------

    # Per-scenario served / unserved counts. Only meaningful with a feasible
    # solution, so an infeasible run records None rather than zeros.
    service_rows = None

    if model_data is not None and results.get("objective") is not None:

        try:
            service_rows = [
                {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in row.items()
                }
                for row in scenario_service(model_data)
            ]
        except Exception as error:  # pragma: no cover - reporting only
            print(
                "[warning] could not build the per-scenario service table (%s: %s)"
                % (type(error).__name__, error)
            )

    summary = {
        "seed": parameters["seed"],
        "formulation": formulation,
        "generated_scenarios": len(raw),
        "reduced_scenarios": len(reduced),
        "reduction": reduced_diagnostics,
        "raw_diagnostics": {
            key: value
            for key, value in raw_diagnostics.items()
        },
        "booking_feasibility": {
            "cut_ok": screen["cut_ok"],
            "exact_ok": screen["exact_ok"],
            "structurally_feasible": screen["structurally_feasible"],
        },
        "parameters": {
            "booking_share": parameters["booking_share"],
            "cvar": parameters["cvar"],
            "max_initial_fleet": parameters["max_initial_fleet"],
            "charging_facilities": parameters["charging_facilities"],
            "max_departures": parameters["max_departures"],
            "total_passengers": parameters["total_passengers"],
        },
        "status": results["status"],
        "proven_optimal": results.get("proven_optimal"),
        "objective": results["objective"],
        "mip_gap": results.get("gap"),
        "dual_bound": results.get("bound"),
        "size": results.get("size"),
        "scenario_service": service_rows,
    }

    written = None

    try:

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        written = RESULTS_DIR / "run_summary.json"

        written.write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )

    except OSError as error:

        print(
            "[warning] could not write %s (%s). The run summary is available in "
            "the returned dictionary instead." % (written, error)
        )

    return {
        "parameters": parameters,
        "baseline": baseline,
        "feasibility": screen,
        "scenarios": raw,
        "reduced": reduced,
        "model_input": model_input,
        "model_data": model_data,
        "results": results,
        "summary": summary,
    }


# ============================================================
# Exactness comparison
# ============================================================

def run_exactness_check(
    seed=None,
    solver="auto",
    time_limit=None,
    gap=0.0,
    verbose=True,
    **experiment_kwargs
):
    """
    Compare the two formulations on the same instance.

    The models are not compared as two free solves. The projected
    (duration-aggregated) model is solved, its first-stage plan ``(x0, n, b)``
    is imposed on the aircraft-indexed model, and the detailed recourse is
    solved and the objectives compared. Fixing the first stage keeps the check
    independent of alternate optima: a degenerate instance can have several
    optimal first-stage plans. A free detailed solve also tends to stop on its
    time limit, which is a performance result, not a formulation result.

    Parameters
    ----------
    seed, solver, time_limit, gap
        Forwarded to the solver. ``gap=0`` is the right setting for an equality
        test; a loose gap makes both sides incumbents.
    **experiment_kwargs
        Forwarded to :func:`run_experiment` (``n_destinations``, ``n_periods``,
        ``max_initial_fleet``, ``optimization_scenarios``, ...).

    Returns
    -------
    dict
        ``projected``, ``detailed`` (each an outcome dict) and ``difference``.
    """
    import aggregated_model, stochastic_model

    if time_limit is None:
        time_limit = 3600

    built = run_experiment(seed=seed, solve=False, verbose=False, **experiment_kwargs)

    model_input = built["model_input"]

    if verbose:
        print(
            "\n[exactness] instance: dest=%d periods=%d scenarios=%d"
            % (
                len(model_input["destinations"]),
                len(model_input["periods"]),
                len(model_input["scenarios"]),
            )
        )

    # ---- 1. projected formulation ------------------------------------
    projected_data = aggregated_model.build_aggregated_model(model_input)

    projected_problem = aggregated_model.add_constraints_and_objective(
        projected_data
    )

    if verbose:
        size = stochastic_model.formulation_size(projected_problem)
        print(
            "[exactness] projected: vars=%d int=%d cons=%d"
            % (size["variables"], size["integer_variables"], size["constraints"])
        )

    projected = stochastic_model.solve_model(
        projected_problem, solver=solver, time_limit=time_limit, gap_rel=gap,
        verbose=False,
    )

    if verbose:
        print(
            "[exactness] projected: %s obj=%s"
            % (projected["status"], projected["objective"])
        )

    if projected["objective"] is None:
        return {
            "projected": projected,
            "detailed": None,
            "difference": None,
            "reason": "the projected formulation returned no solution",
        }

    # ---- 2. impose its first stage on the detailed formulation --------
    aggregated_variables = projected_data["variables"]

    x0_star = int(round(float(aggregated_variables["x0"].value() or 0.0)))

    n_star = {
        key: int(round(float(variable.value() or 0.0)))
        for key, variable in aggregated_variables["n"].items()
    }

    b_star = {
        key: int(round(float(variable.value() or 0.0)))
        for key, variable in aggregated_variables["b"].items()
    }

    if verbose:
        print(
            "[exactness] projected first stage: x0=%d  sum n=%d  sum b=%d"
            % (x0_star, sum(n_star.values()), sum(b_star.values()))
        )

    detailed_data = stochastic_model.build_stochastic_model(model_input)

    detailed_problem = stochastic_model.add_constraints_and_objective(detailed_data)

    detailed_variables = detailed_data["variables"]

    detailed_problem += (detailed_variables["x0"] == x0_star, "Fix_x0")

    for (destination, period), value in n_star.items():
        detailed_problem += (
            detailed_variables["n"][destination, period] == value,
            stochastic_model.name("Fix_n", period, destination),
        )

    for period, value in b_star.items():
        detailed_problem += (
            detailed_variables["b"][period] == value,
            "Fix_b_%d" % period,
        )

    if verbose:
        size = stochastic_model.formulation_size(detailed_problem)
        print(
            "[exactness] detailed : vars=%d int=%d bin=%d cons=%d"
            % (size["variables"], size["integer_variables"],
               size["binary_variables"], size["constraints"])
        )

    detailed = stochastic_model.solve_model(
        detailed_problem, solver=solver, time_limit=time_limit, gap_rel=gap,
        verbose=False,
    )

    if verbose:
        print(
            "[exactness] detailed : %s obj=%s"
            % (detailed["status"], detailed["objective"])
        )

    if detailed["objective"] is None:
        return {
            "projected": projected,
            "detailed": detailed,
            "difference": None,
            "reason": "the detailed recourse returned no solution",
        }

    difference = abs(float(detailed["objective"]) - float(projected["objective"]))

    scale = max(1.0, abs(float(projected["objective"])))

    close = difference <= 1e-5 * scale

    # Verdict. Equality is two-sided: the aggregation makes the projected
    # optimum an upper bound on the detailed optimum, so a detailed incumbent
    # that matches a proved projected optimum already proves equality. If
    # neither side is proved optimal the comparison is only suggestive.
    if close and projected["proven_optimal"]:

        if detailed["proven_optimal"]:
            verdict = "exact"
            message = (
                "EXACT: both formulations are proved optimal and agree to %.1e."
                % difference
            )
        else:
            verdict = "exact"
            message = (
                "EXACT: the detailed model achieved the projected optimum "
                "(difference %.1e). Its own bound was not closed, but that is "
                "irrelevant here: matching a proved projected optimum from the "
                "detailed side pins the detailed optimum from below, and the "
                "aggregation pins it from above." % difference
            )

    elif close:
        verdict = "matching_incumbents"
        message = (
            "MATCHING INCUMBENTS (difference %.1e): the values agree, but the "
            "projected side is not proved optimal either, so equality is not "
            "established. Re-run with a longer time_limit." % difference
        )

    elif projected["proven_optimal"] and detailed["proven_optimal"]:
        verdict = "mismatch"
        message = (
            "MISMATCH: both are proved optimal but differ by %.3e (relative "
            "%.2e). That is a formulation inconsistency, not a solver effect."
            % (difference, difference / scale)
        )

    else:
        verdict = "inconclusive"
        message = (
            "INCONCLUSIVE: the detailed side is far from the projected value, "
            "so it simply did not solve (difference %.1e). Raise time_limit or "
            "shrink the instance; the detailed model is the larger one by "
            "design." % difference
        )

    if verbose:
        print("[exactness] |projected - detailed| = %.3e" % difference)
        print("[exactness] %s" % message)

    return {
        "projected": projected,
        "detailed": detailed,
        "difference": difference,
        "relative_difference": difference / scale,
        "verdict": verdict,
        "exact": verdict == "exact",
        "message": message,
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage risk-averse stochastic eVTOL scheduling "
            "(Section 4.2 indexed / Section 4.3 duration-aggregated)."
        )
    )

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--skip-solve", action="store_true")

    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "gurobi", "highs", "cbc"],
        help=(
            "'auto' prefers Gurobi, then HiGHS, then CBC, skipping any solver "
            "that cannot hold the model."
        ),
    )

    parser.add_argument(
        "--formulation",
        default=DEFAULT_FORMULATION,
        choices=list(FORMULATIONS),
        help=(
            "'aggregated' (default) is the duration-aggregated cumulative-flow "
            "reformulation of Section 4.3; 'indexed' is the aircraft-indexed "
            "reference of Section 4.2, which is exact but much larger."
        ),
    )

    parser.add_argument("--theta", type=float, default=None,
                        help="CVaR risk weight (default 22; use 0 for TS-RN)")

    parser.add_argument("--alpha", type=float, default=None,
                        help="CVaR confidence level (default 0.90)")

    parser.add_argument("--booking-share", type=float, default=None,
                        help="advance-booking share phi_B (default 0.40)")

    parser.add_argument("--raw-scenarios", type=int, default=None)

    parser.add_argument("--optimization-scenarios", type=int, default=None)

    parser.add_argument("--out-of-sample-scenarios", type=int, default=None)

    parser.add_argument("--n-destinations", type=int, default=None,
                        help="restrict to the first k destinations (validation)")

    parser.add_argument("--n-periods", type=int, default=None,
                        help="override the number of modeled periods")

    parser.add_argument(
        "--max-initial-fleet",
        type=int,
        default=None,
        help=(
            "fleet cap N_fleet (default 40). Lower it to create a scarce ready "
            "fleet -- and to make the indexed formulation solvable, since every "
            "candidate initial aircraft is an explicit variable block there."
        ),
    )

    parser.add_argument(
        "--charging-facilities",
        type=int,
        default=None,
        help="physical charging units m (default 35); lower it to make chargers bind",
    )

    parser.add_argument(
        "--max-departures",
        type=int,
        default=None,
        help="hub takeoff limit per period (default 17); lower it to make it bind",
    )

    parser.add_argument(
        "--initial-aircraft-cost",
        type=float,
        default=None,
        help="per-aircraft initial fleet cost c_fleet (default 40)",
    )

    parser.add_argument(
        "--cost-profile",
        default=None,
        choices=["low", "medium", "high"],
        help="commitment/adjustment cost profile (default medium)",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="solver time limit in seconds (default 3600)",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="relative MIP gap target (default 0.05)",
    )

    parser.add_argument("--export", action="store_true",
                        help="write the solution CSV dump")

    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "run the Table 4 exactness check: solve the projected formulation, "
            "impose its first-stage plan on the aircraft-indexed formulation, "
            "and report whether the objectives agree."
        ),
    )

    args = parser.parse_args()

    if args.compare:

        run_exactness_check(
            seed=args.seed,
            solver=args.solver,
            time_limit=args.time_limit,
            gap=0.0 if args.gap is None else args.gap,
            n_destinations=args.n_destinations,
            n_periods=args.n_periods,
            max_initial_fleet=args.max_initial_fleet,
            charging_facilities=args.charging_facilities,
            max_departures=args.max_departures,
            booking_share=args.booking_share,
            optimization_scenarios=args.optimization_scenarios,
            raw_scenarios=args.raw_scenarios,
        )

        return

    run_experiment(
        seed=args.seed,
        solve=not args.skip_solve,
        solver=args.solver,
        formulation=args.formulation,
        theta=args.theta,
        alpha=args.alpha,
        booking_share=args.booking_share,
        raw_scenarios=args.raw_scenarios,
        optimization_scenarios=args.optimization_scenarios,
        out_of_sample_scenarios=args.out_of_sample_scenarios,
        export=args.export,
        n_destinations=args.n_destinations,
        n_periods=args.n_periods,
        max_initial_fleet=args.max_initial_fleet,
        charging_facilities=args.charging_facilities,
        max_departures=args.max_departures,
        initial_aircraft_cost=args.initial_aircraft_cost,
        cost_profile=args.cost_profile,
        time_limit=args.time_limit,
        gap=args.gap,
    )


if __name__ == "__main__":

    main()
