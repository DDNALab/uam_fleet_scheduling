"""Train fixed first-stage UAM policies; test on INDEPENDENT paired scenarios.

Training optimizes shared x0, n[d,t], b[t]. OOS evaluation fixes all three and
optimizes a separate stage-2 recourse model for EACH independently sampled
scenario; recourse does NOT choose a new initial fleet or advance plan.

For a CVaR policy, keep the training-optimal VaR threshold zeta fixed when
solving each independent recourse problem. This retains the tail-sensitive
recourse rule corresponding to the trained offline policy. The empirical OOS
CVaR is estimated separately from the realized losses (not from training zeta).

Due to its full-scenario recourse information, this evaluates the manuscript's
offline two-stage approximation, NOT an implementable online dispatcher.

Examples:
 python -m src.diagnostics.oos_evaluate --scope active-renormalized \
   --passengers 20 --train-raw 60 --train-retained 5 --test-scenarios 30 \
   --policies rn cvar bo --seed-train 60 --seed-test 1060
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pulp

from src import config
from src.diagnostics.demand_design import SCOPES, expected_demand_from_geodata
from src.scenario_generator import (
    Aircraft, Scenario, generate_scenarios,
)
from src.scenario_reduction import reduce_scenarios
from src.model_builder import build_model_input
from src.policy_benchmarks import (
    booking_only_planning_input, extract_first_stage_solution, fix_first_stage_decisions,
)
from src.stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model
from src.diagnostics.fleet_sensitivity import check_rows_and_integrality, check_inventory

POLICIES = ("rn", "cvar", "bo", "ev")
FIELDS = [
    "policy", "test_index", "test_scenario_id", "status", "gap", "solve_seconds",
    "booked_demand", "on_demand_demand", "unserved_booked", "unserved_on_demand",
    "initial_deployed", "realized_departures", "added_departures", "cancelled_departures",
    "emergency_charger_periods", "scenario_loss", "scenario_profit_after_penalties",
    "fixed_stage1_cost", "net_payoff_after_penalties", "cash_profit_before_unserved_penalties",
    "max_constraint_violation", "max_integrality_error", "max_bound_violation", "error",
]


def _v(expr):
    val = pulp.value(expr)
    if val is None or not math.isfinite(float(val)):
        raise RuntimeError("Solver did not return a finite primal solution")
    return float(val)


def _certified(res, gap_target, solver):
    if res["status"] != "Optimal" or res.get("objective") is None:
        return False
    if solver == "highs":
        gap = res.get("solver_reported_gap")
        return gap is not None and math.isfinite(gap) and gap <= gap_target + 1e-7
    return True  # Other solver's own Optimal status; log its available gap.


def _stage1_json(stage):
    return {
        "x0": stage["x0"],
        "n": [[d, t, val] for (d, t), val in sorted(stage["n"].items())],
        "b": [[t, val] for t, val in sorted(stage["b"].items())],
    }


def _stage1_restore(raw):
    return {
        "x0": int(raw["x0"]),
        "n": {(d, int(t)): int(val) for d, t, val in raw["n"]},
        "b": {int(t): int(val) for t, val in raw["b"]},
    }


def _fixed_stage1_cost(stage, params):
    c = params["costs"]
    return (
        c["initial_fleet"] * stage["x0"]
        + c["commitment"] * sum(stage["n"].values())
        + c["charging_reservation"] * sum(stage["b"].values())
    )


def _mean_scenario(raw_scenarios, bookings, profile):
    """Integerized expected-value scenario; NOT a representative observed scenario.

    On-demand means are rounded by request cell; mean incoming-aircraft counts
    are allocated to arrival periods via largest remainders, preserving the
    rounded mean TOTAL of arrivals. Mean SoC is per arrival period where data
    are available, otherwise the pooled arrival-SoC mean.
    """
    n = len(raw_scenarios)
    keys = sorted({key for s in raw_scenarios for key in s.on_demand_demand})
    means = {key: int(round(sum(s.on_demand_demand.get(key, 0) for s in raw_scenarios) / n))
             for key in keys}
    tcount = len(profile)
    counts = np.zeros(tcount, dtype=float)
    soc_by_t = [[] for _ in range(tcount)]
    pooled_soc = []
    for s in raw_scenarios:
        for a in s.aircraft:
            idx = int((a.arrival_time - config.HORIZON_START) // config.BIN_SIZE)
            if 0 <= idx < tcount:
                counts[idx] += 1.0 / n
                soc_by_t[idx].append(float(a.initial_soc))
                pooled_soc.append(float(a.initial_soc))
    integral_counts = np.floor(counts).astype(int)
    remaining = int(round(float(counts.sum()))) - int(integral_counts.sum())
    if remaining > 0:
        order = np.argsort(-(counts-integral_counts), kind="stable")
        integral_counts[order[:remaining]] += 1
    default_soc = float(np.mean(pooled_soc)) if pooled_soc else float(config.SOC_BASE_MEAN)
    aircraft = []
    for idx, count in enumerate(integral_counts):
        soc = float(np.mean(soc_by_t[idx])) if soc_by_t[idx] else default_soc
        for _ in range(int(count)):
            aircraft.append(Aircraft(id=len(aircraft),
                         arrival_time=config.HORIZON_START + idx*config.BIN_SIZE,
                         initial_soc=soc))
    return Scenario(id=-1, probability=1., advance_bookings=dict(bookings),
                    on_demand_demand=means, aircraft=aircraft,
                    common_factor=np.zeros(tcount))


def train_policy(policy, full_mi, raw_scenarios, profile, args):
    planning_mi = copy.deepcopy(full_mi)
    if policy in ("rn", "bo", "ev"):
        planning_mi["parameters"]["cvar"]["weight"] = 0.0
    if policy == "bo":
        planning_mi = booking_only_planning_input(planning_mi)
    if policy == "ev":
        ev = _mean_scenario(raw_scenarios, full_mi["advance_bookings"], profile)
        planning_mi = build_model_input([ev])
        # Match ALL economic settings with stochastic training model.
        planning_mi["parameters"] = copy.deepcopy(full_mi["parameters"])
        planning_mi["parameters"]["cvar"]["weight"] = 0.0
    md = build_stochastic_model(planning_mi)
    model = add_constraints_and_objective(md)
    start = time.perf_counter()
    res = solve_model(model, solver=args.solver, gap_rel=args.train_gap,
                      time_limit=args.train_time_limit, verbose=args.verbose)
    seconds = time.perf_counter() - start
    if not _certified(res, args.train_gap, args.solver):
        raise RuntimeError(f"Training policy {policy} missed requested gap/status: {res}")
    checks = check_rows_and_integrality(model)
    if max(checks.values()) > 1e-5:
        raise RuntimeError(f"Training {policy} solution violates constraints/integrality: {checks}")
    check_inventory(md)
    stage = extract_first_stage_solution(md)
    zeta = _v(md["variables"]["zeta"]) if policy == "cvar" else None
    return {
        "name": policy, "first_stage": _stage1_json(stage),
        "training_status": res["status"], "training_gap": res["solver_reported_gap"],
        "training_objective": res["objective"], "training_seconds": seconds,
        "training_zeta": zeta,
        "number_train_variables": res["n_variables"],
        "number_train_constraints": res["n_constraints"],
    }


def eval_policy_one(policy, trained, scenario, index, params, args):
    mi = build_model_input([scenario])
    # Each one-scenario OOS recourse problem must carry unit probability.
    # The raw independent test generator initially assigns probability 1/N.
    mi["scenarios"][scenario.id]["probability"] = 1.0
    mi["parameters"] = copy.deepcopy(params)
    # Neutral policies use neutral recourse; risk-averse policy uses the SAME
    # training-fixed VaR threshold so recourse decisions can be optimized one
    # independent scenario at a time without leaking OOS tail distribution.
    mi["parameters"]["cvar"]["weight"] = (
        params["cvar"]["weight"] if policy == "cvar" else 0.0)
    md = build_stochastic_model(mi)
    model = add_constraints_and_objective(md)
    stage = _stage1_restore(trained["first_stage"])
    fix_first_stage_decisions(md, stage)
    if policy == "cvar":
        if trained["training_zeta"] is None:
            raise RuntimeError("Risk-averse training policy missing zeta")
        model += md["variables"]["zeta"] == float(trained["training_zeta"]), "FixTrainingCVaRThreshold"

    row = {"policy": policy, "test_index": index,
           "test_scenario_id": scenario.id,
           "booked_demand": sum(scenario.advance_bookings.values()),
           "on_demand_demand": sum(scenario.on_demand_demand.values()),
           "fixed_stage1_cost": _fixed_stage1_cost(stage, mi["parameters"])}
    start = time.perf_counter()
    try:
        res = solve_model(model, solver=args.solver, gap_rel=args.eval_gap,
                          time_limit=args.eval_time_limit, verbose=args.verbose)
        row["solve_seconds"] = time.perf_counter() - start
        row["status"] = res["status"]
        row["gap"] = res.get("solver_reported_gap")
        if not _certified(res, args.eval_gap, args.solver):
            row["error"] = "No certified target-gap solution; omitted from performance statistics"
            return row
        checks = check_rows_and_integrality(model)
        row.update(checks)
        if max(checks.values()) > 1e-5:
            raise RuntimeError(f"Solution audit failed: {checks}")
        check_inventory(md)
        v = md["variables"]
        sid = scenario.id
        periods, destinations = md["periods"], md["destinations"]
        row["initial_deployed"] = _v(v["deployed_initial"][sid])
        row["unserved_booked"] = sum(_v(v["unserved_booked"][sid,r,d]) for r in periods for d in destinations)
        row["unserved_on_demand"] = sum(_v(v["unserved_on_demand"][sid,r,d]) for r in periods for d in destinations)
        row["realized_departures"] = sum(_v(v["flights"][sid,d,k]) for d in destinations for k in periods)
        row["added_departures"] = sum(_v(v["added_departures"][sid,d,k]) for d in destinations for k in periods)
        row["cancelled_departures"] = sum(_v(v["cancelled_departures"][sid,d,k]) for d in destinations for k in periods)
        row["emergency_charger_periods"] = sum(_v(v["emergency_capacity"][sid,t]) for t in periods)
        expr = md["expressions"]
        row["scenario_loss"] = _v(expr["scenario_loss"][sid])
        row["scenario_profit_after_penalties"] = _v(expr["scenario_profit"][sid])
        row["net_payoff_after_penalties"] = (row["scenario_profit_after_penalties"]
                                            - row["fixed_stage1_cost"])
        c = params["costs"]
        penalties = (c["unserved_booked"]*row["unserved_booked"]
                     + c["unserved_on_demand"]*row["unserved_on_demand"])
        row["cash_profit_before_unserved_penalties"] = row["net_payoff_after_penalties"] + penalties
        row["error"] = ""
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = repr(exc)
        row["solve_seconds"] = time.perf_counter() - start
    return row


def empirical_cvar(values, alpha):
    """CVaR of the empirical equally weighted OOS loss distribution."""
    x = sorted([float(v) for v in values], reverse=True)
    if not x or not 0 <= alpha < 1:
        raise ValueError("Need nonempty losses and 0<=alpha<1")
    mass = (1-alpha) * len(x)
    full = int(math.floor(mass+1e-10))
    fractional = mass - full
    total = sum(x[:full]) + (fractional*x[full] if full < len(x) else 0.)
    return total / mass


def _mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def _bootstrap_paired(rows_a, rows_b, key, reps=400, seed=2026):
    common = sorted(set(rows_a) & set(rows_b))
    if len(common) < 3:
        return None
    diffs = np.array([float(rows_a[k][key])-float(rows_b[k][key]) for k in common])
    rng = np.random.default_rng(seed)
    sample = rng.integers(0,len(diffs),size=(reps,len(diffs)))
    means = diffs[sample].mean(axis=1)
    return {"n_paired": len(common), "mean_difference": float(diffs.mean()),
            "paired_bootstrap_95pct": [float(v) for v in np.quantile(means,[0.025,0.975])]}


def summarize(rows, trained, manifest):
    # Use same test scenario IDs for ALL policy comparisons, excluding any
    # incomplete or uncertified scenario solve from ALL policy samples.
    policy_names = list(trained)
    good = {}
    for row in rows:
        if row.get("status") != "Optimal" or row.get("error"):
            continue
        if manifest["solver"] == "highs":
            if row.get("gap") in (None, "") or float(row["gap"]) > manifest["eval_gap"] + 1e-7:
                continue
        good[(row["policy"],int(row["test_index"]))] = row
    common = [i for i in range(manifest["test_scenarios"])
              if all((p,i) in good for p in policy_names)]
    summaries = {}
    for p in policy_names:
        sample = [good[p,i] for i in common]
        if not sample:
            summaries[p] = {"n_complete_paired":0}
            continue
        bd = sum(float(v["booked_demand"]) for v in sample)
        od = sum(float(v["on_demand_demand"]) for v in sample)
        summaries[p] = {
            "n_complete_paired": len(sample),
            "allocated_initial_aircraft": trained[p]["first_stage"]["x0"],
            "planned_departures": sum(item[2] for item in trained[p]["first_stage"]["n"]),
            "reserved_charger_periods": sum(item[1] for item in trained[p]["first_stage"]["b"]),
            "mean_cash_profit_excluding_unserved_penalties": _mean(sample,"cash_profit_before_unserved_penalties"),
            "mean_net_payoff_including_unserved_penalties": _mean(sample,"net_payoff_after_penalties"),
            "mean_unserved_booked": _mean(sample,"unserved_booked"),
            "mean_unserved_on_demand": _mean(sample,"unserved_on_demand"),
            "booked_service_fraction": 1.-sum(float(v["unserved_booked"]) for v in sample)/bd if bd else None,
            "on_demand_service_fraction": 1.-sum(float(v["unserved_on_demand"]) for v in sample)/od if od else None,
            "empirical_loss_cvar": empirical_cvar([v["scenario_loss"] for v in sample],manifest["alpha"]),
            "mean_added_departures": _mean(sample,"added_departures"),
            "mean_cancelled_departures": _mean(sample,"cancelled_departures"),
            "mean_emergency_charger_periods": _mean(sample,"emergency_charger_periods"),
            "mean_initial_deployed": _mean(sample,"initial_deployed"),
        }
        summaries[p]["oos_risk_adjusted_objective"] = (
            summaries[p]["mean_net_payoff_including_unserved_penalties"]
            - manifest["risk_weight"] * summaries[p]["empirical_loss_cvar"])
    contrasts = {}
    for p in policy_names:
        if p == "rn":
            continue
        if "rn" not in policy_names:
            break
        a={i:good[p,i] for i in common};b={i:good["rn",i] for i in common}
        contrasts[f"{p}_minus_rn_net_payoff"] = _bootstrap_paired(a,b,"net_payoff_after_penalties")
        contrasts[f"{p}_minus_rn_unserved_booked"] = _bootstrap_paired(a,b,"unserved_booked")
    return {
        "n_common_completed":len(common), "n_requested": manifest["test_scenarios"],
        "warning": ("Fewer than 50 paired evaluation scenarios; tail CVaR is descriptive only"
                    if len(common)<50 else None),
        "policies":summaries, "paired_differences":contrasts,
        "notes": ["Each test scenario has full-scenario offline recourse (not online dispatch).",
                  "Net payoff includes unserved-demand penalties; cash profit excludes those penalties.",
                  "For CVaR recourse, the VaR threshold remains fixed at its training value.",
                  "Only common test scenarios with certified solves for every policy enter comparative statistics."],
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope",choices=SCOPES,default="active-renormalized")
    ap.add_argument("--passengers",type=float,default=config.TOTAL_PASSENGER_TRIPS)
    ap.add_argument("--booking-share",type=float,default=config.ADVANCE_BOOKING_FRACTION)
    ap.add_argument("--seed-train",type=int,default=60)
    ap.add_argument("--seed-test",type=int,default=1060)
    ap.add_argument("--train-raw",type=int,default=60)
    ap.add_argument("--train-retained",type=int,default=5)
    ap.add_argument("--test-scenarios",type=int,default=50)
    ap.add_argument("--policies",nargs="+",choices=POLICIES,default=["rn","cvar","bo"])
    ap.add_argument("--solver",choices=["highs","gurobi","cbc","auto"],default="highs")
    ap.add_argument("--train-gap",type=float,default=0.01)
    ap.add_argument("--eval-gap",type=float,default=1e-5)
    ap.add_argument("--train-time-limit",type=int,default=900)
    ap.add_argument("--eval-time-limit",type=int,default=90)
    ap.add_argument("--max-test",type=int,default=None,help="Process first N tests now; resume later")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--overwrite",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--out",type=Path,default=None,
                    help="Output directory; defaults to outputs/oos/<scope>_...; never overwrites existing run without --overwrite/--resume")
    args=ap.parse_args()
    if args.seed_train==args.seed_test:
        ap.error("Training and OOS seeds MUST be different")
    if not 0<=args.booking_share<=1 or not 2<=args.train_retained<=args.train_raw or args.test_scenarios<1:
        ap.error("Booking share [0,1], 2<=train-retained<=train-raw, test-scenarios>=1")
    if args.max_test is not None and not 1<=args.max_test<=args.test_scenarios:
        ap.error("--max-test must lie between 1 and --test-scenarios")
    if args.overwrite and args.resume:
        ap.error("Choose --overwrite or --resume, not both")
    policies=list(dict.fromkeys(args.policies))
    output=args.out or config.OUTPUT_DIR/"oos"/(
        f"{args.scope}_p{args.passengers:g}_train{args.seed_train}_test{args.seed_test}")
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"manifest.json"
    training_path=output/"training_policies.json"
    rows_path=output/"oos_rows.csv"
    summary_path=output/"oos_summary.json"

    expected, profile=expected_demand_from_geodata(scope=args.scope,passengers=args.passengers)
    manifest={
        "scope":args.scope,"passengers":args.passengers,
        "expected_optimization_passengers":sum(expected.values()),
        "destinations":expected,"booking_share":args.booking_share,
        "seed_train":args.seed_train,"seed_test":args.seed_test,
        "train_raw":args.train_raw,"train_retained":args.train_retained,
        "test_scenarios":args.test_scenarios,"policies":policies,
        "solver":args.solver,"train_gap":args.train_gap,"eval_gap":args.eval_gap,
        "train_time_limit":args.train_time_limit,"eval_time_limit":args.eval_time_limit,
        "horizon_minutes":config.HORIZON_MINUTES,"bin_size":config.BIN_SIZE,
        "los_minutes":config.LOS_MINUTES,"aircraft_passenger_ratio":config.AIRCRAFT_PASSENGER_RATIO,
        "physical_chargers":config.M_FACILITIES,
        "max_departures_per_period":config.MAX_DEPARTURES_PER_PERIOD,
        "initial_fleet_cap":config.INITIAL_FLEET_MAX,
        "initial_fleet_cost":config.INITIAL_FLEET_COST,
        "alpha":config.CVaR_ALPHA,"risk_weight":config.CVaR_RISK_WEIGHT,
        "cvar_booked_weight":config.CVAR_BOOKED_WEIGHT,
        "cvar_ondemand_weight":config.CVAR_ONDEMAND_WEIGHT,
    }
    if manifest_path.exists() and not args.overwrite:
        if not args.resume:
            ap.error(f"Output {output} exists; pass --resume or specify another --out")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            ap.error("Existing run manifest differs. Use a new --out directory or --overwrite")
    if args.overwrite:
        for path in (manifest_path,training_path,rows_path,summary_path):
            path.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print("OOS protocol:",json.dumps({k:manifest[k] for k in (
        "scope","expected_optimization_passengers","seed_train","seed_test",
        "train_raw","train_retained","test_scenarios","horizon_minutes")},indent=2),flush=True)
    print("Loading independent training scenarios...",flush=True)
    raw=generate_scenarios(expected,profile,args.train_raw,args.seed_train,
                           booking_fraction=args.booking_share)
    reduced=reduce_scenarios(raw,target_size=args.train_retained)
    full_mi=build_model_input(reduced)
    expected_bookings=full_mi["advance_bookings"]
    trained=(json.loads(training_path.read_text(encoding="utf-8"))
             if training_path.exists() and args.resume else {})
    for p in policies:
        if p in trained:
            print("Using previously saved training policy",p,flush=True)
            continue
        print("TRAIN",p,flush=True)
        trained[p]=train_policy(p,full_mi,raw,profile,args)
        training_path.write_text(json.dumps(trained,indent=2),encoding="utf-8")
        print("  x0",trained[p]["first_stage"]["x0"],
              "gap",trained[p]["training_gap"],flush=True)

    print("Generating INDEPENDENT test scenarios...",flush=True)
    test=generate_scenarios(expected,profile,args.test_scenarios,args.seed_test,
                            booking_fraction=args.booking_share)
    if any(s.advance_bookings != expected_bookings for s in test):
        raise AssertionError("OOS bookings differ from training bookings")
    if any(s.id in {x.id for x in reduced} for s in test):
        # IDs may coincide numerically across independent seeds; only seeds matter.
        print("[note] test scenario numerical IDs may overlap training IDs; draws are independent")

    previous=[]
    if args.resume and rows_path.exists():
        with rows_path.open(newline="",encoding="utf-8") as file:
            previous=list(csv.DictReader(file))
    # Successful checks are cached; failed solves are retried on resume.
    done={(r["policy"],int(r["test_index"])) for r in previous
          if r["status"]=="Optimal" and not r.get("error") and
          (args.solver!="highs" or (
              r.get("gap") not in (None, "") and float(r["gap"])<=args.eval_gap+1e-7))}
    mode="a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode,newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        if mode=="w": writer.writeheader()
        for index,scenario in enumerate(test):
            if args.max_test is not None and index>=args.max_test:
                break
            for p in policies:
                if (p,index) in done:
                    continue
                row=eval_policy_one(p,trained[p],scenario,index,full_mi["parameters"],args)
                writer.writerow(row)
                file.flush()
                print(f"TEST {index+1}/{len(test)} {p}: {row['status']} "
                      f"gap={row.get('gap')} "
                      f"loss={row.get('scenario_loss')} "
                      f"net={row.get('net_payoff_after_penalties')} "
                      f"{row.get('error','')}",flush=True)
    # If prior failed case was retried, take last row for each (policy,index).
    rows_by_key={}
    with rows_path.open(newline="",encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows_by_key[(row["policy"],int(row["test_index"]))]=row
    summary=summarize(list(rows_by_key.values()),trained,manifest)
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nPAIRED COMPLETED",summary["n_common_completed"],"/",args.test_scenarios)
    for p,res in summary["policies"].items():
        print(p,json.dumps(res,indent=2))
    print("Saved:",training_path,rows_path,summary_path,sep="\n  ")
    if summary["n_common_completed"] < args.test_scenarios:
        print("WARNING: OOS study is incomplete. Inspect failed cases or continue with --resume")


if __name__=="__main__":
    main()
