"""Run fixed-capacity or demand-intensity-scaled projected UAM MILP experiments.

Canonical proportional study (from project root, PowerShell):
  python -m src.diagnostics.scale_instances `
      --profile proportional --scope all-geographic `
      --cases 200:20:10 500:50:15 1000:100:20 2000:200:30 `
      --seeds 60 64 67 --solver highs --gap 0.05 --time-limit 3600

An inexpensive booking-protection check BEFORE scenario generation:
  python -m src.diagnostics.scale_instances --profile proportional `
      --scope all-geographic --cases 2000:200:30 --seeds 60 `
      --preflight-only

Each CASE is target_passengers:raw_scenarios:retained_scenarios.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import pulp

from src import config
from src.diagnostics.demand_design import (
    SCOPES, booking_capacity_preflight, select_expected_demand,
    time_profile_for_horizon,
)
from src.uam_data_pipeline import build_uam_data
from src.scenario_generator import generate_advance_bookings, generate_scenarios
from src.scenario_reduction import reduce_scenarios
from src.model_builder import build_model_input
from src.stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model
from src.diagnostics.fleet_sensitivity import check_rows_and_integrality, check_inventory


FIELDS = [
    "profile", "scope", "seed", "requested_passengers", "expected_model_passengers",
    "passengers_per_hour", "bookings", "booking_share", "raw_scenarios",
    "retained_scenarios", "effective_scenarios", "max_scenario_probability",
    "horizon_start", "horizon_end", "horizon_minutes", "n_periods", "destinations",
    "initial_fleet_cap", "initial_fleet_cost", "charger_capacity", "departure_limit",
    "capacity_multiplier", "aircraft_passenger_ratio", "raw_incoming_mean",
    "raw_incoming_min", "raw_incoming_max", "booking_precheck",
    "booking_precheck_nominal_flights", "booking_precheck_seconds", "booking_precheck_message",
    "variables", "integer_variables", "constraints", "generation_seconds",
    "reduction_seconds", "build_seconds", "solve_seconds", "status", "gap", "objective",
    "initial_fleet_allocated", "max_constraint_violation", "max_integrality_error",
    "max_bound_violation", "inventory_passed", "error",
]


def parse_case(text):
    try:
        p, raw, retained = text.split(":")
        p, raw, retained = float(p), int(raw), int(retained)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Use TOTAL:RAW:RETAINED, e.g. 200:20:10")
    if p <= 0 or not math.isfinite(p) or raw < 2 or not (2 <= retained <= raw):
        raise argparse.ArgumentTypeError("Require finite total>0 and 2<=retained<=raw")
    return p, raw, retained


def _validate_overrides(ap, args):
    for option in ("horizons", "fleet_caps", "charger_caps", "departure_caps"):
        supplied = getattr(args, option)
        if supplied is not None and len(supplied) != len(args.cases):
            ap.error(f"--{option.replace('_', '-')} requires one number per --cases entry")
    if not 0 <= args.booking_share <= 1:
        ap.error("--booking-share must be in [0,1]")
    if args.aircraft_ratio < 0 or not math.isfinite(args.aircraft_ratio):
        ap.error("--aircraft-ratio must be finite and nonnegative")
    if args.capacity_factor <= 0 or not math.isfinite(args.capacity_factor):
        ap.error("--capacity-factor must be finite and positive")
    if args.preflight_only and args.skip_booking_preflight:
        ap.error("--preflight-only cannot be combined with --skip-booking-preflight")


def _case_settings(total, index, args, actual_expected):
    hours = None if args.horizons is None else args.horizons[index]
    # Scale to the actual modeled OD total, not to a full-network target that
    # active-geographic scope may intentionally filter down.
    s = config.scalability_settings(
        actual_expected, horizon_hours=hours, mode=args.profile,
        capacity_factor=args.capacity_factor,
    )
    for field, option in (("initial_fleet_max", "fleet_caps"),
                          ("charging_facilities", "charger_caps"),
                          ("max_departures", "departure_caps")):
        supplied = getattr(args, option)
        if supplied is not None:
            s[field] = supplied[index]
    return s


def apply_case_settings(model_input, settings):
    """Apply per-instance settings after model_builder's legacy constant imports.

    We deliberately override model-input PARAMETERS, not global config constants:
    the source model_builder imports config values with `from .config import`.
    Mutating config later would therefore NOT change its imported constants.
    All projected-model capacity and time constraints read these parameters.
    """
    params = model_input["parameters"]
    params.update({
        "periods": settings["n_periods"],
        "initial_fleet_max": settings["initial_fleet_max"],
        "charging_facilities": settings["charging_facilities"],
        "max_departures": settings["max_departures"],
    })
    params["horizon"] = {
        "start": config.HORIZON_START,
        "end": config.HORIZON_START + settings["horizon_minutes"],
        "bin_size": config.BIN_SIZE,
    }
    last = settings["n_periods"]
    if any(not 1 <= int(a["arrival_period"]) <= last
           for s in model_input["scenarios"].values() for a in s["aircraft"].values()):
        raise ValueError("An incoming aircraft falls outside the case-specific horizon")
    if any(not 1 <= int(r) <= last for (r, _) in model_input["advance_bookings"]):
        raise ValueError("Advance bookings fall outside the case-specific horizon")
    if any(not 1 <= int(r) <= last
           for s in model_input["scenarios"].values()
           for (r, _) in s["on_demand_demand"]):
        raise ValueError("On-demand requests fall outside the case-specific horizon")
    return model_input


def _args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=SCOPES, default="active-renormalized")
    ap.add_argument("--profile", choices=["fixed", "proportional"], default="fixed",
                    help="fixed reproduces the original four-hour defaults; proportional "
                         "uses 4/6/8/12 hours for 200/500/1000/2000 passengers")
    ap.add_argument("--cases", nargs="+", type=parse_case,
                    default=[(20., 30, 3), (40., 60, 5), (80., 100, 10)])
    ap.add_argument("--seeds", nargs="+", type=int, default=[60, 61])
    ap.add_argument("--booking-share", type=float, default=config.ADVANCE_BOOKING_FRACTION)
    ap.add_argument("--aircraft-ratio", type=float, default=config.AIRCRAFT_PASSENGER_RATIO,
                    help="Expected incoming aircraft per expected passenger (already "
                         "scales with TOTAL demand; do not multiply twice)")
    ap.add_argument("--capacity-factor", type=float, default=1.0,
                    help="Additional multiplicative capacity sensitivity factor")
    ap.add_argument("--horizons", nargs="+", type=float,
                    help="Optional horizon hours, one per --cases entry")
    ap.add_argument("--fleet-caps", nargs="+", type=int,
                    help="Optional initial ready-fleet caps, one per case")
    ap.add_argument("--charger-caps", nargs="+", type=int,
                    help="Optional simultaneous charging-unit caps, one per case")
    ap.add_argument("--departure-caps", nargs="+", type=int,
                    help="Optional departures per 15 minutes, one per case")
    ap.add_argument("--booking-preflight-time-limit", type=float, default=30.)
    ap.add_argument("--skip-booking-preflight", action="store_true")
    ap.add_argument("--preflight-only", action="store_true",
                    help="Check nominal booking capacity, without generating scenarios")
    ap.add_argument("--solver", choices=["highs", "gurobi", "cbc", "auto"], default="highs")
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--no-solve", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    _validate_overrides(ap, args)
    for option in ("fleet_caps", "charger_caps", "departure_caps"):
        supplied = getattr(args, option)
        if supplied is not None and any(v <= 0 for v in supplied):
            ap.error(f"--{option.replace('_', '-')} must be positive")
    if args.booking_preflight_time_limit <= 0:
        ap.error("--booking-preflight-time-limit must be positive")
    args.out = args.out or (config.DIAGNOSTICS_DIR /
                            f"scale_instances_{args.scope}_{args.profile}.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = _args()
    print("Reading actual geographic inputs once:",
          config.SHAPEFILE_PATH, config.ACS_JSON_PATH, flush=True)
    data = build_uam_data()

    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for index, (total, raw, retained) in enumerate(args.cases):
            expected = select_expected_demand(
                data["od_demand"], scope=args.scope, passengers=total,
                active_destinations=config.ACTIVE_DESTINATIONS,
            )
            actual = sum(expected.values())
            # For the canonical proportional study, exactly 9 destinations
            # come from --scope all-geographic. We never alter OD shares.
            settings = _case_settings(total, index, args, actual)
            time_profile = time_profile_for_horizon(settings["horizon_minutes"])
            bookings = generate_advance_bookings(expected, time_profile,
                                                  booking_fraction=args.booking_share)
            preflight, preflight_seconds = {"status": "NotRun"}, 0.0
            if not args.skip_booking_preflight:
                start = time.perf_counter()
                preflight = booking_capacity_preflight(
                    bookings, destinations=expected, n_periods=settings["n_periods"],
                    los_periods=config.LOS_PERIODS,
                    seats_per_flight=config.EVTOL_CAPACITY,
                    max_departures_per_period=settings["max_departures"],
                    time_limit=args.booking_preflight_time_limit,
                )
                preflight_seconds = time.perf_counter() - start
            print(
                f"\nCASE requested={total:g} modeled={actual:.2f} "
                f"periods={settings['n_periods']} "
                f"hours={settings['horizon_minutes']/60:g} "
                f"bookings={sum(bookings.values())} "
                f"initial_cap={settings['initial_fleet_max']} "
                f"chargers={settings['charging_facilities']} "
                f"departures/period={settings['max_departures']} "
                f"ratio={args.aircraft_ratio:g} "
                f"booking_preflight={preflight['status']}", flush=True,
            )
            for seed in args.seeds:
                row = {
                    "profile": args.profile, "scope": args.scope, "seed": seed,
                    "requested_passengers": total, "expected_model_passengers": actual,
                    "passengers_per_hour": settings["passengers_per_hour"],
                    "bookings": sum(bookings.values()), "booking_share": args.booking_share,
                    "raw_scenarios": raw, "retained_scenarios": retained,
                    "horizon_start": config.HORIZON_START,
                    "horizon_end": config.HORIZON_START + settings["horizon_minutes"],
                    "horizon_minutes": settings["horizon_minutes"],
                    "n_periods": settings["n_periods"], "destinations": len(expected),
                    "initial_fleet_cap": settings["initial_fleet_max"],
                    "initial_fleet_cost": config.INITIAL_FLEET_COST,
                    "charger_capacity": settings["charging_facilities"],
                    "departure_limit": settings["max_departures"],
                    "capacity_multiplier": settings["capacity_multiplier"],
                    "aircraft_passenger_ratio": args.aircraft_ratio,
                    "booking_precheck": preflight["status"],
                    "booking_precheck_nominal_flights": preflight.get("nominal_flights_certificate"),
                    "booking_precheck_seconds": preflight_seconds,
                    "booking_precheck_message": preflight.get("message", ""),
                }
                if args.preflight_only:
                    row["status"] = "PreflightOnly"
                    print(f"  seed={seed}: booking preflight only", flush=True)
                    writer.writerow(row)
                    file.flush()
                    continue
                if preflight["status"] == "Infeasible":
                    row.update(status="BookingInfeasible", inventory_passed=False,
                               error="Exact nominal booking-capacity check infeasible; "
                                     "full stochastic MILP not built")
                    print(f"  seed={seed}: booking protection INFEASIBLE", flush=True)
                    writer.writerow(row)
                    file.flush()
                    continue
                if preflight["status"] == "Unknown":
                    print("  WARNING: booking preflight inconclusive; "
                          "continuing full solve", flush=True)
                try:
                    start = time.perf_counter()
                    scenarios = generate_scenarios(
                        expected, time_profile, raw, seed,
                        booking_fraction=args.booking_share,
                        aircraft_passenger_ratio=args.aircraft_ratio,
                        horizon_start=config.HORIZON_START,
                    )
                    row["generation_seconds"] = time.perf_counter() - start
                    actual_bookings = scenarios[0].advance_bookings
                    if actual_bookings != bookings:
                        raise AssertionError("Preflight and generated bookings do not agree")
                    arrivals = [len(s.aircraft) for s in scenarios]
                    row["raw_incoming_mean"] = sum(arrivals) / len(arrivals)
                    row["raw_incoming_min"] = min(arrivals)
                    row["raw_incoming_max"] = max(arrivals)
                    start = time.perf_counter()
                    reduced = reduce_scenarios(scenarios, target_size=retained)
                    row["reduction_seconds"] = time.perf_counter() - start
                    weights = [s.probability for s in reduced]
                    row["effective_scenarios"] = 1. / sum(w*w for w in weights)
                    row["max_scenario_probability"] = max(weights)
                    start = time.perf_counter()
                    mi = apply_case_settings(build_model_input(reduced), settings)
                    if mi["parameters"]["periods"] != len(time_profile):
                        raise AssertionError("Case horizon did not reach model builder")
                    md = build_stochastic_model(mi)
                    model = add_constraints_and_objective(md)
                    row["build_seconds"] = time.perf_counter() - start
                    row["variables"] = len(model.variables())
                    row["integer_variables"] = sum(
                        v.cat == pulp.LpInteger for v in model.variables())
                    row["constraints"] = len(model.constraints)
                    if args.no_solve:
                        row["status"] = "BuiltOnly"
                    else:
                        start = time.perf_counter()
                        res = solve_model(
                            model, solver=args.solver, gap_rel=args.gap,
                            time_limit=args.time_limit, verbose=args.verbose,
                        )
                        row["solve_seconds"] = time.perf_counter() - start
                        row["status"] = res["status"]
                        row["gap"] = res.get("solver_reported_gap")
                        row["objective"] = res.get("objective")
                        certified = (res["status"] == "Optimal" and
                                     (row["gap"] is None or
                                      row["gap"] <= args.gap + 1e-7))
                        if certified:
                            row["initial_fleet_allocated"] = round(
                                pulp.value(md["variables"]["initial_fleet"]))
                            row.update(check_rows_and_integrality(model))
                            check_inventory(md)
                            row["inventory_passed"] = True
                        else:
                            row["inventory_passed"] = False
                            row["error"] = (
                                "No certified target-gap solution; do not interpret "
                                "decision metrics")
                    print("  ", row["status"], "seed", seed,
                          "vars", row["variables"],
                          "integer", row["integer_variables"],
                          "constraints", row["constraints"],
                          "objective", row.get("objective"),
                          "gap", row.get("gap"), flush=True)
                except Exception as exc:
                    row["status"] = "ERROR"
                    row["error"] = repr(exc)
                    print("  ERROR", repr(exc), flush=True)
                writer.writerow(row)
                file.flush()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
