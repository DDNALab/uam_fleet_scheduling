"""Audit whether passenger target refers to ALL destinations or ACTIVE ones.

Run from project root (directory containing src):
  python -m src.diagnostics.demand_audit --passengers 20 --scope active-renormalized
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src import config
from src.diagnostics.demand_design import SCOPES, expected_demand_from_geodata
from src.scenario_generator import generate_advance_bookings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--passengers", type=float, default=config.TOTAL_PASSENGER_TRIPS)
    p.add_argument("--scope", choices=SCOPES, default="active-geographic")
    p.add_argument("--booking-share", type=float, default=config.ADVANCE_BOOKING_FRACTION)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not 0 <= args.booking_share <= 1:
        p.error("--booking-share must be in [0,1]")
    expected, profile = expected_demand_from_geodata(scope=args.scope, passengers=args.passengers)
    bookings = generate_advance_bookings(expected, profile, booking_fraction=args.booking_share)
    result = {
        "demand_scope": args.scope,
        "requested_passengers": args.passengers,
        "expected_optimization_passengers": sum(expected.values()),
        "configured_aircraft_passenger_ratio": config.AIRCRAFT_PASSENGER_RATIO,
        "approximate_expected_incoming_aircraft": sum(expected.values()) * config.AIRCRAFT_PASSENGER_RATIO,
        "advance_booking_fraction": args.booking_share,
        "fixed_advance_bookings": sum(bookings.values()),
        "destinations": expected,
        "horizon_minutes": config.HORIZON_MINUTES,
        "period_minutes": config.BIN_SIZE,
        "los_minutes": config.LOS_MINUTES,
        "census_shapefile": str(config.SHAPEFILE_PATH),
        "acs_json": str(config.ACS_JSON_PATH),
    }
    print(json.dumps(result, indent=2))
    out = args.out or config.DIAGNOSTICS_DIR / f"demand_audit_{args.scope}_{int(args.passengers)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
