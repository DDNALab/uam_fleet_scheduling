"""Explicit, auditable interpretation of the TOTAL_PASSENGER_TRIPS target.

No geospatial files are read here: pass the DataFrame returned by build_uam_data().
This helper is shared by demand_audit, scale_instances and oos_evaluate.
"""

from __future__ import annotations
import math

SCOPES = ("active-geographic", "active-renormalized", "all-geographic")


def select_expected_demand(od_table, *, scope, passengers, active_destinations):
    """Return destination->expected daily passengers from a full geographic OD table.

    active-geographic: keep the geographic shares allocated to active destinations;
        total after filtering is generally LESS than the requested full-network target.
    active-renormalized: distribute the requested total entirely across active
        destinations, preserving their relative geographic shares.
    all-geographic: include ALL modeled geographic destinations, with requested
        total distributed by the original gravity-model proportions.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; select from {SCOPES}")
    total = float(passengers)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("passengers must be a positive, finite total")
    if od_table.empty or "destination" not in od_table or "expected_trips" not in od_table:
        raise ValueError("OD table must have destination and expected_trips columns")

    all_shares = {
        str(row["destination"]): float(row["expected_trips"])
        for _, row in od_table.iterrows()
    }
    if not all_shares or any(v < 0 or not math.isfinite(v) for v in all_shares.values()):
        raise ValueError("OD shares must be finite and nonnegative")
    geographic_total = sum(all_shares.values())
    if geographic_total <= 0:
        raise ValueError("All geographic OD shares are zero")

    active = {d: all_shares[d] for d in active_destinations if d in all_shares}
    if scope != "all-geographic" and not active:
        raise ValueError("Configured ACTIVE_DESTINATIONS have zero overlap with OD table")
    if scope == "all-geographic":
        return {d: total * v / geographic_total for d, v in all_shares.items()}
    if scope == "active-geographic":
        return {d: total * v / geographic_total for d, v in active.items()}
    active_total = sum(active.values())
    if active_total <= 0:
        raise ValueError("No positive gravity mass among active destinations")
    return {d: total * v / active_total for d, v in active.items()}


def expected_demand_from_geodata(*, scope, passengers):
    """Read configured Census and ACS files through the existing spatial pipeline."""
    from src import config
    from src.uam_data_pipeline import build_uam_data
    data = build_uam_data()
    expected = select_expected_demand(
        data["od_demand"], scope=scope, passengers=passengers,
        active_destinations=config.ACTIVE_DESTINATIONS,
    )
    profile = data["time_profile"]["weight"].to_numpy()
    if not math.isclose(float(profile.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError("Demand time-profile weights do not sum to one")
    return expected, profile


def time_profile_for_horizon(horizon_minutes, *, start_minutes=None):
    """Build a case-specific, properly normalized 15-minute demand profile."""
    import numpy as np
    from src import config
    from src.uam_data_pipeline import passenger_time_profile

    if start_minutes is None:
        start_minutes = config.HORIZON_START
    hours = passenger_time_profile(
        start_minutes=start_minutes, end_minutes=start_minutes + int(horizon_minutes),
        bin_minutes=config.BIN_SIZE,
    )
    weights = hours["weight"].to_numpy(dtype=float)
    expected_bins = int(horizon_minutes) // config.BIN_SIZE
    if (horizon_minutes <= 0 or horizon_minutes % config.BIN_SIZE
            or len(weights) != expected_bins
            or not np.isclose(weights.sum(), 1.0, atol=1e-12)):
        raise ValueError("invalid case-specific passenger time profile")
    return weights


def booking_capacity_preflight(bookings, *, destinations, n_periods,
                               los_periods, seats_per_flight,
                               max_departures_per_period, time_limit=30):
    """Check EXACT first-stage nominal booking feasibility using a small MILP.

    Enforces the two booking-protection families and the shared per-period
    planned-departure limit of stochastic_model.py. Planned departures remain
    integer and booking flows are continuous (a TU network for fixed departures).

    This is a necessary check for the COMPLETE stochastic model, not proof that
    aircraft arrivals, charging, and other scenario constraints are feasible.
    No geography files or PuLP installations are needed for this preflight.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    if any(v < 0 or int(v) != v for v in bookings.values()):
        raise ValueError("advance bookings must be nonnegative integers")
    if min(n_periods, seats_per_flight, max_departures_per_period) <= 0:
        raise ValueError("periods, seats, and departure capacity must be positive")
    if los_periods < 0:
        raise ValueError("los_periods must be nonnegative")
    dest_set = set(destinations)
    if any(d not in dest_set or r < 1 or r > n_periods for (r, d) in bookings):
        raise ValueError("booking lies outside destinations or the case horizon")
    demand = {(r, d): int(value) for (r, d), value in bookings.items() if value > 0}
    total_booked = sum(demand.values())
    upper_aggregate = seats_per_flight * max_departures_per_period * n_periods
    if total_booked > upper_aggregate:
        return {"status": "Infeasible", "nominal_flights_certificate": None,
                "message": f"{total_booked} bookings exceed {upper_aggregate} aggregate nominal seats"}
    if not demand:
        return {"status": "Feasible", "nominal_flights_certificate": 0,
                "message": "No advance bookings"}

    # A cheap EXACT integer certificate: assign each booking group to its
    # own requested service period. If all departure limits are respected,
    # there is no reason to solve a redundant, highly degenerate MILP.
    direct_flights = {
        (r, d): (value + seats_per_flight - 1) // seats_per_flight
        for (r, d), value in demand.items()
    }
    peak_flights = max(
        sum(v for (r, _d), v in direct_flights.items() if r == period)
        for period in range(1, n_periods + 1)
    )
    if peak_flights <= max_departures_per_period:
        return {
            "status": "Feasible",
            "nominal_flights_certificate": sum(direct_flights.values()),
            "message": "Exact constructive certificate: protect bookings in their "
                       "own service periods (not necessarily minimum flights)",
        }

    active_dests = sorted({d for _, d in demand})
    n_keys = [(d, k) for d in active_dests for k in range(1, n_periods + 1)]
    q_keys = [
        (r, d, k) for r, d in sorted(demand)
        for k in range(r, min(n_periods, r + los_periods) + 1)
    ]
    n_id = {key: i for i, key in enumerate(n_keys)}
    q_id = {key: len(n_keys) + i for i, key in enumerate(q_keys)}
    count = len(n_keys) + len(q_keys)
    # Row blocks: exact demand by (r,d); seat capacity by (d,k);
    # shared maximum planned departures by k.
    rows = len(demand) + len(n_keys) + n_periods
    A = lil_matrix((rows, count), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.zeros(rows)
    seat_rows = {key: len(demand) + i for i, key in enumerate(n_keys)}
    for idx, ((r, d), requested) in enumerate(sorted(demand.items())):
        lb[idx] = ub[idx] = requested
        for k in range(r, min(n_periods, r + los_periods) + 1):
            col = q_id[r, d, k]
            A[idx, col] = 1
            A[seat_rows[d, k], col] = 1
    for (d, k), col in n_id.items():
        A[seat_rows[d, k], col] = -seats_per_flight
        A[len(demand) + len(n_keys) + k - 1, col] = 1
    ub[len(demand) + len(n_keys):] = max_departures_per_period

    objective = np.zeros(count)
    objective[:len(n_keys)] = 1.0
    integrality = np.zeros(count, dtype=np.int32)
    integrality[:len(n_keys)] = 1
    upper = np.full(count, np.inf)
    upper[:len(n_keys)] = max_departures_per_period
    for key, col in q_id.items():
        upper[col] = demand[key[:2]]
    # The LP relaxation provides a cheap *necessary* feasibility check.
    # If even fractional departures cannot protect bookings, the booking
    # subsystem is rigorously infeasible; no branch-and-bound is needed.
    from scipy.optimize import linprog
    eq_count = len(demand)
    A_csr = A.tocsr()
    lp = linprog(
        c=np.zeros(count), A_eq=A_csr[:eq_count], b_eq=ub[:eq_count],
        A_ub=A_csr[eq_count:], b_ub=ub[eq_count:],
        bounds=list(zip(np.zeros(count), upper)), method="highs",
    )
    if lp.status == 2:
        return {"status": "Infeasible", "nominal_flights_certificate": None,
                "message": "Booking-capacity LP relaxation is infeasible"}
    if lp.status != 0:
        return {"status": "Unknown", "nominal_flights_certificate": None,
                "message": "Booking-capacity LP preflight: " + str(lp.message)}

    result = milp(
        c=objective, integrality=integrality,
        bounds=Bounds(np.zeros(count), upper),
        constraints=LinearConstraint(A_csr, lb, ub),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.001},
    )
    # A valid incumbent proves existence even if the preflight times out;
    # absence of a solution on time limit does not prove infeasibility.
    if result.x is not None:
        residual = A_csr @ result.x
        feasible = (
            np.all(residual >= lb - 1e-5) and np.all(residual <= ub + 1e-5)
            and np.max(np.abs(result.x[:len(n_keys)] - np.rint(
                result.x[:len(n_keys)]))) < 1e-5
        )
        if feasible:
            return {
                "status": "Feasible" if result.status == 0 else "FeasibleTimed",
                "nominal_flights_certificate": int(round(sum(result.x[:len(n_keys)]))),
                "message": str(result.message),
            }
    return {
        "status": "Infeasible" if result.status == 2 else "Unknown",
        "nominal_flights_certificate": None,
        "message": str(result.message),
    }
