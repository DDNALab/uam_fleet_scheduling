"""
Structural first-stage feasibility screening.

Infeasibility is a property of the *booking pattern*, not of the operating
policy: accepted bookings are exogenous and the capacity and protection
constraints are hard. :func:`screen_booking_share` runs a cheap necessary data
cut followed by an exact zero-objective MILP. Neither test says anything about
realized service -- a structurally feasible pattern can still leave booked
passengers unserved when scenario aircraft shortages bite.
"""

import pulp

from stochastic_model import admissible_departures, name


# ============================================================
# Necessary data cut
# ============================================================

def necessary_data_cut(booked_demand, destinations, periods, parameters):
    """
    Check the necessary data cut on the booked demand.

    Over every interval of request periods, the departures needed to protect the
    bookings must fit in the takeoff slots offered from the interval start to
    its latest service window. Necessary but not sufficient: destination-specific
    departures still compete for the same period-by-period slots.

    Returns
    -------
    (bool, list of dict)
        ``ok`` is True when no interval violates the cut. Each violation is
        reported as ``{"start", "end", "required", "available"}`` where
        ``required`` is the number of departures the bookings demand and
        ``available`` is the number of takeoff slots the hub can offer inside
        the corresponding service window.
    """
    capacity = int(parameters["evtol_capacity"])
    max_departures = int(parameters["max_departures"])
    los = int(parameters["los_periods"])

    horizon = len(periods)

    periods = [int(t) for t in periods]

    # Booked demand restricted to the modeled grid.
    booked = {
        (int(r), d): int(v)
        for (r, d), v in booked_demand.items()
        if int(v) > 0 and int(r) in set(periods) and d in destinations
    }

    violations = []

    for a in periods:

        for b in periods:

            if b < a:
                continue

            # Demand in request periods a..b, per destination.
            demand_by_destination = {
                d: sum(booked.get((r, d), 0) for r in periods if a <= r <= b)
                for d in destinations
            }

            if not any(demand_by_destination.values()):
                continue

            # Departures needed to protect those bookings, given integer flights.
            required = sum(
                -(-count // capacity) for count in demand_by_destination.values()
            )

            window_end = min(horizon, b + los)

            available = max_departures * len(
                [k for k in periods if a <= k <= window_end]
            )

            if required > available:

                violations.append(
                    {
                        "start": a,
                        "end": b,
                        "required": required,
                        "available": available,
                    }
                )

    return not violations, violations


# ============================================================
# Exact first-stage feasibility MILP
# ============================================================

def _feasibility_solver(time_limit, verbose):
    """
    Pick an available solver for the small feasibility MILP.

    This check runs before the main solve and must never be the reason a run
    fails, so it walks the same preference order as the main solver and returns
    the first one that is usable.
    """
    candidates = (
        ("HIGHS", lambda: pulp.HiGHS(msg=verbose, timeLimit=time_limit)),
        ("CBC", lambda: pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)),
    )

    for label, factory in candidates:

        try:

            solver = factory()

            if solver.available():
                return solver

        except Exception:  # pragma: no cover - solver availability is optional
            continue

    raise RuntimeError(
        "no MILP solver available for the first-stage feasibility check "
        "(tried HiGHS and CBC)"
    )


def exact_first_stage_feasibility(
    booked_demand, destinations, periods, parameters, time_limit=60, verbose=False
):
    """
    Decide first-stage feasibility exactly.

    Solves a zero-objective MILP over integer ``n_dk >= 0`` and continuous
    ``q^B_rdk >= 0``, subject to the departure-capacity, protection and seat
    constraints. Returns ``(feasible, information)``.

    A feasible point is an admissible first-stage departure plan. Because the
    protection system is a capacitated bipartite transportation problem with a
    totally unimodular incidence matrix, an integer protection flow can always be
    recovered even though ``q^B`` is modeled as continuous.
    """
    capacity = int(parameters["evtol_capacity"])
    max_departures = int(parameters["max_departures"])
    los = int(parameters["los_periods"])

    horizon = len(periods)

    periods = [int(t) for t in periods]

    booked = {
        (int(r), d): int(v)
        for (r, d), v in booked_demand.items()
        if int(v) > 0 and int(r) in set(periods) and d in destinations
    }

    if not booked:
        return True, {"reason": "no booked demand", "plan": {}}

    windows = {r: admissible_departures(r, horizon, los) for r in periods}

    model = pulp.LpProblem(name("FirstStage_Feasibility"), pulp.LpMinimize)

    n = {
        (destination, t): pulp.LpVariable(
            name("Plan", destinations.index(destination), t),
            lowBound=0,
            cat="Integer",
        )
        for destination in destinations
        for t in periods
    }

    q = {}

    for (r, destination), count in booked.items():

        for k in windows.get(r, []):

            q[r, destination, k] = pulp.LpVariable(
                name("Protect", r, destinations.index(destination), k),
                lowBound=0,
                cat="Continuous",
            )

    # At most ``max_departures`` takeoffs per period.
    for t in periods:
        model += (
            pulp.lpSum(n[destination, t] for destination in destinations)
            <= max_departures
        ), name("Capacity", t)

    # Every booked group must be fully protected.
    for (r, destination), count in booked.items():

        terms = [
            q[r, destination, k]
            for k in windows.get(r, [])
            if (r, destination, k) in q
        ]

        if not terms:
            # No admissible departure period inside the horizon: structurally
            # infeasible for this group.
            return False, {
                "reason": "request period %d for %s has no admissible departure "
                          "inside the %d-period horizon" % (r, destination, horizon),
                "plan": {},
            }

        model += (
            pulp.lpSum(terms) == count
        ), name("Protection", r, destinations.index(destination))

    # Protected passengers must fit the planned flights' seat capacity.
    for destination in destinations:

        for k in periods:

            terms = [
                q[r, destination, k]
                for r in periods
                if k in windows.get(r, []) and (r, destination, k) in q
            ]

            if not terms:
                continue

            model += (
                pulp.lpSum(terms) <= capacity * n[destination, k]
            ), name("Seats", destinations.index(destination), k)

    model += pulp.lpSum(0.0 * n[key] for key in n)

    status = model.solve(_feasibility_solver(time_limit, verbose))

    feasible = pulp.LpStatus[status] == "Optimal"

    plan = {
        (destination, t): int(round(n[destination, t].value() or 0.0))
        for destination in destinations
        for t in periods
        if (n[destination, t].value() or 0.0) > 0.5
    }

    return feasible, {
        "reason": pulp.LpStatus[status],
        "plan": plan,
        "departures": int(sum(plan.values())),
    }


# ============================================================
# Combined screen
# ============================================================

def screen_booking_share(
    booked_demand,
    destinations,
    periods,
    parameters,
    run_exact=True,
    verbose=False,
):
    """
    Apply both feasibility tests to one booking pattern.

    Returns
    -------
    dict
        ``cut_ok``, ``cut_violations``, ``exact_ok``, ``exact_detail`` and an
        overall ``structurally_feasible`` flag.
    """
    cut_ok, violations = necessary_data_cut(
        booked_demand, destinations, periods, parameters
    )

    if not cut_ok:

        return {
            "cut_ok": False,
            "cut_violations": violations,
            "exact_ok": None,
            "exact_detail": {"reason": "skipped: Eq. (10) data cut failed"},
            "structurally_feasible": False,
        }

    if not run_exact:

        return {
            "cut_ok": True,
            "cut_violations": [],
            "exact_ok": None,
            "exact_detail": {"reason": "exact test not requested"},
            "structurally_feasible": None,
        }

    exact_ok, detail = exact_first_stage_feasibility(
        booked_demand, destinations, periods, parameters, verbose=verbose
    )

    return {
        "cut_ok": True,
        "cut_violations": [],
        "exact_ok": exact_ok,
        "exact_detail": detail,
        "structurally_feasible": bool(exact_ok),
    }


def largest_feasible_booking_share(
    baseline_builder,
    parameters,
    shares=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    **baseline_kwargs
):
    """
    Report the largest tested ``phi_B`` whose booking pattern is structurally
    feasible.

    This is structural feasibility only; it says nothing about realized booking
    service. ``baseline_builder`` is called as
    ``baseline_builder(booking_share=phi, **baseline_kwargs)`` and must return a
    baseline block with ``booked``, ``destinations`` and ``periods``.
    """
    results = []

    best = None

    for share in shares:

        baseline = baseline_builder(booking_share=share, **baseline_kwargs)

        screen = screen_booking_share(
            baseline["booked"],
            baseline["destinations"],
            baseline["periods"],
            parameters,
            run_exact=True,
        )

        results.append(
            {
                "booking_share": float(share),
                "booked_total": int(sum(baseline["booked"].values())),
                **screen,
            }
        )

        if screen["structurally_feasible"]:
            best = float(share)

    return {"largest_feasible": best, "results": results}
