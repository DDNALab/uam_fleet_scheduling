"""
Tail-preserving scenario reduction.

Splits raw scenarios by a proxy cost score into central and tail strata and
runs classical backward reduction (Dupacova et al.) separately in each, so
distance alone cannot discard the high-cost tail a high-confidence CVaR
objective depends on. Tail representatives are kept at or above a configured
minimum and near a configured fraction of the target when possible.

Deleting a scenario hands its probability to its nearest surviving
representative in the same stratum, so reduction is mass preserving: each
stratum keeps its original total mass, the weights already sum to one, and the
final normalization is a defensive no-op. Tail mass is never inflated.

The score uses exogenous inputs only -- realized demand, arrival counts and
charging durations -- never the selected initial fleet ``x0``.
"""

import numpy as np

import config
# ============================================================
# 1. Feature vectors
# ============================================================

def build_feature_matrix(scenarios, periods=None, destinations=None, booked=None):
    """
    Standardized feature matrix for reduction.

    Columns, in order: realized passenger demand ``B_td + lambda^O_tds`` for
    every (period, destination), incoming arrival counts ``N^e_ts`` per period,
    and charging duration workload per arrival period. Standardizing removes the
    unit mismatch between passenger counts (tens) and aircraft/duration counts
    (single digits), which makes Euclidean distance meaningful.

    Returns
    -------
    (numpy.ndarray, list of Scenario)
        The ``(|Omega|, n_features)`` matrix and the scenarios with
        ``feature_vector`` attached.
    """
    scenarios = list(scenarios)

    if not scenarios:
        return np.zeros((0, 0)), scenarios

    booked = {} if booked is None else booked

    if destinations is None:
        destinations = sorted(
            {
                d
                for scenario in scenarios
                for (_, d) in scenario.ondemand_demand
            }
            | {d for (_, d) in booked}
        )
    else:
        destinations = list(destinations)

    if periods is None:
        periods = sorted(
            {
                t
                for scenario in scenarios
                for (t, _) in scenario.ondemand_demand
            }
            | {t for (t, _) in booked}
            | {a.arrival_period for scenario in scenarios for a in scenario.aircraft}
        )
    else:
        periods = list(periods)

    period_index = {t: i for i, t in enumerate(periods)}

    rows = []

    for scenario in scenarios:

        demand_features = []

        # Realized demand trajectories, destination by destination.
        for d in destinations:
            for t in periods:
                demand_features.append(
                    float(booked.get((t, d), 0))
                    + float(scenario.ondemand_demand.get((t, d), 0))
                )

        # Incoming arrival counts.
        arrival_features = np.zeros(len(periods), dtype=float)

        if scenario.arrival_counts is not None and len(scenario.arrival_counts) == len(
            periods
        ):
            arrival_features = np.asarray(scenario.arrival_counts, dtype=float)
        else:
            for aircraft in scenario.aircraft:
                position = period_index.get(aircraft.arrival_period)
                if position is not None:
                    arrival_features[position] += 1.0

        # Charging duration workload.
        workload_features = np.zeros(len(periods), dtype=float)

        for aircraft in scenario.aircraft:
            position = period_index.get(aircraft.arrival_period)
            if position is not None:
                workload_features[position] += float(aircraft.duration)

        rows.append(
            np.concatenate(
                [
                    np.asarray(demand_features, dtype=float),
                    arrival_features,
                    workload_features,
                ]
            )
        )

    matrix = np.vstack(rows)

    # Standardize to unitless features; zero spread would divide by zero.
    mean = matrix.mean(axis=0)

    spread = matrix.std(axis=0)

    spread[spread < 1e-9] = 1.0

    normalized = (matrix - mean) / spread

    for scenario, vector in zip(scenarios, normalized):
        scenario.feature_vector = vector

    return normalized, scenarios


def squared_distance_matrix(features):
    """
    Euclidean distance matrix of the standardized features.

    Uses the Gram trick, ``d^2 = |a|^2 + |b|^2 - 2 a.b``, so no Python double
    loop is needed.
    """
    features = np.asarray(features, dtype=float)

    if features.size == 0:
        return np.zeros((0, 0))

    gram = features @ features.T

    diagonal = np.diag(gram)

    squared = diagonal[:, None] + diagonal[None, :] - 2.0 * gram

    np.maximum(squared, 0.0, out=squared)

    return np.sqrt(squared)


# ============================================================
# 2. Proxy score
# ============================================================

def proxy_score(scenario, booked=None, parameters=None, kappa=None):
    """
    Proxy cost score ``psi_s``.

    Rewards realized passenger load and incoming charging workload, and credits
    the stochastic incoming seat supply.
    """
    parameters = config.build_parameters() if parameters is None else parameters

    booked = {} if booked is None else booked

    reduction = parameters.get("reduction", {})

    kappa_p = reduction.get("kappa_p", config.REDUCTION_KAPPA_P) if kappa is None else kappa[0]
    kappa_e = reduction.get("kappa_e", config.REDUCTION_KAPPA_E) if kappa is None else kappa[1]
    kappa_c = reduction.get("kappa_c", config.REDUCTION_KAPPA_C) if kappa is None else kappa[2]

    nmax = parameters.get("evtol_capacity", config.EVTOL_CAPACITY)

    realized_load = sum(
        float(booked.get(key, 0)) + float(value)
        for key, value in scenario.ondemand_demand.items()
    )

    # Booked cells absent from the on-demand dict still count towards load.
    realized_load += sum(
        float(value)
        for key, value in booked.items()
        if key not in scenario.ondemand_demand
    )

    arrival_supply = sum(
        int(value)
        for value in (
            scenario.arrival_counts
            if scenario.arrival_counts is not None
            else np.array([], dtype=int)
        )
    )

    if arrival_supply == 0 and scenario.arrival_counts is None:
        arrival_supply = len(scenario.aircraft)

    charging_workload = sum(float(a.duration) for a in scenario.aircraft)

    return (
        kappa_p * realized_load
        - kappa_e * nmax * arrival_supply
        + kappa_c * charging_workload
    )


def score_scenarios(scenarios, booked=None, parameters=None):
    """Attach ``proxy_score`` to every scenario and return the score array."""
    scores = np.array(
        [proxy_score(s, booked, parameters) for s in scenarios], dtype=float
    )

    for scenario, score in zip(scenarios, scores):
        scenario.proxy_score = float(score)

    return scores


# ============================================================
# 3. Stratification and allocation
# ============================================================

def tail_size(n_scenarios, alpha=None, parameters=None):
    """``ceil((1 - alpha) |Omega_raw|)`` scenarios form the tail stratum."""
    parameters = config.build_parameters() if parameters is None else parameters

    if alpha is None:
        alpha = parameters.get("cvar", {}).get("alpha", config.CVAR_ALPHA)

    return max(1, int(np.ceil((1.0 - float(alpha)) * int(n_scenarios))))


def split_strata(scores, alpha=None, parameters=None):
    """
    Indices of the tail and central strata.

    The tail holds the largest ``ceil((1 - alpha) N)`` proxy scores, i.e. the
    scenarios with the most demand per available seat and the most charging
    work.
    """
    scores = np.asarray(scores, dtype=float)

    n_tail = min(len(scores), tail_size(len(scores), alpha, parameters))

    # Stable ordering keeps ties deterministic for a fixed seed.
    order = np.argsort(-scores, kind="stable")

    tail = np.sort(order[:n_tail])

    central = np.sort(order[n_tail:])

    return tail, central


def allocate_strata(n_available_tail, n_available_central, target_size, parameters=None):
    """
    Split ``target_size`` retained representatives between the two strata.

    Keeps at least ``tail_min`` and roughly ``fraction`` of the representatives
    in the tail, falling back to a single stratum when the request is too small
    to honour that.
    """
    parameters = config.build_parameters() if parameters is None else parameters

    reduction = parameters.get("reduction", {})

    fraction = float(reduction.get("tail_fraction", config.TAIL_STRATUM_FRACTION))
    tail_min = int(reduction.get("tail_min", config.TAIL_STRATUM_MIN))

    target_size = int(target_size)

    # A degenerate stratum cannot supply representatives; the other one takes
    # the whole request.
    if n_available_tail <= 0 and n_available_central <= 0:
        return 0, 0

    if n_available_tail <= 0:
        return 0, min(target_size, n_available_central)

    if n_available_central <= 0:
        return min(target_size, n_available_tail), 0

    if target_size < tail_min + 1:
        # Too few representatives to protect a minimum tail: keep proportional.
        n_tail = max(1, int(round(fraction * target_size)))
    else:
        n_tail = max(tail_min, int(round(fraction * target_size)))

    n_tail = min(n_tail, n_available_tail, max(1, target_size - 1))

    n_central = min(target_size - n_tail, n_available_central)

    # Borrow from the tail when the central stratum cannot supply its share.
    n_tail = min(n_available_tail, target_size - n_central)

    return n_tail, n_central


# ============================================================
# 4. Backward reduction
# ============================================================

def backward_reduce(features, probabilities, target_size):
    """
    Classical backward reduction on a fixed point set.

    At each step the scenario ``i`` minimizing ``pi_i * min_{j != i} d(i, j)`` is
    deleted and its probability is added to its nearest surviving scenario
    (Dupacova, Growe-Kuska and Romisch). Distances between the fixed points do
    not change, so the full matrix is computed once.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n, d)`` standardized features.
    probabilities : numpy.ndarray
        ``(n,)`` scenario probabilities.
    target_size : int
        Number of scenarios to retain.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Original indices of the retained scenarios (ascending) and their
        probabilities.
    """

    features = np.asarray(features, dtype=float)

    probabilities = np.asarray(probabilities, dtype=float).copy()

    n = features.shape[0]

    target_size = int(max(1, min(target_size, n)))

    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)

    distance = squared_distance_matrix(features)

    alive = np.ones(n, dtype=bool)

    for _ in range(n - target_size):

        indices = np.flatnonzero(alive)

        # Distance to the nearest other surviving scenario.
        sub = distance[np.ix_(indices, indices)]

        np.fill_diagonal(sub, np.inf)

        nearest_position = np.argmin(sub, axis=1)

        nearest_distance = sub[np.arange(indices.size), nearest_position]

        cost = probabilities[indices] * nearest_distance

        position = int(np.argmin(cost))

        victim = indices[position]

        target = indices[nearest_position[position]]

        probabilities[target] += probabilities[victim]

        probabilities[victim] = 0.0

        alive[victim] = False

    retained = np.flatnonzero(alive)

    return retained, probabilities[retained]


# ============================================================
# 5. Quality diagnostics
# ============================================================

def effective_scenario_count(probabilities):
    """
    ``N_eff = (sum_s pi_s^2)^-1``.

    A reduced set with one dominant weight collapses towards ``N_eff = 1``;
    report it next to the largest representative weight.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    squared = float(np.sum(probabilities ** 2))

    if squared <= 0:
        return 0.0

    return 1.0 / squared


def reduction_diagnostics(scenarios):
    """Effective count, largest weight and stratum composition."""
    probabilities = np.array([float(s.probability) for s in scenarios], dtype=float)

    if probabilities.size == 0:
        return {
            "retained": 0,
            "effective_scenarios": 0.0,
            "max_probability": 0.0,
            "tail": 0,
            "central": 0,
        }

    return {
        "retained": int(probabilities.size),
        "effective_scenarios": effective_scenario_count(probabilities),
        "max_probability": float(probabilities.max()),
        "tail": int(sum(1 for s in scenarios if s.stratum == "tail")),
        "central": int(sum(1 for s in scenarios if s.stratum == "central")),
    }


# ============================================================
# 6. Main entry point
# ============================================================

def reduce_scenarios(
    scenarios,
    target_size=None,
    parameters=None,
    booked=None,
    alpha=None,
    stratify=True,
):
    """
    Reduce a raw ensemble to ``target_size`` representative scenarios.

    Parameters
    ----------
    scenarios : list of Scenario
        Raw scenarios from :func:`scenario_generator.generate_scenarios`.
    target_size : int, optional
        Defaults to ``parameters["optimization_scenarios"]``.
    parameters : dict, optional
        Parameter block from :func:`config.build_parameters`.
    booked : dict, optional
        ``{(t, d): int}`` known bookings; only used by the proxy score.
    alpha : float, optional
        CVaR level that defines the tail stratum. Defaults to
        ``parameters["cvar"]["alpha"]``.
    stratify : bool, optional
        Set ``False`` for plain unstratified backward reduction.

    Returns
    -------
    list of Scenario
        Representatives in ascending original-id order, with weights summing
        to one and ``stratum`` set.
    """
    scenarios = list(scenarios)

    parameters = config.build_parameters() if parameters is None else parameters

    if target_size is None:
        target_size = parameters.get(
            "optimization_scenarios", config.OPTIMIZATION_SCENARIOS
        )

    target_size = int(target_size)

    if not scenarios:
        return []

    if target_size >= len(scenarios):
        weight = 1.0 / len(scenarios)
        for scenario in scenarios:
            scenario.probability = weight
            scenario.stratum = scenario.stratum or "raw"
        return scenarios

    # Match the feature grid to the scenarios' own period count; reuse the
    # configured horizon only when every arrival_counts has exactly that length.
    configured_periods = parameters.get("periods")

    periods = None

    if configured_periods and all(
        s.arrival_counts is not None
        and len(s.arrival_counts) == int(configured_periods)
        for s in scenarios
    ):
        periods = list(range(1, int(configured_periods) + 1))

    features, scenarios = build_feature_matrix(
        scenarios,
        periods=periods,
        destinations=None,
        booked=booked,
    )

    scores = score_scenarios(scenarios, booked=booked, parameters=parameters)

    if stratify and len(scenarios) > 2:

        tail, central = split_strata(scores, alpha=alpha, parameters=parameters)

        n_tail, n_central = allocate_strata(
            len(tail), len(central), target_size, parameters=parameters
        )

        strata = []

        if n_tail > 0 and len(tail) > 0:
            strata.append((tail, n_tail, "tail"))

        if n_central > 0 and len(central) > 0:
            strata.append((central, n_central, "central"))

        retained_indices = []

        probabilities = np.zeros(len(scenarios), dtype=float)

        # Each stratum starts from raw uniform mass 1/|Omega_raw| per member, so
        # cutting it to a few representatives preserves its total probability.
        raw_weight = 1.0 / len(scenarios)

        for indices, requested, label in strata:

            local_features = features[indices]

            local_weights = np.full(len(indices), raw_weight)

            kept, kept_weights = backward_reduce(
                local_features, local_weights, requested
            )

            for position, weight in zip(kept, kept_weights):
                global_index = int(indices[position])
                retained_indices.append(global_index)
                probabilities[global_index] = float(weight)
                scenarios[global_index].stratum = label

        # Defensive top-up if a stratum was too small to fill its allocation.
        if len(retained_indices) < target_size:

            chosen = set(retained_indices)

            shortfall = target_size - len(retained_indices)

            for index in np.argsort(-scores, kind="stable"):
                if shortfall <= 0:
                    break
                index = int(index)
                if index in chosen:
                    continue
                chosen.add(index)
                retained_indices.append(index)
                probabilities[index] = raw_weight
                scenarios[index].stratum = scenarios[index].stratum or "central"
                shortfall -= 1

        retained_indices = sorted(set(retained_indices))

    else:

        for scenario in scenarios:
            scenario.stratum = "raw"

        local_weights = np.full(len(scenarios), 1.0 / len(scenarios))

        kept, kept_weights = backward_reduce(features, local_weights, target_size)

        retained_indices = [int(index) for index in kept]

        probabilities = np.zeros(len(scenarios), dtype=float)

        for index, weight in zip(retained_indices, kept_weights):
            probabilities[index] = float(weight)

    reduced = []

    for index in retained_indices:
        scenario = scenarios[index]
        scenario.probability = float(probabilities[index])
        reduced.append(scenario)

    # Normalize to one; a no-op for mass-preserving reduction.
    total = sum(s.probability for s in reduced)

    if total > 0:
        for scenario in reduced:
            scenario.probability /= total

    return reduced


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from demand_model import build_baseline_demand
    from scenario_generator import generate_scenarios, scenario_diagnostics

    parameters = config.build_parameters()

    baseline = build_baseline_demand()

    raw = generate_scenarios(baseline, n_scenarios=100, parameters=parameters)

    print("raw:", scenario_diagnostics(raw))

    reduced = reduce_scenarios(raw, target_size=10, parameters=parameters)

    print(
        "reduced topology:",
        [(s.id, round(s.probability, 3), s.stratum) for s in reduced],
    )

    print("reduced:", reduction_diagnostics(reduced))
