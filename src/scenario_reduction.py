"""Tail-aware backward reduction of hybrid UAM operating scenarios.

The initial ready fleet is a first-stage decision, NOT a random scenario input.
All scenarios share the same advance bookings.  Distances use standardized
trajectories of realized demand, incoming-aircraft arrivals, and the charging
workload of incoming aircraft.  Highest-stress raw scenarios form a separate
tail stratum, reducing the chance that distance-based reduction discards them.

The public interface used by experiment_runner.py is:
    reduced = reduce_scenarios(raw_scenarios, target_size=OPTIMIZATION_SCENARIOS)

This script has no project-specific file paths and works from the project root
when imported as src.scenario_reduction.
"""

from __future__ import annotations

from copy import copy
import math

import numpy as np

from .config import (
    BATTERY_CAPACITY,
    BIN_SIZE,
    CHARGING_RATE,
    CVaR_ALPHA,
    EVTOL_CAPACITY,
    HORIZON_START,
    SOC_REQUIRED,
    STRESS_WEIGHT_AIRCRAFT,
    STRESS_WEIGHT_CHARGING,
    STRESS_WEIGHT_PASSENGER,
    TARGET_SCENARIOS,
)


def charging_periods_required(soc: float) -> int:
    """Charging periods for one incoming aircraft; consistent with p_js."""
    if soc >= SOC_REQUIRED:
        return 0
    energy_kwh = BATTERY_CAPACITY * (SOC_REQUIRED - float(soc)) / 100.0
    return int(math.ceil(energy_kwh * 60.0 / (CHARGING_RATE * BIN_SIZE)))


def calculate_stress_score(scenario) -> float:
    """Paper's exogenous proxy: passenger load - incoming seats + workload.

    The score deliberately EXCLUDES optimized initial-fleet capacity x_0 so
    all fleet-cost and fleet-cap sensitivity runs share the same raw scenarios.
    """
    total_demand = sum(scenario.passenger_demand.values())
    incoming = len(scenario.aircraft)
    workload = sum(
        charging_periods_required(a.initial_soc) for a in scenario.aircraft
    )
    return float(
        STRESS_WEIGHT_PASSENGER * total_demand
        - STRESS_WEIGHT_AIRCRAFT * EVTOL_CAPACITY * incoming
        + STRESS_WEIGHT_CHARGING * workload
    )


def build_feature_vectors(scenarios):
    """Attach standardized demand/arrival/workload features to scenarios.

    Uses the scenario time-profile length; this also permits short synthetic
    validation cases with fewer periods than the production configuration.
    """
    scenarios = list(scenarios)
    if not scenarios:
        return scenarios

    n_periods = len(scenarios[0].common_factor)
    if n_periods <= 0 or any(len(s.common_factor) != n_periods for s in scenarios):
        raise ValueError("Scenarios must use the same nonempty time horizon")

    destinations = sorted(
        {d for s in scenarios for (_period, d) in s.passenger_demand}
    )
    vectors = []
    for scenario in scenarios:
        demand = scenario.passenger_demand
        features = [
            demand.get((period, destination), 0)
            for period in range(1, n_periods + 1)
            for destination in destinations
        ]
        arrivals = np.zeros(n_periods, dtype=float)
        workload = np.zeros(n_periods, dtype=float)
        for aircraft in scenario.aircraft:  # stochastic INCOMING aircraft only
            period_index = int(
                (aircraft.arrival_time - HORIZON_START) // BIN_SIZE
            )
            if not 0 <= period_index < n_periods:
                raise ValueError(
                    f"Incoming aircraft outside scenario horizon: S{scenario.id}"
                )
            arrivals[period_index] += 1
            workload[period_index] += charging_periods_required(
                aircraft.initial_soc
            )
        features.extend(arrivals)
        features.extend(workload)
        vectors.append(features)

    matrix = np.asarray(vectors, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Scenario features must be finite")
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1.0  # identical bookings/features carry no distance
    standardized = (matrix - matrix.mean(axis=0)) / std
    for scenario, vector in zip(scenarios, standardized):
        scenario.feature_vector = vector
    return scenarios


def compute_distance_matrix(scenarios) -> np.ndarray:
    """Pairwise standardized Euclidean distances; symmetric and zero diagonal."""
    scenarios = list(scenarios)
    if not scenarios:
        return np.zeros((0, 0), dtype=float)
    matrix = np.stack([s.feature_vector for s in scenarios], axis=0)
    norms = np.sum(matrix * matrix, axis=1)
    distance_sq = np.maximum(
        norms[:, None] + norms[None, :] - 2.0 * matrix @ matrix.T,
        0.0,
    )
    distances = np.sqrt(distance_sq)
    np.fill_diagonal(distances, 0.0)
    return distances


def backward_reduce(scenarios, target_size: int):
    """Greedy weighted-distance deletion within one fixed stress stratum.

    Removed probability mass is assigned to the closest remaining scenario.
    Copies scenario containers, so repeated reductions do not corrupt the
    raw scenario probabilities or previously evaluated sensitivity cases.
    """
    scenarios = [copy(s) for s in scenarios]
    if target_size < 1 or target_size > len(scenarios):
        raise ValueError("Stratum target must be between 1 and its size")
    if target_size == len(scenarios):
        return scenarios

    # Feature vectors are standardized *once across all raw scenarios* by
    # reduce_scenarios, not recalculated separately within each stratum.
    distances = compute_distance_matrix(scenarios)
    weights = np.asarray([s.probability for s in scenarios], dtype=float)
    active = list(range(len(scenarios)))
    while len(active) > target_size:
        local = distances[np.ix_(active, active)].copy()
        np.fill_diagonal(local, np.inf)
        nearest = np.argmin(local, axis=1)
        cost_to_remove = weights[active] * local[
            np.arange(len(active)), nearest
        ]
        removed_position = int(np.argmin(cost_to_remove))
        removed = active[removed_position]
        recipient = active[int(nearest[removed_position])]
        weights[recipient] += weights[removed]
        active.pop(removed_position)

    result = [scenarios[index] for index in active]
    for scenario, index in zip(result, active):
        scenario.probability = float(weights[index])
    return result


def reduce_scenarios(scenarios, target_size: int = TARGET_SCENARIOS):
    """Tail-stratified backward reduction with nearest-neighbor mass transfer.

    The tail contains the largest ceil((1-alpha)*N_raw) stress scores; about
    30% of retained representatives are assigned to that stratum (at least
    two when >=4 total and feasible). Independent scenario probabilities are
    propagated and normalized. No moment-matching optimization is imposed.
    """
    raw = list(scenarios)
    if not raw:
        raise ValueError("At least one raw scenario is required")
    if not isinstance(target_size, int) or not 1 <= target_size <= len(raw):
        raise ValueError("target_size must be an integer from 1 to N_raw")
    if target_size == 1 and len(raw) > 1:
        raise ValueError("Tail-aware reduction requires at least two representatives")
    if any(s.advance_bookings != raw[0].advance_bookings for s in raw[1:]):
        raise ValueError("Stage-1 advance bookings differ across raw scenarios")

    reduced_inputs = [copy(s) for s in raw]
    weights = np.asarray([s.probability for s in reduced_inputs], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Raw scenario probabilities must be finite and nonnegative")
    weights /= weights.sum()
    for s, p in zip(reduced_inputs, weights):
        s.probability = float(p)
        s.stress_score = calculate_stress_score(s)
    build_feature_vectors(reduced_inputs)

    if target_size == len(raw):
        return reduced_inputs

    tail_count = max(1, math.ceil((1.0 - CVaR_ALPHA) * len(raw)))
    ordered = sorted(reduced_inputs, key=lambda s: (s.stress_score, s.id))
    central = ordered[:-tail_count]
    tail = ordered[-tail_count:]
    if not central:
        # Covers degenerate alpha / one-stratum edge cases.
        return backward_reduce(tail, target_size)

    desired_tail = max(1, math.ceil(0.30 * target_size))
    if target_size >= 4:
        desired_tail = max(2, desired_tail)
    tail_target = min(desired_tail, len(tail), target_size - 1)
    central_target = min(target_size - tail_target, len(central))
    while central_target + tail_target < target_size:
        if tail_target < len(tail):
            tail_target += 1
        elif central_target < len(central):
            central_target += 1
        else:
            break

    selected = (
        backward_reduce(central, central_target)
        + backward_reduce(tail, tail_target)
    )
    total = sum(s.probability for s in selected)
    for scenario in selected:
        scenario.probability /= total
    return selected
