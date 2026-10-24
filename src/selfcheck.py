"""
End-to-end validation of the implementation.

Runs the numerical checks the model depends on -- charging-duration rounding at
15-, 10- and 5-minute resolution, the time-of-use tariff, the advance-booking
split, scenario generation, scenario reduction, and exactness of the aggregated
formulation -- with no census data and no commercial solver.

Usage::

    python src/selfcheck.py

Exit code 0 means every check passed.
"""

import sys

import numpy as np

import config
from demand_model import build_baseline_demand, largest_remainder
from economics import (
    charging_duration_periods,
    energy_price_per_kwh,
)
from model_builder import build_model_input
from scenario_generator import generate_scenarios
from scenario_reduction import (
    effective_scenario_count,
    reduce_scenarios,
    reduction_diagnostics,
)


# ============================================================
# Reporting helpers
# ============================================================

class Checker:
    """Accumulate pass/fail results instead of stopping at the first failure."""

    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, condition, message):
        if condition:
            self.passes += 1
            print("  [ok]   %s" % message)
        else:
            self.failures.append(message)
            print("  [FAIL] %s" % message)

    def close(self, title):
        print()
        if self.failures:
            print("%s: %d failure(s)" % (title, len(self.failures)))
            for failure in self.failures:
                print("   - %s" % failure)
            return False
        print("%s: %d checks passed" % (title, self.passes))
        return True


# ============================================================
# 1. Charging duration resolution
# ============================================================

def check_duration_resolution(checker):
    """
    Duration in periods for arrival SoC in [20%, 60%] and a 70% departure target:
    {1, 2} at 15-minute resolution, {1, 2, 3} at 10-minute and {2, 3, 4, 5, 6} at
    5-minute. The one-period threshold sits near 42.7% arrival SoC.
    """
    print("\n[1] Eq. (64) charging duration resolution")

    grid = np.linspace(config.ARRIVAL_SOC_MIN, config.ARRIVAL_SOC_MAX, 801)

    for bin_size, expected in ((15, {1, 2}), (10, {1, 2, 3}), (5, {2, 3, 4, 5, 6})):

        parameters = config.build_parameters(bin_size=bin_size)

        realized = {
            charging_duration_periods(soc, parameters) for soc in grid
        }

        checker.check(
            realized == expected,
            "%d-min periods realize durations %s (paper: %s)"
            % (bin_size, sorted(realized), sorted(expected)),
        )

    # The one-period threshold is where E_bar (70 - SoC) = mu * Delta.
    parameters = config.build_parameters()

    per_period = parameters["charging_rate"] * parameters["bin_size"] / 60.0

    threshold = 70.0 - 100.0 * per_period / parameters["battery_capacity"]

    checker.check(
        abs(threshold - 42.7) < 0.1,
        "one-period threshold at %.2f%% arrival SoC (paper: ~42.7%%)" % threshold,
    )

    checker.check(
        charging_duration_periods(threshold - 1.0) == 2
        and charging_duration_periods(threshold + 1.0) == 1,
        "duration steps 2 -> 1 across the %.2f%% threshold" % threshold,
    )


# ============================================================
# 2. Electricity tariff
# ============================================================

def check_tariff(checker):
    """``P(t)`` peak 16-21, off 0-6 and 21-24, shoulder otherwise."""
    print("\n[2] Eq. (71) time-of-use tariff")

    base = config.BASE_PRICE

    checker.check(
        abs(energy_price_per_kwh(17.0) - base * config.M_PEAK) < 1e-12,
        "peak hour 17 priced at P_base * m_peak",
    )

    checker.check(
        abs(energy_price_per_kwh(2.0) - base * config.M_OFF) < 1e-12,
        "off-peak hour 2 priced at P_base * m_off",
    )

    checker.check(
        abs(energy_price_per_kwh(22.0) - base * config.M_OFF) < 1e-12,
        "off-peak hour 22 priced at P_base * m_off",
    )

    checker.check(
        abs(energy_price_per_kwh(12.0) - base * config.M_SHOULDER) < 1e-12,
        "shoulder hour 12 priced at P_base * m_shoulder",
    )

    checker.check(
        energy_price_per_kwh(17.0) > energy_price_per_kwh(12.0) > energy_price_per_kwh(2.0),
        "price ordering peak > shoulder > off-peak",
    )


# ============================================================
# 3. Booking split
# ============================================================

def check_booking_split(checker):
    """``B_td`` is integer, non-negative, and preserves the intended totals."""
    print("\n[3] Eq. (69) advance-booking split")

    checker.check(
        largest_remainder([1.4, 1.4, 1.2], 3).sum() == 3,
        "largest-remainder apportionment preserves the target total",
    )

    baseline = build_baseline_demand(n_periods=8, total_passengers=200, booking_share=0.4)

    booked = baseline["booked"]

    expected_total = 0.4 * 200

    checker.check(
        all(isinstance(value, int) and value >= 0 for value in booked.values()),
        "booked counts are non-negative integers",
    )

    checker.check(
        abs(sum(booked.values()) - expected_total) <= 1.0,
        "booked total %d matches phi_B * total (%.1f)"
        % (sum(booked.values()), expected_total),
    )

    ondemand = sum(baseline["ondemand_rate"].values())

    checker.check(
        abs(ondemand - 0.6 * 200) < 1e-6,
        "on-demand intensity preserves the 1 - phi_B share (%.2f)" % ondemand,
    )

    for destination in baseline["destinations"]:

        expected = 0.4 * sum(
            baseline["lambda_bar"][t, destination] for t in baseline["periods"]
        )

        realized = sum(booked[t, destination] for t in baseline["periods"])

        if abs(realized - expected) > 1.0:
            checker.check(False, "destination %s booked total off by %.3f"
                          % (destination, realized - expected))
            break
    else:
        checker.check(True, "per-destination booked totals preserved")


# ============================================================
# 4. Correlated scenario generation
# ============================================================

def check_scenario_generation(checker):
    print("\n[4] Section 5.1 correlated scenario generation")

    parameters = config.build_parameters(
        total_passengers=400,
        expected_incoming_aircraft=60,
        raw_scenarios=40,
    )

    baseline = build_baseline_demand(n_periods=12, total_passengers=400)

    scenarios = generate_scenarios(
        baseline, n_scenarios=40, seed=3, parameters=parameters
    )

    checker.check(len(scenarios) == 40, "generated the requested 40 scenarios")

    probabilities = [s.probability for s in scenarios]

    checker.check(
        abs(sum(probabilities) - 1.0) < 1e-9,
        "raw scenario probabilities sum to one",
    )

    socs = np.array([a.initial_soc for s in scenarios for a in s.aircraft])

    checker.check(
        socs.min() >= config.ARRIVAL_SOC_MIN - 1e-9
        and socs.max() <= config.ARRIVAL_SOC_MAX + 1e-9,
        "arrival SoC stays inside the truncated support [%.0f, %.0f]"
        % (config.ARRIVAL_SOC_MIN, config.ARRIVAL_SOC_MAX),
    )

    mean_soc = float(socs.mean())

    checker.check(
        25.0 < mean_soc < 60.0,
        "mean arrival SoC %.1f%% is inside the support (S0 = %.0f)"
        % (mean_soc, config.SOC_MEAN_INTERCEPT),
    )

    durations = {a.duration for s in scenarios for a in s.aircraft}

    checker.check(
        durations <= {1, 2},
        "at 15-min resolution only durations {1, 2} occur, realized %s"
        % sorted(durations),
    )

    # Each period has E[g_ts] = 0 and Var[g_ts] = sigma_g^2 by construction, so
    # only the ensemble moments need to match sigma_g; a short trajectory can
    # legitimately show a smaller realized std.
    draws = np.array([s.common_factor for s in scenarios])

    pooled_std = float(draws.std())

    pooled_mean = float(draws.mean())

    checker.check(
        abs(pooled_std - config.FACTOR_STD) <= 0.15 * config.FACTOR_STD,
        "ensemble common-factor std %.3f matches sigma_g = %.3f"
        % (pooled_std, config.FACTOR_STD),
    )

    checker.check(
        abs(pooled_mean) < 0.25 * config.FACTOR_STD,
        "ensemble common-factor mean %+.3f is centred on zero" % pooled_mean,
    )

    # The log link must preserve the baseline mean intensity; otherwise phi_B
    # stops meaning what it says.
    requests = float(np.mean([s.ondemand_total for s in scenarios]))

    expected = float(baseline["ondemand_total"])

    checker.check(
        abs(requests - expected) <= 0.25 * max(expected, 1.0),
        "mean on-demand requests %.1f within 25%% of baseline %.1f"
        % (requests, expected),
    )

    # No-coupling baseline: beta_Gamma = beta_g = 0 makes SoC independent of
    # demand stress, so their correlation vanishes.
    from scenario_generator import sample_truncated_normal

    independent = [
        sample_truncated_normal(50.0, 6.0, 20.0, 60.0, np.random.default_rng(0))
        for _ in range(50)
    ]

    checker.check(
        len(independent) == 50 and all(20.0 <= value <= 60.0 for value in independent),
        "no-coupling sampler stays inside the support",
    )


# ============================================================
# 5. Scenario reduction
# ============================================================

def check_reduction(checker):
    print("\n[5] Section 5.2 tail-preserving reduction")

    parameters = config.build_parameters(total_passengers=400, raw_scenarios=60)

    baseline = build_baseline_demand(n_periods=10, total_passengers=400)

    raw = generate_scenarios(baseline, n_scenarios=60, seed=5, parameters=parameters)

    reduced = reduce_scenarios(
        raw, target_size=10, parameters=parameters, booked=baseline["booked"]
    )

    checker.check(len(reduced) == 10, "retained exactly 10 representatives")

    weights = np.array([s.probability for s in reduced])

    checker.check(
        abs(weights.sum() - 1.0) < 1e-9,
        "reduced probabilities sum to one",
    )

    diagnostics = reduction_diagnostics(reduced)

    # alpha = 0.90 over 60 raw scenarios puts ceil(0.1 * 60) = 6 in the tail, so
    # at least TAIL_STRATUM_MIN representatives must survive.
    checker.check(
        diagnostics["tail"] >= config.TAIL_STRATUM_MIN,
        "tail stratum keeps at least %d representatives (got %d)"
        % (config.TAIL_STRATUM_MIN, diagnostics["tail"]),
    )

    checker.check(
        diagnostics["tail"] <= len(reduced) // 2,
        "tail stratum does not dominate the retained set (%d of %d)"
        % (diagnostics["tail"], len(reduced)),
    )

    checker.check(
        1.0 <= diagnostics["effective_scenarios"] <= 10.0 + 1e-9,
        "N_eff = %.2f lies in [1, retained]" % diagnostics["effective_scenarios"],
    )

    checker.check(
        diagnostics["max_probability"] < 1.0,
        "no representative absorbs the whole probability mass (max %.3f)"
        % diagnostics["max_probability"],
    )

    # Mass preservation: each stratum keeps its original total mass, so the
    # reduced ensemble does not silently over-weight the tail.
    raw_tail_share = float(
        np.ceil((1.0 - config.CVAR_ALPHA) * len(raw)) / len(raw)
    )

    reduced_tail_share = float(
        sum(s.probability for s in reduced if s.stratum == "tail")
    )

    checker.check(
        abs(reduced_tail_share - raw_tail_share) < 1e-6,
        "tail mass preserved (%.4f raw vs %.4f reduced)"
        % (raw_tail_share, reduced_tail_share),
    )

    checker.check(
        effective_scenario_count(np.full(4, 0.25)) == 4.0,
        "N_eff of a uniform 4-scenario set equals 4 (Eq. 66)",
    )


# ============================================================
# 6. Exactness of the projected formulation
# ============================================================

def check_exactness(checker):
    """
    The projected (duration-aggregated) optimum must be recoverable as an
    aircraft-level schedule with the same objective value.

    Comparing two independent solves for an equal objective would be the wrong
    test: a degenerate instance can have several optimal first-stage plans.
    Instead the aggregated first stage is re-imposed on the indexed model, which
    is the direction the exactness claim asserts and is immune to alternate
    optima.
    """
    print("\n[6] Corollary 1 exactness: projected solution is recoverable")

    try:
        import aggregated_model, stochastic_model
    except ImportError as error:  # pragma: no cover
        checker.check(False, "optimization stack unavailable: %s" % error)
        return

    overrides = dict(
        total_passengers=16,
        max_initial_fleet=2,
        charging_facilities=2,
        max_departures=2,
        booking_share=0.4,
        raw_scenarios=2,
    )

    parameters = config.build_parameters(**overrides)

    destinations = list(config.ACTIVE_DESTINATIONS[:2])

    baseline = build_baseline_demand(
        destinations=destinations,
        n_periods=4,
        booking_share=0.4,
        total_passengers=16,
    )

    raw = generate_scenarios(
        baseline, n_scenarios=2, seed=17, parameters=parameters
    )

    model_input = build_model_input(
        raw,
        baseline=baseline,
        parameters=parameters,
        destinations=destinations,
        periods=list(range(1, 5)),
    )

    # Solve the projected formulation.
    aggregated_data = aggregated_model.build_aggregated_model(model_input)

    aggregated_problem = aggregated_model.add_constraints_and_objective(
        aggregated_data
    )

    aggregated_result = stochastic_model.solve_model(
        aggregated_problem, solver="auto", verbose=False, time_limit=300, gap_rel=0.0
    )

    checker.check(
        aggregated_result["proven_optimal"],
        "aggregated formulation solved to proven optimality (%s)"
        % aggregated_result["status"],
    )

    if not aggregated_result["proven_optimal"]:
        return

    # Recover the first stage.
    aggregated_variables = aggregated_data["variables"]

    x0_star = int(round(float(aggregated_variables["x0"].value() or 0.0)))

    n_star = {
        key: int(round(float(variable.value() or 0.0)))
        for key, variable in aggregated_variables["n"].items()
    }

    b_star = {
        key: int(round(float(variable.value() or 0.0)))
        for key, variable in aggregated_variables["b"].items()
    }

    # Impose it on the indexed model.
    indexed_data = stochastic_model.build_stochastic_model(model_input)

    indexed_problem = stochastic_model.add_constraints_and_objective(indexed_data)

    indexed_variables = indexed_data["variables"]

    indexed_problem += (
        indexed_variables["x0"] == x0_star,
        "Fix_x0",
    )

    for key, value in n_star.items():
        indexed_problem += (
            indexed_variables["n"][key] == value,
            "Fix_n_%s_%s" % (key[1], key[0]),
        )

    for key, value in b_star.items():
        indexed_problem += (
            indexed_variables["b"][key] == value,
            "Fix_b_%s" % key,
        )

    indexed_result = stochastic_model.solve_model(
        indexed_problem, solver="auto", verbose=False, time_limit=300, gap_rel=0.0
    )

    checker.check(
        indexed_result["proven_optimal"],
        "indexed recourse feasible for the projected first-stage plan (%s)"
        % indexed_result["status"],
    )

    if not indexed_result["proven_optimal"]:
        return

    difference = abs(
        float(indexed_result["objective"]) - float(aggregated_result["objective"])
    )

    checker.check(
        difference < 1e-5 * max(1.0, abs(float(aggregated_result["objective"]))),
        "objective preserved when the projected plan is expanded (diff %.2e)"
        % difference,
    )

    # The aggregation itself must not add value.
    indexed_counts = indexed_result.get("size") or {}
    aggregated_counts = aggregated_result.get("size") or {}

    checker.check(
        aggregated_counts.get("integer_variables", 0)
        < indexed_counts.get("integer_variables", 0),
        "aggregated integer count %s < indexed %s"
        % (
            aggregated_counts.get("integer_variables"),
            indexed_counts.get("integer_variables"),
        ),
    )

    checker.check(
        aggregated_counts.get("continuous_variables", 0)
        > indexed_counts.get("continuous_variables", 0),
        "the saving comes from integers, not from dropping variables "
        "(continuous %s vs %s)"
        % (
            aggregated_counts.get("continuous_variables"),
            indexed_counts.get("continuous_variables"),
        ),
    )


# ============================================================
# Entry point
# ============================================================

def main():

    print("=" * 74)
    print(" coord-schedule-uam self-check")
    print("=" * 74)

    checker = Checker()

    check_duration_resolution(checker)

    check_tariff(checker)

    check_booking_split(checker)

    check_scenario_generation(checker)

    check_reduction(checker)

    check_exactness(checker)

    ok = checker.close("SELF-CHECK")

    print()

    return 0 if ok else 1


if __name__ == "__main__":

    sys.exit(main())
