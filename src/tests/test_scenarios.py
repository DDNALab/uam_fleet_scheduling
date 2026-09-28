import numpy as np

from src.config import ARRIVAL_SOC_MIN, ARRIVAL_SOC_MAX
from src.scenario_generator import (
    conditional_soc_mean,
    generate_scenarios,
    sample_arrival_soc,
)
from src.scenario_reduction import (
    calculate_stress_score,
    charging_periods_required,
    reduce_scenarios,
)
from src.scenario_generator import Aircraft, Scenario


def test_advance_bookings_are_scenario_independent():
    expected = {"D": 20}
    profile = np.ones(4) / 4
    scenarios = generate_scenarios(expected, profile, n_scenarios=5, seed=3, booking_fraction=0.4)
    assert sum(scenarios[0].advance_bookings.values()) == 8
    assert all(s.advance_bookings == scenarios[0].advance_bookings for s in scenarios)


def test_conditional_soc_mean_decreases_with_stress_and_common_factor():
    assert conditional_soc_mean(2.0, 0.0) < conditional_soc_mean(0.5, 0.0)
    assert conditional_soc_mean(1.0, 1.0) < conditional_soc_mean(1.0, -1.0)


def test_truncated_soc_samples_respect_bounds():
    np.random.seed(7)
    values = [sample_arrival_soc(2.0, 1.0) for _ in range(500)]
    assert min(values) >= ARRIVAL_SOC_MIN
    assert max(values) <= ARRIVAL_SOC_MAX


def test_tail_score_uses_charging_duration_workload():
    scenario = Scenario(
        id=0,
        probability=1.0,
        advance_bookings={(1, "D"): 4},
        on_demand_demand={(1, "D"): 2},
        aircraft=[Aircraft(id=0, arrival_time=480, initial_soc=50.0)],
        common_factor=np.zeros(4),
    )
    expected = 6 - 4 + 0.05 * charging_periods_required(50.0)
    assert np.isclose(calculate_stress_score(scenario), expected)


def test_tail_aware_reduction_preserves_probability_mass():
    expected = {"D": 20}
    profile = np.ones(4) / 4
    scenarios = generate_scenarios(expected, profile, n_scenarios=10, seed=4, booking_fraction=0.4)
    reduced = reduce_scenarios(scenarios, target_size=4)
    assert len(reduced) == 4
    assert np.isclose(sum(s.probability for s in reduced), 1.0)
    assert all(hasattr(s, "stress_score") for s in reduced)
