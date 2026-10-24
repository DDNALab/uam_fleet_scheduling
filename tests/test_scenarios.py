from pathlib import Path
import sys

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from economics import energy_price_per_kwh
from scenario_generator import Scenario
from scenario_reduction import effective_scenario_count, reduce_scenarios


def _scenario(identifier, passengers, evtols):
    a_t = np.asarray(passengers, dtype=float)
    e_t = np.asarray(evtols, dtype=float)
    return Scenario(
        id=identifier,
        probability=1.0 / 4.0,
        ondemand_demand={(t + 1, "DFW"): int(value) for t, value in enumerate(a_t)},
        aircraft=[],
        common_factor=np.ones(len(a_t)),
        arrival_counts=e_t,
    )


def test_time_of_use_prices_have_expected_order():
    assert energy_price_per_kwh(17) > energy_price_per_kwh(12)
    assert energy_price_per_kwh(12) > energy_price_per_kwh(2)


def test_reduction_preserves_probability_mass():
    scenarios = [
        _scenario(0, [1, 2, 1], [1, 0, 1]),
        _scenario(1, [2, 2, 2], [1, 1, 0]),
        _scenario(2, [4, 3, 2], [0, 1, 1]),
        _scenario(3, [0, 1, 0], [1, 0, 0]),
    ]

    reduced = reduce_scenarios(scenarios, target_size=2)

    assert len(reduced) == 2
    assert np.isclose(sum(item.probability for item in reduced), 1.0)


def test_effective_scenario_count_reflects_concentration():
    assert np.isclose(effective_scenario_count([1.0]), 1.0)
    assert np.isclose(effective_scenario_count([0.5, 0.5]), 2.0)
    assert np.isclose(effective_scenario_count([0.9, 0.1]), 1.0 / (0.81 + 0.01))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
