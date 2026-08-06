from pathlib import Path
import sys

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coord_schedule_uam.scenario_generation import energy_price_per_kwh
from coord_schedule_uam.scenario_reduction import (
    scenario_reduction_backward_with_evtol_filter,
)
from coord_schedule_uam.model_construction import conditional_soc_sampling_per_evtol


def _scenario(identifier, passengers, evtols):
    a_t = np.asarray(passengers)
    e_t = np.asarray(evtols)
    return {
        "id": identifier,
        "a_t": a_t,
        "e_t": e_t,
        "total_passengers": int(a_t.sum()),
        "total_evtols": int(e_t.sum()),
    }


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

    reduced = scenario_reduction_backward_with_evtol_filter(
        scenarios,
        target_size=2,
        plot=False,
    )

    assert len(reduced) == 2
    assert np.isclose(sum(item["probability"] for item in reduced), 1.0)


def test_leading_empty_evtol_bin_has_finite_load_ratio():
    scenarios = [_scenario(0, [5, 2], [0, 1])]
    scenarios[0]["probability"] = 1.0

    result = conditional_soc_sampling_per_evtol(scenarios)

    assert result[0]["rho_t"][0] == 0.0
    assert np.isfinite(result[0]["avg_load_ratio"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
