"""Fast checks for revised domains, strict A4, value contrasts, and pilot grids."""
import copy
import pytest
pulp=pytest.importorskip('pulp')

from src.stochastic_model import build_stochastic_model, add_constraints_and_objective
from src.group_flow_model import build_group_flow_model
from src.policy_benchmarks import policy_value_measures
from src.diagnostics.charging_resolution import build_physical_pilot


def test_booking_certificate_is_continuous():
    md=build_stochastic_model(build_physical_pilot(15))
    assert all(var.cat==pulp.LpContinuous for var in md['variables']['protected_bookings'].values())


def test_strict_A4_rejects_zero_duration_incoming_aircraft():
    model_input=build_physical_pilot(15)
    model_input['scenarios'][0]['aircraft'][0]['initial_soc']=70.
    with pytest.raises(ValueError,match='positive-duration'):
        build_stochastic_model(model_input)


def test_group_flow_has_integer_group_starts_and_ready_inventory():
    md=build_group_flow_model(build_physical_pilot(15))
    assert md['variables']['group_ready']
    assert md['variables']['group_departures']
    assert all(var.cat==pulp.LpInteger for var in md['variables']['chi'].values())
    # Duration margins are expressions, not additional integer variables.
    assert not any(x.name.startswith('DurationStart_') for x in md['model'].variables())


def test_identical_physical_arrivals_at_each_resolution():
    cardinalities=[]
    for bins in (5,10,15):
        md=build_stochastic_model(build_physical_pilot(bins))
        cardinalities.append(max(len(md['durations'][s]) for s in md['scenarios']))
        assert md['parameters']['los_periods']*bins==30
        assert md['parameters']['hourly_takeoff_limit']==12
    assert cardinalities[0]>=cardinalities[2]


def test_population_value_contrasts():
    vals=policy_value_measures({'rn':110.,'bo':100.,'ev':95.,'ws':120.})
    assert vals=={'VSS':15.,'EVPI':10.,'VOP':10.}


def test_hourly_booking_preflight_uses_extra_limit():
    from src.diagnostics.demand_design import booking_capacity_preflight
    # Three bookings of four seats requested at the beginning of one hour.
    # Three 5-minute flights fit the per-period limit, but an hourly cap of 2
    # makes the first-stage booking-protection subsystem infeasible.
    bookings={(1,'A'):12}
    result=booking_capacity_preflight(
        bookings,destinations=['A'],n_periods=12,los_periods=0,
        seats_per_flight=4,max_departures_per_period=3,
        hourly_takeoff_limit=2,bin_size_minutes=5)
    assert result['status']=='Infeasible'


def test_no_coupling_scenario_generator_and_reduction_share_physics():
    import numpy as np
    from src.scenario_generator import generate_scenarios
    from src.scenario_reduction import charging_periods_required as reducer_duration
    from src.stochastic_model import charging_periods_required as optimizer_duration
    from src.model_builder import build_model_input
    raw=generate_scenarios({'A':40.},np.ones(12)/12,n_scenarios=2,seed=9,
        booking_fraction=.4,aircraft_passenger_ratio=.8,bin_size=5,
        stress_beta=0.,common_beta=0.,soc_std=0.)
    assert any(s.aircraft for s in raw)
    assert all(abs(a.initial_soc-50.)<1e-9 for s in raw for a in s.aircraft)
    md=build_model_input(raw)
    assert md['parameters']['bin_size']==5 and md['parameters']['periods']==12
    assert md['parameters']['los_periods']==6
    for scenario in raw:
        for aircraft in scenario.aircraft:
            assert reducer_duration(aircraft.initial_soc,scenario=scenario)==optimizer_duration(
                aircraft.initial_soc,md['parameters'])
