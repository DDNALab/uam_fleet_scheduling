"""Run the explicit 14-hour manuscript-profile UAM case (9 destinations).

Requires user's ORIGINAL data/ shapefile and 2022 ACS JSON, unchanged here.
This is a SINGLE-grid study. For paired grid comparisons, generate shared
physical-time latent scenarios and rebin them (see charging_resolution.py's
small paired pilot); same random seed alone is NOT a paired comparison.

From the repository root:
  python -m src.diagnostics.paper_case --preflight-only
  python -m src.diagnostics.paper_case --raw 100 --retained 10 --solver gurobi
  python -m src.diagnostics.paper_case --no-soc-coupling --solver gurobi
"""
import argparse
import json
import math
import time
from pathlib import Path

import pulp
from src import config
from src.diagnostics.demand_design import (
    expected_demand_from_geodata, time_profile_for_horizon,
    booking_capacity_preflight,
)
from src.scenario_generator import generate_advance_bookings, generate_scenarios
from src.scenario_reduction import reduce_scenarios
from src.model_builder import build_model_input
from src.stochastic_model import build_stochastic_model,add_constraints_and_objective,solve_model
from src.group_flow_model import build_group_flow_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin-minutes',type=int,choices=[5,10,15],default=15)
    p.add_argument('--raw',type=int,default=100)
    p.add_argument('--retained',type=int,default=10)
    p.add_argument('--booking-share',type=float,default=.4)
    p.add_argument('--charging-power',type=float,default=120.)
    p.add_argument('--initial-fleet-max',type=int,default=config.INITIAL_FLEET_MAX,
                   help='SYNTHETIC fleet-cap assumption; paper does not calibrate this')
    p.add_argument('--initial-fleet-cost',type=float,default=config.INITIAL_FLEET_COST)
    p.add_argument('--seed',type=int,default=60)
    p.add_argument('--no-soc-coupling',action='store_true')
    p.add_argument('--with-group-flow',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--solver',choices=['gurobi','highs','cbc','auto'],default='gurobi')
    p.add_argument('--time-limit',type=int,default=3600)
    p.add_argument('--gap',type=float,default=.05)
    p.add_argument('--out',type=Path,default=Path('outputs/results/paper_case.json'))
    a=p.parse_args()
    if not 2<=a.retained<=a.raw: p.error('require 2<=retained<=raw')
    if not 0<=a.booking_share<=1: p.error('--booking-share must be in [0,1]')
    physical=config.paper_grid(a.bin_minutes,charging_power=a.charging_power)
    expected,_=expected_demand_from_geodata(scope='all-geographic',passengers=2000)
    profile=time_profile_for_horizon(14*60,start_minutes=physical['horizon_start'],
                                     bin_size=a.bin_minutes)
    bookings=generate_advance_bookings(expected,profile,booking_fraction=a.booking_share)
    check=booking_capacity_preflight(bookings,destinations=expected,
        n_periods=physical['periods'], los_periods=physical['los_periods'],
        seats_per_flight=config.EVTOL_CAPACITY,
        max_departures_per_period=physical['max_departures'],time_limit=90,
        hourly_takeoff_limit=physical['hourly_takeoff_limit'],
        bin_size_minutes=a.bin_minutes)
    if check['status']=='Infeasible':
        raise ValueError(f'Booking preflight infeasible: {check}')
    if check['status']=='Unknown':
        print('WARNING: nominal booking feasibility unknown; full MILP must check')
    report={'study':'paper_14h_not_paired_grid','seed':a.seed,'physical':physical,
            'booking_share':a.booking_share,'booking_preflight':check,
            'no_soc_coupling':a.no_soc_coupling,'raw':a.raw,'retained':a.retained,
            'initial_fleet_max_assumed':a.initial_fleet_max,
            'initial_fleet_cost_assumed':a.initial_fleet_cost}
    if not a.preflight_only:
        scenarios=generate_scenarios(expected,profile,n_scenarios=a.raw,seed=a.seed,
            booking_fraction=a.booking_share,aircraft_passenger_ratio=.30,
            bin_size=a.bin_minutes,horizon_start=physical['horizon_start'],
            stress_beta=0. if a.no_soc_coupling else config.SOC_STRESS_BETA,
            common_beta=0. if a.no_soc_coupling else config.SOC_COMMON_FACTOR_BETA,
            charging_rate=a.charging_power)
        reduced=reduce_scenarios(scenarios,target_size=a.retained)
        overrides={
            'initial_fleet_max':a.initial_fleet_max,
            'charging_facilities':physical['charging_facilities'],
            'max_departures':physical['max_departures'],
            'hourly_takeoff_limit':physical['hourly_takeoff_limit'],
            'require_positive_incoming_duration':True,
            'costs':{'initial_fleet':a.initial_fleet_cost,
                     'charging_reservation':config.CHARGING_RESERVATION_COST*a.bin_minutes/15.,
                     'emergency_capacity':config.EMERGENCY_CAPACITY_COST*a.bin_minutes/15.},
        }
        mi=build_model_input(reduced,overrides=overrides)
        # Rounding may alter exact per-bin feasibility; optimization will
        # apply BOTH hourly takeoff constraints and the per-bin limits.
        report['duration_sets_per_scenario']={str(sid):sorted(set(
            math.ceil(config.BATTERY_CAPACITY*max(0,config.SOC_REQUIRED-a_.initial_soc)/100
                      *60/(a.charging_power*a.bin_minutes)) for a_ in s.aircraft))
            for sid,s in [(s.id,s) for s in reduced]}
        for formulation,factory in (
            [('projected',lambda:build_stochastic_model(mi))]
            + ([('group_flow',lambda:build_group_flow_model(mi))] if a.with_group_flow else [])
        ):
            start=time.perf_counter();md=factory()
            if formulation=='projected':add_constraints_and_objective(md)
            mdl=md['model'];counts={'variables':len(mdl.variables()),
                'integers':sum(v.cat==pulp.LpInteger for v in mdl.variables()),
                'constraints':len(mdl.constraints),
                'build_seconds':time.perf_counter()-start}
            if not a.build_only:
                start=time.perf_counter()
                result=solve_model(mdl,solver=a.solver,gap_rel=a.gap,
                                   time_limit=a.time_limit,verbose=False)
                counts.update(result);counts['solve_seconds']=time.perf_counter()-start
            report[formulation]=counts
            print(formulation,counts,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('Saved',a.out)

if __name__=='__main__':main()
