"""Paired physical-instance charging-grid pilot (5, 10, and 15 min).

All grids use the same physical bookings, on-demand events, aircraft arrivals,
SoC, battery/charging power, 30-min service windows, 2 physical chargers,
12 takeoffs/hour, and per-HOUR reservation and emergency costs. At different
resolutions the feasible discrete schedules need not have identical objectives.

    python -m src.diagnostics.charging_resolution --no-solve
    python -m src.diagnostics.charging_resolution --solver highs --time-limit 120

This controlled small pilot is NOT the 14-hour DFW numerical study.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import pulp

from src.group_flow_model import build_group_flow_model
from src.stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model


# Event timestamps in elapsed minutes relative to 08:00. Every timestamp
# aligns to all tested grids; no different random draw is introduced by rebinning.
PHYSICAL_BOOKINGS = {(0, 'A'): 2, (30, 'B'): 2, (60, 'A'): 2, (90, 'B'): 2}
PHYSICAL_ON_DEMAND = (
    {(30, 'A'): 2, (90, 'B'): 1},
    {(0, 'B'): 1, (60, 'A'): 3},
)
PHYSICAL_AIRCRAFT = (
    ((0,35.), (0,50.), (30,60.), (30,25.), (60,40.), (90,55.)),
    ((0,50.), (30,35.), (60,60.), (60,25.), (90,40.)),
)


def build_physical_pilot(bin_size):
    if bin_size not in (5,10,15):
        raise ValueError('Supported pilot grids: 5, 10, 15 minutes')
    periods = 120 // bin_size
    idx = lambda minute: minute // bin_size + 1
    scenarios = {}
    for sid, (on_events, air_events) in enumerate(zip(PHYSICAL_ON_DEMAND, PHYSICAL_AIRCRAFT)):
        scenarios[sid] = {
            'probability': .5,
            'on_demand_demand': {(idx(t),d): count for (t,d),count in on_events.items()},
            'aircraft': {aid:{'arrival_period':idx(t),'initial_soc':soc}
                         for aid,(t,soc) in enumerate(air_events)},
            'common_factor': [0.0]*periods,
        }
    hourly_limit = 12
    return {
        'parameters': {
            'periods':periods, 'bin_size':bin_size,
            'horizon':{'start':480,'end':600,'bin_size':bin_size},
            'los_periods':30//bin_size,
            'battery_capacity':110., 'charging_rate':120., 'soc_min':70.,
            'require_positive_incoming_duration':True,
            'initial_fleet_max':3,
            'charging_facilities':2,
            # The hourly cap prevents rounding a 5-minute per-bin limit into
            # more permitted takeoffs over the same physical hour.
            'hourly_takeoff_limit':hourly_limit,
            'max_departures':math.ceil(hourly_limit*bin_size/60),
            'evtol_capacity':4,
            'destination_fares':{'A':36.,'B':42.},
            'base_price':.15, 'peak_multiplier':1.,
            'shoulder_multiplier':1., 'off_multiplier':1.,
            'costs':{
                'initial_fleet':30.,
                'charging_reservation':8.*bin_size/60,  # $8 per charger-hour
                'emergency_capacity':20.*bin_size/60,
                'commitment':1., 'added_departure':5.,
                'cancelled_departure':3.,'flight':8.,
                'unserved_booked':150., 'unserved_on_demand':100.,
            },
            'cvar':{'alpha':.9,'weight':22.,'booked_weight':2.,
                    'on_demand_weight':1.},
        },
        'advance_bookings':{(idx(t),d):count for (t,d),count in PHYSICAL_BOOKINGS.items()},
        'scenarios':scenarios,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-solve', action='store_true')
    p.add_argument('--solver', choices=['highs','gurobi','cbc','auto'], default='highs')
    p.add_argument('--gap',type=float,default=.001)
    p.add_argument('--time-limit',type=float,default=120)
    p.add_argument('--out',type=Path,default=Path('outputs/diagnostics/charging_resolution.csv'))
    args=p.parse_args()
    rows=[]
    for bin_size in (5,10,15):
        mi=build_physical_pilot(bin_size)
        for name,factory in (
            ('projected',lambda:build_stochastic_model(mi)),
            ('group_flow',lambda:build_group_flow_model(mi)),
        ):
            md=factory()
            if name=='projected':
                add_constraints_and_objective(md)
            model=md['model']
            card=[len(set(g['duration'] for g in md['groups'][sid].values()))
                  for sid in md['scenarios']]
            row={'bin_minutes':bin_size, 'formulation':name,
                 'n_periods':mi['parameters']['periods'],
                 'distinct_durations_per_scenario':str(card),
                 'variables':len(model.variables()),
                 'integer_variables':sum(x.cat==pulp.LpInteger for x in model.variables()),
                 'constraints':len(model.constraints)}
            if not args.no_solve:
                start=time.perf_counter()
                solve=solve_model(model,solver=args.solver, gap_rel=args.gap,
                                  time_limit=args.time_limit,verbose=False)
                row.update(status=solve['status'],objective=solve['objective'],
                           gap=solve.get('solver_reported_gap'),
                           seconds=time.perf_counter()-start)
            rows.append(row)
            print(row,flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
    print('Saved',args.out)

if __name__=='__main__':
    main()
