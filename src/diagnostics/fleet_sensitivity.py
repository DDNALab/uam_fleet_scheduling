"""Paired initial-fleet cap/cost sensitivity with a fixed reduced scenario set.

From <project>:
  python -m src.diagnostics.fleet_sensitivity --seed 60 \
      --caps 0 6 12 --costs 30 60 120 --solver highs --gap 1e-6

All experiments use the same advance bookings, random demand, incoming aircraft
and scenario probabilities. LoS is left at its configured 30-minute hard limit.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import pulp

from src import config
from src.uam_data_pipeline import build_uam_data
from src.scenario_generator import generate_scenarios
from src.scenario_reduction import reduce_scenarios
from src.model_builder import build_model_input
from src.stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model


def check_rows_and_integrality(model):
    max_row = 0.0
    max_integer = 0.0
    max_bound = 0.0
    for name, row in model.constraints.items():
        value = row.value()
        if value is None:
            raise RuntimeError(f'Missing constraint solution: {name}')
        violation = (abs(value) if row.sense == pulp.LpConstraintEQ
                     else max(0, value) if row.sense == pulp.LpConstraintLE
                     else max(0, -value))
        max_row = max(max_row, violation)
    for v in model.variables():
        value = pulp.value(v)
        if value is None:
            raise RuntimeError(f'Missing variable solution: {v.name}')
        if v.cat == pulp.LpInteger:
            max_integer = max(max_integer, abs(value-round(value)))
        if v.lowBound is not None:
            max_bound = max(max_bound, v.lowBound-value)
        if v.upBound is not None:
            max_bound = max(max_bound, value-v.upBound)
    return {'max_constraint_violation':float(max_row),
            'max_integrality_error':float(max_integer),
            'max_bound_violation':float(max(0.0,max_bound))}


def check_inventory(model_data):
    v=model_data['variables']; T=model_data['periods']; D=model_data['destinations']; report={}
    x0=pulp.value(v['initial_fleet'])
    for sid in model_data['scenarios']:
        deployed=pulp.value(v['deployed_initial'][sid])
        if deployed > x0+1e-5:
            raise AssertionError(f's{sid}: deployed initial aircraft exceeds allocated')
        completions=0.0; departures=0.0; max_prefix=0.0
        for k in T:
            completions += sum(pulp.value(v['w'][sid,h,k-h])
                               for h in model_data['durations'][sid]
                               if (sid,h,k-h) in v['w'] and k-h in T)
            departures += sum(pulp.value(v['flights'][sid,d,k]) for d in D)
            max_prefix=max(max_prefix,departures-deployed-completions)
        if max_prefix>1e-5 or abs(departures-deployed-completions)>1e-5:
            raise AssertionError(f's{sid}: inventory mismatch')
        report[str(sid)]={'initial_deployed':deployed,'incoming_charged':completions,
                          'total_departures':departures,'max_prefix_excess':max(0.,max_prefix)}
    return report


def fixed_input(seed):
    d=build_uam_data()
    expected={row['destination']:float(row['expected_trips']) for _,row in d['od_demand'].iterrows()
              if row['destination'] in config.ACTIVE_DESTINATIONS}
    if not expected:
        raise ValueError('No active destinations in spatial OD table; check config.ACTIVE_DESTINATIONS')
    print('Expected passengers (all geographic destinations):',round(d['od_demand']['expected_trips'].sum(),3))
    print('Expected passengers (active optimization destinations):',round(sum(expected.values()),3))
    print('Configured total passengers:', config.TOTAL_PASSENGER_TRIPS)
    profile=d['time_profile']['weight'].to_numpy()
    generated=generate_scenarios(expected,profile,n_scenarios=config.RAW_SCENARIOS,seed=seed)
    reduced=reduce_scenarios(generated,target_size=config.OPTIMIZATION_SCENARIOS)
    mi=build_model_input(reduced)
    if not math.isclose(sum(s['probability'] for s in mi['scenarios'].values()),1.,abs_tol=1e-9):
        raise AssertionError('Scenario probabilities not normalized')
    print('Fixed booked passengers:', sum(mi['advance_bookings'].values()))
    print('Fixed incoming aircraft per scenario:',
          {sid:len(s['aircraft']) for sid,s in mi['scenarios'].items()})
    print('LoS minutes (not relaxed):', mi['parameters']['los_periods']*mi['parameters']['bin_size'])
    return mi


def solve_case(base, cap, cost, args):
    mi=copy.deepcopy(base)
    mi['parameters']['initial_fleet_max']=int(cap)
    mi['parameters']['costs']['initial_fleet']=float(cost)
    md=build_stochastic_model(mi)
    model=add_constraints_and_objective(md)
    res=solve_model(model,solver=args.solver,gap_rel=args.gap,
                    time_limit=args.time_limit,verbose=args.verbose)
    out={'seed':args.seed,'initial_fleet_cap':int(cap),'initial_fleet_cost_per_aircraft':float(cost),
         'status':res['status'],'solver_reported_gap':res.get('solver_reported_gap'),
         'objective':res.get('objective'),'variables':res.get('n_variables'),
         'constraints':res.get('n_constraints')}
    if res['status']!='Optimal' or res.get('objective') is None:
        print('WARNING: no certified solution for',cap,cost, res)
        return out
    if args.solver=='highs' and (res['solver_reported_gap'] is None or
                                 res['solver_reported_gap'] > args.gap+1e-8):
        raise AssertionError('HiGHS returned Optimal without the requested gap certificate')
    out.update(check_rows_and_integrality(model))
    if max(out[k] for k in ('max_constraint_violation','max_integrality_error','max_bound_violation'))>1e-5:
        raise AssertionError('Failed constraint/integrality audit')
    v=md['variables']; S=md['scenarios']; T=md['periods']; D=md['destinations']
    out['initial_fleet_allocated']=int(round(pulp.value(v['initial_fleet'])))
    out['expected_initial_deployed']=sum(S[sid]['probability']*pulp.value(v['deployed_initial'][sid]) for sid in S)
    out['initial_allocation_cost']=out['initial_fleet_allocated']*cost
    out['expected_unserved_booked']=sum(S[sid]['probability']*pulp.value(v['unserved_booked'][sid,r,d])
                                       for sid in S for r in T for d in D)
    out['expected_unserved_on_demand']=sum(S[sid]['probability']*pulp.value(v['unserved_on_demand'][sid,r,d])
                                          for sid in S for r in T for d in D)
    out['expected_departures']=sum(S[sid]['probability']*pulp.value(v['flights'][sid,d,k])
                                   for sid in S for d in D for k in T)
    out['inventory']=check_inventory(md)
    e=md['expressions']; p=md['parameters']
    check=(sum(S[sid]['probability']*pulp.value(e['scenario_profit'][sid]) for sid in S)
           -pulp.value(e['initial_fleet_cost'])-pulp.value(e['commitment_cost'])
           -pulp.value(e['reservation_cost'])-p['cvar']['weight']*pulp.value(e['cvar']))
    out['objective_reconciliation_error']=abs(check-pulp.value(model.objective))
    if out['objective_reconciliation_error']>1e-5:
        raise AssertionError('Objective-component reconciliation failed')
    print(f"cap={cap:>3} cost={cost:>7g} | x0={out['initial_fleet_allocated']:>3} "
          f"deployed={out['expected_initial_deployed']:.2f} "
          f"unB={out['expected_unserved_booked']:.2f} "
          f"unO={out['expected_unserved_on_demand']:.2f} "
          f"profit-risk={out['objective']:.2f}")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=60)
    p.add_argument('--caps',type=int,nargs='+',default=[0,6,12])
    p.add_argument('--costs',type=float,nargs='+',default=[30,60,120])
    p.add_argument('--solver',choices=('highs','gurobi','cbc'),default='highs')
    p.add_argument('--gap',type=float,default=1e-6)
    p.add_argument('--time-limit',type=int,default=120)
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args()
    if args.out is None:
        args.out = config.DIAGNOSTICS_DIR / f'fleet_sensitivity_seed{args.seed}'
    if args.caps and min(args.caps)<0:
        p.error('All fleet caps must be nonnegative')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    scenarios=fixed_input(args.seed)
    results=[solve_case(scenarios,cap,cost,args) for cost in args.costs for cap in args.caps]
    csv_path=args.out.with_suffix('.csv');json_path=args.out.with_suffix('.json')
    header=['seed','initial_fleet_cap','initial_fleet_cost_per_aircraft','status',
            'solver_reported_gap','objective','initial_fleet_allocated',
            'expected_initial_deployed','initial_allocation_cost',
            'expected_unserved_booked','expected_unserved_on_demand',
            'expected_departures','variables','constraints','max_constraint_violation',
            'max_integrality_error','max_bound_violation','objective_reconciliation_error']
    with csv_path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=header,extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    json_path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Results:',csv_path,'and',json_path)


if __name__=='__main__':
    main()
