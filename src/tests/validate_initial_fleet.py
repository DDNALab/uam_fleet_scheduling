"""Small-instance independent exactness and initial-inventory validation.

Run as a package module from the project root (parent of src).
Does NOT require Census/GeoPandas or regenerate the empirical demand data.

    python -m src.tests.validate_initial_fleet --solver highs --gap 1e-7 --time-limit 120
    python -m src.tests.validate_initial_fleet --solver highs --seeds 60 61 62

It compares the patched integer projected MILP to an independently coded,
aircraft-indexed MILP on matched exogenous scenarios, CVaR, costs, and bounds.
"""
import argparse
import copy
import heapq
import importlib
import json
import math
from pathlib import Path
import random
import sys


def modules():
    # The script lives in src/tests, so its great-grandparent is the project root.
    # Package-mode invocation is preferred; direct invocation remains supported.
    project_dir = Path(__file__).resolve().parents[2]
    if not (project_dir / 'src' / 'stochastic_model.py').is_file():
        raise SystemExit('Expected src/stochastic_model.py under the project root.')
    sys.path.insert(0, str(project_dir))
    try:
        import pulp
        projected = importlib.import_module('src.stochastic_model')
        reference = importlib.import_module('src.detailed_reference_model')
    except ImportError as exc:
        raise SystemExit(f'Import failed: {exc}. Activate uam_net_opt, then install pulp highspy.') from exc
    return pulp, projected, reference


def toy_input(seed=60,cap=3,fleet_cost=35.0,case='mixed'):
    rng=random.Random(seed)
    p={
      'periods':4,'bin_size':15,'los_periods':2,
      'horizon':{'start':480,'end':540,'bin_size':15},
      'base_price':.15,'peak_multiplier':2.,'shoulder_multiplier':1.,'off_multiplier':.8,
      'battery_capacity':110.,'charging_rate':120.,'soc_min':70.,'evtol_capacity':4,
      'initial_fleet_max':cap,'charging_facilities':2,'max_departures':2,
      'destination_fares':{'A':36.,'B':42.},
      'costs':{'initial_fleet':fleet_cost,'charging_reservation':2.,'commitment':1.,
               'added_departure':7.,'cancelled_departure':4.,'emergency_capacity':10.,
               'unserved_booked':150.,'unserved_on_demand':100.,'flight':6.},
      'cvar':{'alpha':.8,'weight':18.,'booked_weight':2.,'on_demand_weight':1.},
    }
    if case=='early_booking':
        p['periods']=3
        p['horizon']['end']=525
        p['initial_fleet_max']=cap
        return {'parameters':p,'advance_bookings':{(1,'A'):4},
                'scenarios':{0:{'probability':1.0,'aircraft':{},'on_demand_demand':{(1,'A'):0},'common_factor':[0,0,0]}}}
    if case=='late_arrivals':
        return {'parameters':p,'advance_bookings':{(1,'A'):3,(2,'B'):2},
                'scenarios':{
                    0:{'probability':.5,'aircraft':{1:{'arrival_period':3,'initial_soc':45.}},
                       'on_demand_demand':{(1,'A'):1,(3,'B'):1},'common_factor':[0]*4},
                    1:{'probability':.5,'aircraft':{2:{'arrival_period':4,'initial_soc':60.}},
                       'on_demand_demand':{(2,'A'):2},'common_factor':[0]*4}}}
    # Nontrivial 2-scenario comparison, with charging and zero-duration starts.
    return {'parameters':p,'advance_bookings':{(1,'A'):3,(3,'B'):2},
            'scenarios':{
                0:{'probability':.6,'aircraft':{
                        11:{'arrival_period':1,'initial_soc':70.},
                        12:{'arrival_period':2,'initial_soc':45.+rng.choice([0.,5.])},
                        13:{'arrival_period':3,'initial_soc':60.}},
                   'on_demand_demand':{(1,'A'):rng.choice([1,2]),(2,'B'):1},'common_factor':[0]*4},
                1:{'probability':.4,'aircraft':{
                        21:{'arrival_period':2,'initial_soc':55.},
                        22:{'arrival_period':3,'initial_soc':35.}},
                   'on_demand_demand':{(1,'A'):1,(3,'B'):rng.choice([1,2])},'common_factor':[0]*4}}}


def check_solver(pulp, core, md, solver, gap, limit):
    result=core.solve_model(md['model'], solver=solver,time_limit=limit,gap_rel=gap,verbose=False)
    if result['status']!='Optimal':
        raise AssertionError(f"Nonoptimal solve: {result}")
    # HiGHS accessible through PuLP's solverModel; insist on a known tight bound.
    reported=result.get('solver_reported_gap')
    if solver=='highs' and (reported is None or reported>gap+1e-8):
        raise AssertionError(f'HiGHS gap not certified within {gap}: {result}')
    return result


def audit(pulp, md):
    model=md['model']
    max_row=0.
    for name,c in model.constraints.items():
        v=c.value()
        if v is None:raise AssertionError(f'Missing row solution {name}')
        residual=abs(v) if c.sense==pulp.LpConstraintEQ else (max(0,v) if c.sense==pulp.LpConstraintLE else max(0,-v))
        max_row=max(max_row,residual)
    int_error=max((abs(pulp.value(x)-round(pulp.value(x))) for x in model.variables() if x.cat==pulp.LpInteger),default=0.)
    bound_error=max((max(0., x.lowBound-pulp.value(x)) if x.lowBound is not None else 0.
                     for x in model.variables()),default=0.)
    bound_error=max(bound_error,max((max(0., pulp.value(x)-x.upBound)
                    if x.upBound is not None else 0. for x in model.variables()),default=0.))
    if max_row>1e-5 or int_error>1e-5 or bound_error>1e-5:
        raise AssertionError(f'Invalid solution: rows={max_row}, integer={int_error}, bounds={bound_error}')
    ex=md['expressions'];p=md['parameters'];sc=md['scenarios']
    obj=(sum(s['probability']*pulp.value(ex['scenario_profit'][sid]) for sid,s in sc.items())
         -pulp.value(ex['initial_fleet_cost'])-pulp.value(ex['commitment_cost'])
         -pulp.value(ex['reservation_cost'])-p['cvar']['weight']*pulp.value(ex['cvar']))
    if abs(obj-pulp.value(model.objective))>1e-5:raise AssertionError('Economic objective failed reconciliation')
    return {'rows':max_row,'integrality':int_error,'bound_error':bound_error,'objective_reconciliation':abs(obj-pulp.value(model.objective))}


def initial_audit(pulp, md):
    v=md['variables']; p=md['parameters'];periods=md['periods'];scenarios=md['scenarios'];dest=md['destinations']
    x0=round(pulp.value(v['initial_fleet']))
    if not 0<=x0<=p['initial_fleet_max']:raise AssertionError('Initial fleet bound violated')
    by_s={}
    for sid in scenarios:
        deployed=round(pulp.value(v['deployed_initial'][sid]))
        if not 0<=deployed<=x0:raise AssertionError('Deployed fleet exceeds allocated fleet')
        depart=0;completion=0;prefix=0.;use=0
        for k in periods:
            depart+=sum(pulp.value(v['flights'][sid,d,k]) for d in dest)
            completion+=sum(pulp.value(v['w'][sid,h,k-h]) for h in md['durations'][sid] if (sid,h,k-h) in v['w'] and k-h in periods)
            prefix=max(prefix,depart-deployed-completion)
            # Verify no charged aircraft finishes after horizon by feasible-start construction.
        if abs(depart-deployed-completion)>1e-5 or prefix>1e-5:
            raise AssertionError(f'Inventory mismatch s{sid}: departures={depart},initial={deployed},completions={completion}')
        for h in md['durations'][sid]:
            for t in periods:
                if t+h>p['periods'] and abs(pulp.value(v['w'][sid,h,t]))>1e-6:
                    raise AssertionError(f'Charging completion exceeds horizon s{sid} h{h} t{t}')
        by_s[str(sid)]={'initial_deployed':deployed,'departures':depart,'incoming_completions':completion,'max_prefix_excess':max(prefix,0.)}
    return {'initial_allocated':x0,'scenarios':by_s}


def recover_projected(pulp, core, md, solver, gap, limit):
    """Construct integer incoming charging starts and aircraft/passenger assignments.

    Recovery starts from solved projected integer w, f and y, solves one small
    minimum-cost group/start transportation MILP per scenario and matches
    completed aircraft chronologically to flights. It never requires fractional
    passengers or charging beyond the horizon.
    """
    p=md['parameters'];T=md['periods'];D=md['destinations'];v=md['variables']; reports={}
    for sid,s in md['scenarios'].items():
        g=md['groups'][sid];starts=md['feasible_starts'][sid]
        recover=pulp.LpProblem(f'IntegralChargeRecovery_s{sid}',pulp.LpMinimize)
        z={(gid,t):pulp.LpVariable(f'Recover_g{gid}_t{t}',lowBound=0,cat='Integer')
           for gid in g for t in starts[gid]}
        for gid,group in g.items():
            recover+=pulp.lpSum(z[gid,t] for t in starts[gid])<=group['count']
        for h in md['durations'][sid]:
            for t in T:
                target=round(pulp.value(v['w'][sid,h,t]))
                recover+=pulp.lpSum(z[gid,t] for gid,group in g.items() if group['duration']==h and t in starts[gid])==target
        recover+=pulp.lpSum(z[gid,t]*g[gid]['energy_kwh']*core.energy_price_per_kwh(
                 (p['horizon']['start']+(t-1)*p['bin_size'])/60.,p)
                 for gid,t in z)
        # No incoming aircraft or feasible charge starts: the empty assignment is exact.
        if z:
            charge_status=core.solve_model(recover,solver=solver,time_limit=limit,gap_rel=gap,verbose=False)
            if charge_status['status']!='Optimal':raise AssertionError(f'Charge recovery failed s{sid}')
        original_cost=sum(pulp.value(v['chi'][sid,gid,t])*g[gid]['energy_kwh']*core.energy_price_per_kwh(
                     (p['horizon']['start']+(t-1)*p['bin_size'])/60.,p)
                     for gid in g for t in starts[gid])
        actual_cost=(pulp.value(recover.objective) or 0.) if z else 0.
        if actual_cost>original_cost+1e-4:raise AssertionError('Integral charging recovery increased cost')
        # Matches available aircraft to route/time, respecting period chronology.
        pool=[(1,idx,'initial') for idx in range(round(pulp.value(v['deployed_initial'][sid])))]
        i=0
        for gid,t in z:
            count=round(pulp.value(z[gid,t]))
            for _ in range(count):
                pool.append((t+g[gid]['duration'],i,'incoming'));i+=1
        heapq.heapify(pool)
        flights_by_route={}
        for k in T:
            for d in D:
                flights_by_route[d,k]=[]
                for _ in range(round(pulp.value(v['flights'][sid,d,k]))):
                    if not pool or pool[0][0]>k:raise AssertionError('No recovered aircraft ready by departure period')
                    ready,ident,source=heapq.heappop(pool)
                    flights_by_route[d,k].append({'id':f'{source}{ident}', 'ready':ready,'seats':p['evtol_capacity']})
        if pool:raise AssertionError('Recovered completed aircraft or chosen initial deployment unused')
        passenger_count=0
        for k in T:
            for d in D:
                flights=flights_by_route[d,k];cursor=0
                for r in T:
                    if k not in core._window(r,p['periods'],p['los_periods']):continue
                    for class_name,key in [('B','booked_passengers'),('O','on_demand_passengers')]:
                        remaining=round(pulp.value(v[key][sid,r,d,k]))
                        passenger_count+=remaining
                        while remaining:
                            if cursor>=len(flights):raise AssertionError('Integer passengers exceed recovered seats')
                            taking=min(flights[cursor]['seats'],remaining)
                            if k-r>p['los_periods']:raise AssertionError('Passenger violated service window')
                            flights[cursor]['seats']-=taking;remaining-=taking
                            if flights[cursor]['seats']==0:cursor+=1
        expected=sum(round(pulp.value(v[key][sid,r,d,k])) for key in ('booked_passengers','on_demand_passengers')
                     for r in T for d in D for k in core._window(r,p['periods'],p['los_periods']))
        if passenger_count!=expected:raise AssertionError('Integer passenger disaggregation mismatch')
        reports[str(sid)]={'charging_cost_projected':round(original_cost,6),
                           'charging_cost_recovered':round(actual_cost,6),
                           'departures_recovered':sum(len(x) for x in flights_by_route.values()),
                           'passengers_recovered':passenger_count}
    return reports


def compare_one(pulp,core,ref,inp,args,label):
    pd=core.build_stochastic_model(copy.deepcopy(inp));core.add_constraints_and_objective(pd)
    pres=check_solver(pulp,core,pd,args.solver,args.gap,args.time_limit)
    da=ref.build_detailed_model(copy.deepcopy(inp))
    dres=check_solver(pulp,core,da,args.solver,args.gap,args.time_limit)
    po=pulp.value(pd['model'].objective);de=pulp.value(da['model'].objective)
    tolerance=1e-4+1e-7*max(abs(po),abs(de))
    if abs(po-de)>tolerance:raise AssertionError(f'{label} differs: projected={po}, detailed={de}, diff={po-de}')
    pa=audit(pulp,pd);daudit=audit(pulp,da)
    inv=initial_audit(pulp,pd)
    recovered=recover_projected(pulp,core,pd,args.solver,args.gap,args.time_limit)
    # Fix exact projected stage-one decisions in independent detailed model.
    fixed=ref.build_detailed_model(copy.deepcopy(inp));fv=fixed['variables'];pv=pd['variables']
    fixed['model']+=fv['initial_fleet']==round(pulp.value(pv['initial_fleet'])), 'FixReferenceInitial'
    for d in pd['destinations']:
        for k in pd['periods']:
            fixed['model']+=fv['n'][d,k]==round(pulp.value(pv['n'][d,k])),f'FixRefPlan_{pd["destinations"].index(d)}_{k}'
    for k in pd['periods']:
        fixed['model']+=fv['b'][k]==round(pulp.value(pv['b'][k])),f'FixRefReserve_{k}'
    check_solver(pulp,core,fixed,args.solver,args.gap,args.time_limit)
    if abs(pulp.value(fixed['model'].objective)-po)>tolerance:raise AssertionError('Fixed-first-stage detailed model not equivalent to projected choice')
    return {'case':label,'projected_objective':po,'detailed_objective':de,
            'first_stage_allocated':inv['initial_allocated'],
            'same_objective':True,'same_objective_after_fixing_first_stage':True,
            'projected_audit':pa,'detailed_audit':daudit,'initial_inventory':inv,
            'integer_charging_and_passenger_recovery':recovered,
            'projected_solver_gap':pres.get('solver_reported_gap'),
            'detailed_solver_gap':dres.get('solver_reported_gap')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver',choices=('highs','gurobi','cbc'),default='highs')
    parser.add_argument('--gap',type=float,default=1e-7)
    parser.add_argument('--time-limit',type=int,default=120)
    parser.add_argument('--seeds',type=int,nargs='+',default=[60,61])
    from src.config import DIAGNOSTICS_DIR
    parser.add_argument('--out',type=Path,default=DIAGNOSTICS_DIR / 'initial_fleet_validation.json')
    args=parser.parse_args()
    pulp,core,ref=modules()
    results=[]
    for seed in args.seeds:
        for case,cap,cost in [('no_initial_fleet',0,35.),('cheap_initial_fleet',3,5.),
                              ('costly_initial_fleet',3,2500.),('mixed',3,35.),('late_arrivals',2,35.)]:
            inp=toy_input(seed,cap,cost,'late_arrivals' if case=='late_arrivals' else 'mixed')
            results.append(compare_one(pulp,core,ref,inp,args,f'{case}_seed{seed}'))
            print(f"PASS {case}_seed{seed}: objectives agree; x0={results[-1]['first_stage_allocated']}; recovery passed")
    # Strong structural test: no incoming aircraft and bookings at period 1.
    z=toy_input(60,0,5.,case='early_booking')
    nofleet=compare_one(pulp,core,ref,z,args,'zero_fleet_early_booking')
    early=toy_input(60,1,5.,case='early_booking')
    ready=compare_one(pulp,core,ref,early,args,'ready_fleet_early_booking')
    pno=core.build_stochastic_model(z);core.add_constraints_and_objective(pno)
    check_solver(pulp,core,pno,args.solver,args.gap,args.time_limit)
    pyes=core.build_stochastic_model(early);core.add_constraints_and_objective(pyes)
    check_solver(pulp,core,pyes,args.solver,args.gap,args.time_limit)
    missed0=sum(pulp.value(pno['variables']['unserved_booked'][0,r,'A']) for r in pno['periods'])
    missed1=sum(pulp.value(pyes['variables']['unserved_booked'][0,r,'A']) for r in pyes['periods'])
    if not (missed0>=4-1e-6 and missed1<=1e-6 and round(pulp.value(pyes['variables']['initial_fleet']))==1):
        raise AssertionError(f'Initial-readiness structural test failed: nofleet={missed0}, ready={missed1}')
    results.extend([nofleet,ready])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(f'ALL {len(results)} CASES PASSED. Full details: {args.out}')

if __name__=='__main__':main()
