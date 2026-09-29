"""Small, reproducible comparison of independent detailed / group / projected MILPs.

From repository root, after applying the update:
  python -m src.diagnostics.compare_formulations --solver highs --seeds 60 61
The comparison uses toy instances, not geographical input files. A near-zero
objective difference is necessary but not sufficient: the independent
reconstruction validator is run first on these same instances.
"""
import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import pulp

from src.detailed_reference_model import build_detailed_model
from src.group_flow_model import build_group_flow_model
from src.stochastic_model import (
    build_stochastic_model, add_constraints_and_objective, solve_model,
)
from src.tests.validate_initial_fleet import toy_input, compare_one, modules


def _solve(md, label, args):
    start = time.perf_counter()
    result = solve_model(md['model'], solver=args.solver, gap_rel=args.gap,
                         time_limit=args.time_limit, verbose=False)
    result['seconds'] = time.perf_counter() - start
    if result['status'] != 'Optimal':
        raise AssertionError(f'{label}: solver did not reach Optimal: {result}')
    if (result.get('solver_reported_gap') is None
            or result['solver_reported_gap'] > args.gap + 1e-7):
        raise AssertionError(f'{label}: no certified target MIP gap: {result}')
    return result


def compare_case(model_input, case, args):
    pulp_module, core, ref = modules()
    # The existing independent reference/projection comparison also performs
    # integral charging, aircraft-departure, and passenger-flow reconstruction.
    independent = compare_one(pulp_module, core, ref,
                              copy.deepcopy(model_input), args, case)
    builders = {
        'detailed': lambda: build_detailed_model(copy.deepcopy(model_input)),
        'group_flow': lambda: build_group_flow_model(copy.deepcopy(model_input)),
        'projected': lambda: _projected(copy.deepcopy(model_input)),
    }
    results = {}
    for name, factory in builders.items():
        md = factory()
        res = _solve(md, f'{case} {name}', args)
        mdl = md['model']
        results[name] = {
            'objective': res['objective'], 'seconds': res['seconds'],
            'gap': res['solver_reported_gap'],
            'variables': len(mdl.variables()),
            'integer_variables': sum(x.cat == pulp.LpInteger for x in mdl.variables()),
            'constraints': len(mdl.constraints),
            'x0': round(pulp.value(md['variables']['initial_fleet'])),
        }
    anchor = results['projected']['objective']
    for name, value in results.items():
        if not math.isclose(value['objective'], anchor, rel_tol=1e-7, abs_tol=1e-4):
            raise AssertionError(f'{case}: {name} objective mismatch: {results}')
    return {'case': case, 'formulations': results,
            'independent_reconstruction': independent}


def _projected(inp):
    md = build_stochastic_model(inp)
    add_constraints_and_objective(md)
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solver', choices=('highs','gurobi','cbc'), default='highs')
    ap.add_argument('--seeds', type=int, nargs='+', default=[60, 61])
    ap.add_argument('--gap', type=float, default=1e-7)
    ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--out', type=Path, default=Path('outputs/diagnostics/three_formulations.json'))
    args = ap.parse_args()
    reports = []
    for seed in args.seeds:
        for case in ('mixed', 'shared_duration'):
            inp = toy_input(seed, cap=3, fleet_cost=35.)
            # Replace original zero-duration aircraft with positive-duration
            # arrivals so all comparisons enforce manuscript assumption A4.
            for s in inp['scenarios'].values():
                for a in s['aircraft'].values():
                    if a['initial_soc'] >= inp['parameters']['soc_min']:
                        a['initial_soc'] = 40.0
            if case == 'shared_duration':
                # Different group arrivals/costs sharing one-period charge:
                # 50% and 60% both need one 15-min period.
                inp['scenarios'][0]['aircraft'][14] = {
                    'arrival_period':1, 'initial_soc':50.0}
                inp['scenarios'][0]['aircraft'][15] = {
                    'arrival_period':2, 'initial_soc':60.0}
            inp['parameters']['require_positive_incoming_duration'] = True
            reports.append(compare_case(inp, f'{case}_seed{seed}', args))
            print(f'PASS {case} seed={seed}: three objectives match; '
                  'projected charging reconstruction passed', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print('Saved', args.out)

if __name__ == '__main__':
    main()
