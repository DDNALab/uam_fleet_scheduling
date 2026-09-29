"""Quick independent validation of all three revised formulation levels.

Runs without geo data. Uses existing src.tests.validate_initial_fleet tests
PLUS the new group-flow comparator on a strict-positive-duration toy case.
"""
import argparse
import copy
import math
import pulp

from src.group_flow_model import build_group_flow_model
from src.tests.validate_initial_fleet import toy_input, compare_one, modules
from src.stochastic_model import build_stochastic_model, add_constraints_and_objective, solve_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', default='highs', choices=['highs','gurobi','cbc'])
    parser.add_argument('--gap', type=float, default=1e-7)
    parser.add_argument('--time-limit', type=float, default=120)
    args=parser.parse_args()
    pulp_module, core, reference = modules()
    inp=toy_input(seed=60, cap=3)
    for s in inp['scenarios'].values():
        for a in s['aircraft'].values():
            if a['initial_soc'] >= inp['parameters']['soc_min']:
                a['initial_soc']=40.0
    inp['scenarios'][0]['aircraft'][14]={'arrival_period':1,'initial_soc':50.0}
    inp['scenarios'][0]['aircraft'][15]={'arrival_period':2,'initial_soc':60.0}
    inp['parameters']['require_positive_incoming_duration']=True
    report=compare_one(pulp_module, core, reference, copy.deepcopy(inp),args,
                       'strict_positive_shared_duration')
    gf=build_group_flow_model(copy.deepcopy(inp))
    result=solve_model(gf['model'],solver=args.solver, gap_rel=args.gap,
                       time_limit=args.time_limit, verbose=False)
    assert result['status']=='Optimal',result
    assert result.get('solver_reported_gap') is not None and result['solver_reported_gap']<=args.gap+1e-7,result
    assert math.isclose(result['objective'], report['projected_objective'],abs_tol=1e-4,rel_tol=1e-7)
    integer_chi=all(v.cat==pulp.LpInteger for v in gf['variables']['chi'].values())
    assert integer_chi
    print('PASS: 3 formulations; objective, aircraft reconstruction, strict A4, integer group starts')

if __name__=='__main__':
    main()
