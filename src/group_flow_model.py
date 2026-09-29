"""Integer group-flow comparator for the hybrid UAM paper.

Uses exactly the projected model's input schema, booking protection, passenger
flows, costs, and CVaR constraints, while keeping integer group-specific
charging starts and group-specific ready inventory. The destination dimension
is projected out of group departures; destination totals remain common.

IMPORTANT: this module expects the current repository src/stochastic_model.py.
It does not alter src/old_formulation_code or geographic input files.
"""

from collections import defaultdict
import pulp

from .stochastic_model import build_stochastic_model, add_constraints_and_objective


def build_group_flow_model(model_input):
    """Build an exact intermediate group-flow MILP for the single-leg model.

    The baseline projected builder creates continuous chi[g,t] and integer
    w[h,t] variables. Replace each w reference *before* adding equations by
    its group-start sum, and make chi integer. Then enforce integer group
    ready inventories and per-group departures, without group x destination
    assignment. Shared bookings, flights, seats, and all costs are unchanged.
    """
    md = build_stochastic_model(model_input)
    v = md["variables"]
    groups, feasible = md["groups"], md["feasible_starts"]
    for var in v["chi"].values():
        var.cat = pulp.LpInteger

    margins = {}
    for sid in md["scenarios"]:
        for h in md["durations"][sid]:
            for t in md["periods"]:
                margins[sid, h, t] = pulp.lpSum(
                    v["chi"][sid, gid, t]
                    for gid, g in groups[sid].items()
                    if g["duration"] == h and t in feasible[sid][gid]
                )
    # The original w variables cease to appear in the mathematical model.
    # Their incidence equations become tautological and are removed below.
    original_w = v["w"]
    v["w"] = margins
    model = add_constraints_and_objective(md)
    for name in list(model.constraints):
        if name.startswith("DurationStartMargin_"):
            del model.constraints[name]

    initial_departures = {}
    group_departures = {}
    group_ready = {}
    for sid in md["scenarios"]:
        for k in md["periods"]:
            initial_departures[sid, k] = pulp.LpVariable(
                f"GF_InitialDepart_s{sid}_k{k}", lowBound=0, cat="Integer")
        model += (pulp.lpSum(initial_departures[sid, k] for k in md["periods"])
                  == v["deployed_initial"][sid]), f"GF_InitialBalance_s{sid}"
        for gid, g in groups[sid].items():
            h = g["duration"]
            for k in md["periods"]:
                group_departures[sid, gid, k] = pulp.LpVariable(
                    f"GF_Depart_s{sid}_g{gid}_k{k}", lowBound=0, cat="Integer")
                group_ready[sid, gid, k] = pulp.LpVariable(
                    f"GF_Ready_s{sid}_g{gid}_k{k}", lowBound=0, cat="Integer")
                previous = (group_ready[sid, gid, k-1] if k > 1 else 0)
                completions = pulp.lpSum(
                    v["chi"][sid, gid, t] for t in feasible[sid][gid]
                    if t + h == k)
                model += (
                    group_ready[sid, gid, k]
                    == previous + completions - group_departures[sid, gid, k]
                ), f"GF_GroupReady_s{sid}_g{gid}_k{k}"
            model += group_ready[sid, gid, md["periods"][-1]] == 0, (
                f"GF_NoChargedAircraftLeft_s{sid}_g{gid}")
        for k in md["periods"]:
            model += (
                initial_departures[sid, k]
                + pulp.lpSum(group_departures[sid, gid, k] for gid in groups[sid])
                == pulp.lpSum(v["flights"][sid, d, k] for d in md["destinations"])
            ), f"GF_SourceDepartureBalance_s{sid}_k{k}"

    v["group_departures"] = group_departures
    v["group_ready"] = group_ready
    v["initial_departures"] = initial_departures
    md["formulation"] = "group-flow"
    md["unused_projected_w"] = original_w  # only for inspection; not in MILP
    return md
