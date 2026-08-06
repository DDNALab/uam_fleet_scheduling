"""
Solution algorithm.py
=====================
Stochastic eVTOL scheduling MILP using PuLP + HiGHS.

Implements:
- charging assignment
- passenger allocation
- capacity constraints
- level-of-service constraints
- CVaR-based risk-averse objective
"""

import os

import pulp


def _select_solver(verbose=False):
    """Select a portable MILP solver, honoring ``HIGHS_PATH`` when set."""
    highs_path = os.getenv("HIGHS_PATH")
    if highs_path:
        return pulp.HiGHS_CMD(path=highs_path, msg=verbose)

    if hasattr(pulp, "HiGHS"):
        highs = pulp.HiGHS(msg=verbose)
        if highs.available():
            return highs

    cbc = pulp.PULP_CBC_CMD(msg=verbose)
    if cbc.available():
        return cbc

    raise RuntimeError(
        "No MILP solver is available. Install highspy or set HIGHS_PATH to "
        "a HiGHS executable."
    )


# ============================================================
# Main Solver
# ============================================================
def build_and_solve_stochastic_model_aggregated(
        stochastic_input,
        M=1000,
        verbose=True,
        alpha=0.90,
        risk_weight=1.0):

    """
    Solve stochastic eVTOL scheduling problem with CVaR objective.
    """

    # ============================================================
    # 0. Extract global structure
    # ============================================================
    gp = stochastic_input["global_parameters"]

    m_facilities = gp["m_facilities"]
    l_periods = gp["l_periods"]
    eVTOL_capacity = gp["eVTOL_capacity"]
    LoS = gp["LOS"]
    destination_fares = gp["destination_fares"]

    I = list(range(m_facilities))
    Time = list(range(1, l_periods + 1))
    S = list(stochastic_input["scenarios"].keys())
    D = list(destination_fares.keys())

    # ============================================================
    # 1. Model initialization
    # ============================================================
    model = pulp.LpProblem("Stochastic_eVTOL_Scheduling", pulp.LpMaximize)

    # ============================================================
    # 2. Decision variables
    # ============================================================
    x, C, F, z = {}, {}, {}, {}
    pass_assigned, unserved_pass, Y, Tvar = {}, {}, {}, {}
    use_evtol = {}

    for s in S:

        sc = stochastic_input["scenarios"][s]
        J = list(range(sc["n_evtols"]))

        for i in I:
            for j in J:
                for t in Time:
                    x[i, j, t, s] = pulp.LpVariable(f"x_{i}_{j}_{t}_{s}", cat="Binary")        

        for j in J:
            use_evtol[j, s] = pulp.LpVariable(f"use_{j}_{s}", cat="Binary")

            C[j, s] = pulp.LpVariable(f"C_{j}_{s}", lowBound=0, cat="Integer")
            F[j, s] = pulp.LpVariable(f"F_{j}_{s}", lowBound=0, cat="Integer")

            for d in D:
                z[j, d, s] = pulp.LpVariable(f"z_{j}_{d}_{s}", cat="Binary")

            for t in Time:
                for d in D:
                    pass_assigned[j, t, d, s] = pulp.LpVariable(
                        f"p_{j}_{t}_{d}_{s}", lowBound=0, cat="Integer"
                    )
                    Y[j, t, d, s] = pulp.LpVariable(f"Y_{j}_{t}_{d}_{s}", cat="Binary")
                    Tvar[j, t, d, s] = pulp.LpVariable(
                        f"T_{j}_{t}_{d}_{s}", lowBound=0, cat="Continuous"
                    )

        for t in Time:
            for d in D:
                unserved_pass[t, d, s] = pulp.LpVariable(
                    f"u_{t}_{d}_{s}", lowBound=0, cat="Integer"
                )

    # ============================================================
    # 3. CVaR variables
    # ============================================================
    zeta = pulp.LpVariable("zeta", cat="Continuous")
    rho, U = {}, {}

    for s in S:
        rho[s] = pulp.LpVariable(f"rho_{s}", lowBound=0, cat="Continuous")
        U[s] = pulp.LpVariable(f"U_{s}", lowBound=0, cat="Continuous")

    # ============================================================
    # 4. Constraints
    # ============================================================
    for s in S:

        sc = stochastic_input["scenarios"][s]
        J = list(range(sc["n_evtols"]))

        r = {j: sc["evtols"][j]["arrival_bin"] for j in J}
        p = {j: sc["evtols"][j]["charging_periods"] for j in J}
        agg = sc["agg_pass_demand"]

        # -----------------------------
        # 4.1 Charging assignment
        # -----------------------------
        for j in J:

            start, end = r[j], l_periods - p[j] + 1

            if start <= end:
                model += (
                    pulp.lpSum(
                        x[i, j, t, s]
                        for i in I for t in range(start, end + 1)
                    ) == use_evtol[j, s],
                    f"charge_{j}_{s}"
                )
            else:
                model += (use_evtol[j, s] == 0)

                for i in I:
                    for t in Time:
                        x[i, j, t, s] = 0

        # -----------------------------
        # 4.2 Facility capacity
        # -----------------------------
        for i in I:
            for t in Time:
                model += (
                    pulp.lpSum(
                        x[i, j, u, s]
                        for j in J
                        for u in range(max(1, t - p[j] + 1), t + 1)
                        if u >= r[j]
                    ) <= 1
                )

        # -----------------------------
        # 4.3 Completion logic
        # -----------------------------
        for j in J:
            model += (
                C[j, s] == pulp.lpSum(
                    (t + p[j] - 1) * x[i, j, t, s]
                    for i in I for t in Time
                )
            )

            model += (F[j, s] >= C[j, s])

        # -----------------------------
        # 4.4 Passenger constraints
        # -----------------------------
        for j in J:

            model += (
                pulp.lpSum(z[j, d, s] for d in D) <= use_evtol[j, s]
            )

            model += (
                pulp.lpSum(
                    pass_assigned[j, t, d, s]
                    for t in Time for d in D
                ) <= eVTOL_capacity * use_evtol[j, s]
            )

        # -----------------------------
        # 4.5 Demand balance
        # -----------------------------
        for t in Time:
            for d in D:
                model += (
                    pulp.lpSum(pass_assigned[j, t, d, s] for j in J)
                    + unserved_pass[t, d, s]
                    == agg[t, d]
                )

        # -----------------------------
        # 4.6 CVaR structure
        # -----------------------------
        model += (
            U[s] == pulp.lpSum(unserved_pass[t, d, s] for t in Time for d in D)
        )

        model += (rho[s] >= U[s] - zeta)

        # -----------------------------
        # 4.7 Level of service
        # -----------------------------
        for j in J:
            for t in Time:
                for d in D:
                    model += (
                        Tvar[j, t, d, s]
                        >= F[j, s] - t - M * (1 - Y[j, t, d, s])
                    )

                    model += (
                        Tvar[j, t, d, s]
                        <= LoS + M * (1 - Y[j, t, d, s])
                    )

    # ============================================================
    # 5. Objective function
    # ============================================================
    revenue = pulp.lpSum(
        stochastic_input["scenarios"][s]["probability"]
        * pass_assigned[j, t, d, s] * destination_fares[d]
        for s in S
        for j in range(stochastic_input["scenarios"][s]["n_evtols"])
        for t in Time
        for d in D
    )

    unserved_penalty = pulp.lpSum(
        stochastic_input["scenarios"][s]["probability"]
        * unserved_pass[t, d, s] * destination_fares[d]
        for s in S for t in Time for d in D
    )

    charge_cost = pulp.lpSum(
        stochastic_input["scenarios"][s]["probability"]
        * stochastic_input["scenarios"][s]["evtols"][j]["charging_cost"]
        * x[i, j, t, s]
        for s in S
        for j in range(stochastic_input["scenarios"][s]["n_evtols"])
        for i in I for t in Time
    )

    expected_profit = revenue - unserved_penalty - charge_cost

    cvar = zeta + (1 / (1 - alpha)) * pulp.lpSum(
        stochastic_input["scenarios"][s]["probability"] * rho[s]
        for s in S
    )

    model += expected_profit - risk_weight * cvar

    # ============================================================
    # 6. Solve
    # ============================================================
    solver = _select_solver(verbose=verbose)
    model.solve(solver)

    # =====================================================
    # 7. OUTPUT / REPORTING (PUT HERE)
    # =====================================================

    status = pulp.LpStatus[model.status]
    obj_value = pulp.value(model.objective)

    if model.status == pulp.LpStatusOptimal:

        print("\nOptimal solution found.")
        print(f"   Objective Value: ${obj_value:,.2f}")

        print("\n====================================================")
        print(" OPTIMAL SOLUTION SUMMARY")
        print("====================================================")

        zeta_val = pulp.value(zeta)

        cvar_val = zeta_val + (1.0 / (1.0 - alpha)) * sum(
            stochastic_input["scenarios"][s]["probability"] * pulp.value(rho[s])
            for s in S
        )

        print(f"\n CVaR_{alpha:.2f} (Total Unserved) = {cvar_val:.2f}")
        print(f" VaR Threshold (zeta)             = {zeta_val:.2f}")
        print(f" Risk Weight (lambda)             = {risk_weight:.2f}")

        print("----------------------------------------------------")
        print("\n DETAILED RESULTS PER SCENARIO\n")

        for s in S:

            scenario = stochastic_input["scenarios"][s]
            prob = scenario["probability"]
            J = list(range(scenario["n_evtols"]))

            served = sum(
                pulp.value(pass_assigned[j, t, d, s])
                for j in J for t in Time for d in D
            )

            unserved = sum(
                pulp.value(unserved_pass[t, d, s])
                for t in Time for d in D
            )

            revenue = sum(
                pulp.value(pass_assigned[j, t, d, s]) * destination_fares[d]
                for j in J for t in Time for d in D
            )

            lost = sum(
                pulp.value(unserved_pass[t, d, s]) * destination_fares[d]
                for t in Time for d in D
            )

            charge = sum(
                scenario["evtols"][j]["charging_cost"]
                * sum(pulp.value(x[i, j, t, s]) for i in I for t in Time)
                for j in J
            )

            net = revenue - lost - charge

            print(
                f" Scenario {s} (p={prob:.2f}) → "
                f"Served={served:.0f}, "
                f"Unserved={unserved:.0f}, "
                f"U_s={pulp.value(U[s]):.2f}, "
                f"rho_s={pulp.value(rho[s]):.2f}, "
                f"Net=${net:,.2f}"
            )

        print("====================================================")

    # return at end (keep this)
    return model, {
        "status": status,
        "status_code": model.status,
        "objective": obj_value,
    }
