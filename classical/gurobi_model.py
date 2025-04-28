from gurobipy import *
import numpy as np
import time

def create_gurobi_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    model = Model("Linear_Scheduling")
    model.setParam("OutputFlag", 0)  # Suppress solver output

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5, 'memory': 3, 'extent': 2}

    # Decision variables: x[p, s] ∈ {0,1}
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # Initialize expression terms
    preference_expr = LinExpr()
    fairness_expr = LinExpr()
    memory_expr = LinExpr()

    if cl >= 1:
        # Demand satisfaction constraints
        for s in shifts:
            model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

        # Fairness penalties (deviation from average assignments)
        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))
        u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")

        for p in physicians:
            assigned = quicksum(x[p, s] for s in shifts)
            model.addConstr(assigned - avg_assignments <= u[p], name=f"fair_upper_{p}")
            model.addConstr(avg_assignments - assigned <= u[p], name=f"fair_lower_{p}")
        fairness_expr = lambdas['fair'] * quicksum(u[p] for p in physicians)

    if cl >= 2:
        # Preference penalties
        for p in physicians:
            for s in shifts:
                val = preference[p][s]
                if val == 1:
                    preference_expr += -x[p, s]
                elif val == -1:
                    preference_expr += x[p, s]

        # === Corrected MEMORY penalty ===
        # Penalize total number of shifts assigned to physicians
        for p in physicians:
            assigned_vars_p = [x[p, s] for s in shifts]
            if assigned_vars_p:
                total_assigned_p = quicksum(assigned_vars_p)
                memory_expr += total_assigned_p

    # Set the objective function
    model.setObjective(
        lambdas['pref'] * preference_expr + fairness_expr + lambdas['memory'] * memory_expr,
        GRB.MINIMIZE
    )

    # Solve the model
    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for p in physicians:
            for s in shifts:
                if x[p, s].X > 0.5:
                    schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
