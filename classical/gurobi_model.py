from gurobipy import *
import numpy as np
import pandas as pd
from datetime import datetime
import time

def create_gurobi_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    model = Model("Linear_Scheduling")
    model.setParam("OutputFlag", 0)  # Suppress solver output

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5, 'memory': 3, 'extent': 2}

    # Load physician data (needed for extent penalty)
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')

    # Precompute how many days have passed for each shift (for extent priority)
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {}
    for s in shifts:
        shift_date = datetime.strptime(s.split(' ')[0], "%Y-%m-%d")
        shift_days_passed[s] = (shift_date - start_date).days

    # Create decision variables only for available assignments
    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:  # Only create variables if available
                x[p, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{s}")
    model.update()  # Must update model after adding variables

    # Initialize objective terms
    preference_expr = LinExpr()
    fairness_expr = LinExpr()
    memory_expr = LinExpr()
    extent_expr = LinExpr()

    if cl >= 1:
        # Demand satisfaction constraints
        for s in shifts:
            assigned_vars = [x[p, s] for p in physicians if (p, s) in x]
            model.addConstr(quicksum(assigned_vars) >= demand[s], name=f"demand_{s}")

        # Fairness deviation constraints
        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))

        u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")
        for p in physicians:
            assigned_vars = [x[p, s] for s in shifts if (p, s) in x]
            total_assigned = quicksum(assigned_vars)
            model.addConstr(total_assigned - avg_assignments <= u[p], name=f"fair_upper_{p}")
            model.addConstr(avg_assignments - total_assigned <= u[p], name=f"fair_lower_{p}")
        fairness_expr = quicksum(u[p] for p in physicians)

    if cl >= 2:
        # Preference penalties
        for (p, s), var in x.items():
            val = preference[p][s]
            if val == 1:
                preference_expr += -var
            elif val == -1:
                preference_expr += var

        # Memory penalty (dynamic workload balancing)
        for p in physicians:
            assigned_vars_p = [x[p, s] for s in shifts if (p, s) in x]
            memory_expr += quicksum(assigned_vars_p)

        # Extent penalty (long-term workload balancing)
        for idx_p, p in enumerate(physicians):
            work_rate_p = physician_df['work rate'].iloc[idx_p]
            for s in shifts:
                if (p, s) in x:
                    days_passed = shift_days_passed[s]
                    extent_priority = min(days_passed / 7, 1)
                    priority_p = abs(extent_priority * (1 - float(work_rate_p)))

                    if work_rate_p < 1:
                        extent_expr += -priority_p * x[p, s]
                    else:
                        extent_expr += priority_p * x[p, s]

    # Set full objective
    model.setObjective(
        lambdas['pref'] * preference_expr +
        lambdas['fair'] * fairness_expr +
        lambdas['memory'] * memory_expr +
        lambdas['extent'] * extent_expr,
        GRB.MINIMIZE
    )

    # Solve the model
    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    # Extract schedule if optimal
    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for (p, s), var in x.items():
            if var.X > 0.5:
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
