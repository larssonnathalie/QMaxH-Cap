from gurobipy import *
import numpy as np
import pandas as pd
from datetime import datetime
import time

def create_gurobi_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    model = Model("Linear_Scheduling")
    model.setParam("OutputFlag", 0)  # Suppress output for cleaner runs

    # Default lambda values if none provided
    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5, 'memory': 3, 'extent': 2}

    # Load physician data to get work rates
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')

    # Precompute how many days have passed for each shift (needed for extent priority)
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {}
    for s in shifts:
        day_str = s.split(' ')[0]
        shift_date = datetime.strptime(day_str, "%Y-%m-%d")
        days_passed = (shift_date - start_date).days
        shift_days_passed[s] = days_passed

    # Create decision variables: x[p,s] ∈ {0,1}
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # Initialize objective function terms
    preference_expr = LinExpr()
    fairness_expr = LinExpr()
    memory_expr = LinExpr()
    extent_expr = LinExpr()

    # === Constraints and penalties depending on complexity level ===
    if cl >= 1:
        # Demand satisfaction: each shift must meet required demand
        for s in shifts:
            model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

        # Fairness penalty: deviation from average number of assigned shifts
        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))

        u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")  # Auxiliary variables for fairness deviation
        for p in physicians:
            assigned = quicksum(x[p, s] for s in shifts)
            model.addConstr(assigned - avg_assignments <= u[p], name=f"fair_upper_{p}")
            model.addConstr(avg_assignments - assigned <= u[p], name=f"fair_lower_{p}")
        fairness_expr = quicksum(u[p] for p in physicians)  # Will later multiply by lambda['fair']

    if cl >= 2:
        # Preference penalty: reward preferred shifts and penalize non-preferred shifts
        for p in physicians:
            for s in shifts:
                val = preference[p][s]
                if val == 1:
                    preference_expr += -x[p, s]
                elif val == -1:
                    preference_expr += x[p, s]

        # Memory penalty: penalize physicians with more cumulative assignments
        for p in physicians:
            assigned_vars_p = [x[p, s] for s in shifts]
            if assigned_vars_p:
                total_assigned_p = quicksum(assigned_vars_p)
                memory_expr += total_assigned_p

        # Extent penalty: favor physicians based on their target work rates
        for idx_p, p in enumerate(physicians):
            work_rate_p = physician_df['work rate'].iloc[idx_p]
            for s in shifts:
                days_passed = shift_days_passed[s]
                extent_priority = min(days_passed / 7, 1)  # Extent importance grows with time
                priority_p = abs(extent_priority * (1 - float(work_rate_p)))

                if work_rate_p < 1:
                    extent_expr += -priority_p * x[p, s]
                else:
                    extent_expr += priority_p * x[p, s]

    # === Define the full objective function ===
    model.setObjective(
        lambdas['pref'] * preference_expr +
        lambdas['fair'] * fairness_expr +
        lambdas['memory'] * memory_expr +
        lambdas['extent'] * extent_expr,
        GRB.MINIMIZE
    )

    # === Solve the model ===
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