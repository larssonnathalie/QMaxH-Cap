from gurobipy import *
import numpy as np
import pandas as pd
from datetime import datetime
import time

def create_gurobi_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    model = Model("Linear_Scheduling")
    model.setParam("OutputFlag", 0)
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5, 'memory': 3, 'extent': 2}

    preference_expr = LinExpr()
    fairness_expr = LinExpr()
    memory_expr = LinExpr()
    extent_expr = LinExpr()

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')

    # Compute days passed per shift
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {}
    for s in shifts:
        day_str = s.split(' ')[0]
        shift_date = datetime.strptime(day_str, "%Y-%m-%d")
        days_passed = (shift_date - start_date).days
        shift_days_passed[s] = days_passed

    if cl >= 1:
        for s in shifts:
            model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))
        u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")
        for p in physicians:
            assigned = quicksum(x[p, s] for s in shifts)
            model.addConstr(assigned - avg_assignments <= u[p], name=f"fair_upper_{p}")
            model.addConstr(avg_assignments - assigned <= u[p], name=f"fair_lower_{p}")
        fairness_expr = lambdas['fair'] * quicksum(u[p] for p in physicians)

    if cl >= 2:
        for p in physicians:
            for s in shifts:
                val = preference[p][s]
                if val == 1:
                    preference_expr += -lambdas['pref'] * x[p, s]
                elif val == -1:
                    preference_expr += lambdas['pref'] * x[p, s]
                elif val == -2:
                    model.addConstr(x[p, s] == 0, name=f"unavail_{p}_{s}")

        shift_days = [s.split(' ')[0] for s in shifts]
        for i in range(1, len(shifts)):
            if shift_days[i] == shift_days[i-1]:
                for p in physicians:
                    memory_expr += x[p, shifts[i-1]] * x[p, shifts[i]]

        for idx_p, p in enumerate(physicians):
            work_rate_p = physician_df['work rate'].iloc[idx_p]
            for s in shifts:
                days_passed = shift_days_passed[s]
                extent_priority = min(days_passed / 7, 1)
                priority_p = abs(extent_priority * (1 - float(work_rate_p)))

                if work_rate_p < 1:
                    extent_expr += -priority_p * x[p, s]
                else:
                    extent_expr += priority_p * x[p, s]

    model.setObjective(preference_expr + fairness_expr + lambdas['memory'] * memory_expr + lambdas['extent'] * extent_expr, GRB.MINIMIZE)

    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    print(f"Gurobi solver time: {solver_time:.4f} seconds")
    print(f"Gurobi overall time: {overall_time:.4f} seconds")

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for p in physicians:
            for s in shifts:
                if x[p, s].X > 0.5:
                    schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
