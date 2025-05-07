from gurobipy import *
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import psutil
import os

def create_gurobi_model(physicians, shifts, demand, preference, lambdas=None):
    overall_start = time.time()
    process = psutil.Process(os.getpid())

    if lambdas is None:
        lambdas = {'demand': 3, 'fair': 10, 'pref': 5, 'unavail': 10, 'extent': 8, 'rest': 0, 'titles': 5, 'memory': 3}

    # Load physician data
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    work_rate = dict(zip(physician_df['name'], physician_df['work rate']))

    # Precompute days passed per shift
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {
        s: (datetime.strptime(s.split(' ')[0], "%Y-%m-%d") - start_date).days
        for s in shifts
    }

    # Create decision variables only for available (not -2) assignments
    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:
                x[p, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{s}")

    # Demand constraints
    for s in shifts:
        model.addConstr(
            quicksum(x[p, s] for p in physicians if (p, s) in x) == demand[s],
            name=f"demand_{s}"
        )

    # Fairness constraints and penalties
    total_demand = sum(demand.values())
    avg_assignments = int(np.ceil(total_demand / len(physicians)))

    u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")
    for p in physicians:
        assigned_vars = [x[p, s] for s in shifts if (p, s) in x]
        total_assigned = quicksum(assigned_vars)
        model.addConstr(total_assigned - avg_assignments <= u[p])
        model.addConstr(avg_assignments - total_assigned <= u[p])
    fairness_expr = quicksum(u[p] for p in physicians)

    # Preference penalties
    preference_expr = quicksum(
        -x[p, s] if preference[p][s] == 1 else x[p, s]
        for (p, s) in x if preference[p][s] in [1, -1]
    )

    # Memory penalty
    memory_expr = quicksum(x[p, s] for (p, s) in x)

    # Extent penalty (based on work rate over time)
    extent_expr = 0
    for (p, s) in x:
        days_passed = shift_days_passed[s]
        extent_priority = min(days_passed / 7, 1)
        rate_diff = 1 - float(work_rate[p])
        weight = abs(extent_priority * rate_diff)
        if work_rate[p] < 1:
            extent_expr += -weight * x[p, s]
        else:
            extent_expr += weight * x[p, s]

    # Objective
    model.setObjective(
        lambdas['pref'] * preference_expr +
        lambdas['fair'] * fairness_expr +
        lambdas['memory'] * memory_expr +
        lambdas['extent'] * extent_expr
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # === Solve ===
    mem_before = process.memory_info().rss / 1024 / 1024
    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_used = mem_after - mem_before
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    # Extract result
    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for (p, s), var in x.items():
            if var.X > 0.5:
                schedule[p].append(s)
        print(f"Gurobi solver time: {solver_time:.4f} seconds")
        print(f"Gurobi overall time: {overall_time:.4f} seconds")
        return schedule, solver_time, overall_time
    else:
        print("Model status is not optimal.")
        return None, solver_time, overall_time
