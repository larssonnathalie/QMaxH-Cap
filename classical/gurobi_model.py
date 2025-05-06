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
        lambdas = {'pref': 100, 'fair': 2, 'memory': 3, 'extent': 2}

    model = Model("Optimized_Scheduling")
    model.setParam("OutputFlag", 0)

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    extent_map = {row["name"]: row["extent"] for _, row in physician_df.iterrows()}

    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    total_shifts = len(shifts)
    shift_dates = {s: s.split(" ")[0] for s in shifts}

    # === Constraint 1: Demand ===
    for s in shifts:
        model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

    # === Constraint 2: Unavailability ===
    for p in physicians:
        for s in shifts:
            if preference[p][s] == -2:
                model.addConstr(x[p, s] == 0, name=f"unavail_{p}_{s}")

    # === Constraint 3: One shift per day ===
    shifts_by_date = defaultdict(list)
    for s in shifts:
        shifts_by_date[shift_dates[s]].append(s)

    for p in physicians:
        for date, shift_list in shifts_by_date.items():
            model.addConstr(quicksum(x[p, s] for s in shift_list) <= 1, name=f"one_shift_{p}_{date}")

    # === Penalty terms ===
    preference_expr = LinExpr()
    fairness_expr = LinExpr()
    memory_expr = LinExpr()
    extent_expr = LinExpr()

    # === Preference penalty ===
    for p in physicians:
        for s in shifts:
            val = preference[p][s]
            if val == 1:
                preference_expr += -x[p, s]  # reward preferred
            elif val == -1:
                preference_expr += x[p, s]   # penalize unpreferred

    # === Extent & Fairness ===
    extent_groups = defaultdict(list)
    assigned = {}
    for p in physicians:
        extent_groups[extent_map[p]].append(p)
        assigned[p] = model.addVar(vtype=GRB.INTEGER, name=f"assigned_{p}")
        model.addConstr(assigned[p] == quicksum(x[p, s] for s in shifts), name=f"assigned_sum_{p}")

    for extent, group in extent_groups.items():
        group_target = total_shifts * (extent / 100) / len(group)
        target = int(np.round(group_target))

        for p in group:
            dev = model.addVar(vtype=GRB.INTEGER, name=f"deviation_{p}")
            model.addConstr(dev >= assigned[p] - target, name=f"extent_high_{p}")
            model.addConstr(dev >= target - assigned[p], name=f"extent_low_{p}")
            extent_expr += dev

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                p1, p2 = group[i], group[j]
                diff = model.addVar(vtype=GRB.INTEGER, name=f"diff_{p1}_{p2}")
                model.addConstr(diff >= assigned[p1] - assigned[p2])
                model.addConstr(diff >= assigned[p2] - assigned[p1])
                fairness_expr += diff

    # === Memory penalty across days ===
    for p in physicians:
        for i in range(1, len(shifts)):
            s_prev, s_curr = shifts[i - 1], shifts[i]
            d_prev, d_curr = shift_dates[s_prev], shift_dates[s_curr]
            if d_prev != d_curr:
                memory_expr += x[p, s_prev] * x[p, s_curr]

    # === Objective ===
    total_cost = (
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

    print(f"Gurobi solver time: {solver_time:.4f} seconds")
    print(f"Gurobi overall time: {overall_time:.4f} seconds")
    print(f"Gurobi memory used: {memory_used:.2f} MB")

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for p in physicians:
            for s in shifts:
                if x[p, s].X > 0.5:
                    schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
