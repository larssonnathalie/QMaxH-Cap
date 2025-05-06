from z3 import *
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import psutil
import os

def create_z3_model(physicians, shifts, demand, preference, lambdas=None):
    overall_start = time.time()
    process = psutil.Process(os.getpid())

    solver = Optimize()
    set_param("parallel.enable", True)

    # Decision variables
    x = {(p, s): Int(f"x_{p}_{s}") for p in physicians for s in shifts}
    for var in x.values():
        solver.add(var >= 0, var <= 1)

    # Load physician data
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    extent_map = {row["name"]: row["extent"] for _, row in physician_df.iterrows()}
    total_shifts = len(shifts)

    # Shift-date mapping
    shift_dates = {s: s.split(' ')[0] for s in shifts}
    shifts_by_date = defaultdict(list)
    for s in shifts:
        shifts_by_date[shift_dates[s]].append(s)
    dates = sorted(shifts_by_date.keys())

    # === Hard Constraints ===

    # Demand satisfaction
    for s in shifts:
        solver.add(Sum(x[p, s] for p in physicians) == demand[s])

    # Unavailability
    for (p, s), var in x.items():
        if preference[p][s] == -2:
            solver.add(var == 0)

    # One shift per day
    for p in physicians:
        for date in dates:
            solver.add(Sum(x[p, s] for s in shifts_by_date[date]) <= 1)

    # === Penalty Terms ===
    fairness_terms = []
    extent_terms = []
    memory_terms = []
    preference_terms = []

    assigned = {p: Sum([x[(p, s)] for s in shifts]) for p in physicians}

    extent_groups = defaultdict(list)
    for p in physicians:
        extent_groups[extent_map[p]].append(p)

    for extent, group in extent_groups.items():
        group_target = total_shifts * (extent / 100) / len(group)
        target = int(round(group_target))

        for p in group:
            extent_terms.append(Abs(assigned[p] - target))

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pi, pj = group[i], group[j]
                fairness_terms.append(Abs(assigned[pi] - assigned[pj]))

    # Preferences
    for (p, s), var in x.items():
        val = preference[p][s]
        if val == 1:
            preference_terms.append(-var)
        elif val == -1:
            preference_terms.append(var)

    # Memory penalty (across day boundaries)
    for p in physicians:
        for d1, d2 in zip(dates[:-1], dates[1:]):
            for s1 in shifts_by_date[d1]:
                for s2 in shifts_by_date[d2]:
                    memory_terms.append(x[p, s1] * x[p, s2])

    if lambdas is None:
        lambdas = {'pref': 100, 'fair': 2, 'memory': 3, 'extent': 2}

    total_cost = (
        lambdas['pref'] * Sum(preference_terms) +
        lambdas['fair'] * Sum(fairness_terms) +
        lambdas['memory'] * Sum(memory_terms) +
        lambdas['extent'] * Sum(extent_terms)
    )
    solver.minimize(total_cost)

    # === Solve ===
    mem_before = process.memory_info().rss / 1024 / 1024
    solve_start = time.time()
    result = solver.check()
    solve_end = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_used = mem_after - mem_before
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    print(f"Z3 solver time: {solver_time:.4f} seconds")
    print(f"Z3 overall time: {overall_time:.4f} seconds")
    print(f"Z3 memory used: {memory_used:.2f} MB")

    if result == sat:
        model = solver.model()
        schedule = {
            p: [s for s in shifts if model.eval(x[p, s]).as_long() == 1]
            for p in physicians
        }
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
