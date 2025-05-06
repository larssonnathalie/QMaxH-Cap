from z3 import *
import numpy as np
import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

def create_z3_model(physicians, shifts, demand, preference, optimize=False, lambdas=None):
    overall_start = time.time()

    if optimize:
        solver = Optimize()
    else:
        solver = Solver()

    set_param("parallel.enable", True)

    # Decision variables
    x = {(p, s): Int(f"x_{p}_{s}") for p in physicians for s in shifts}
    for var in x.values():
        solver.add(Or(var == 0, var == 1))

    # Load physician data
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    extent_map = {row["name"]: row["extent"] for _, row in physician_df.iterrows()}

    total_shifts = len(shifts)

    # === Hard Constraints ===

    # Demand satisfaction
    for s in shifts:
        solver.add(Sum([x[(p, s)] for p in physicians]) == demand[s])

    # Unavailability
    for (p, s) in x:
        if preference[p][s] == -2:
            solver.add(x[(p, s)] == 0)

    # One shift per day (memory constraint)
    shift_dates = {s: s.split(' ')[0] for s in shifts}
    shifts_by_date = defaultdict(list)
    for s in shifts:
        shifts_by_date[shift_dates[s]].append(s)

    for p in physicians:
        for date, shift_list in shifts_by_date.items():
            solver.add(Sum([x[(p, s)] for s in shift_list]) <= 1)

    # Extent-based fairness + memory
    extent_groups = defaultdict(list)
    for p in physicians:
        extent_groups[extent_map[p]].append(p)

    total_demand = sum(demand.values())

    # Prepare penalty terms if optimizing
    fairness_terms = []
    extent_terms = []
    memory_terms = []
    preference_terms = []

    for extent, group in extent_groups.items():
        group_target = total_shifts * (extent / 100) / len(group)
        lower = int(np.floor(group_target * 0.8))
        upper = int(np.ceil(group_target * 1.2))

        assigned_vars = {}
        for p in group:
            assigned = Sum([x[(p, s)] for s in shifts])
            assigned_vars[p] = assigned
            if not optimize:
                solver.add(assigned >= lower)
                solver.add(assigned <= upper)

        if not optimize:
            # Enforce fairness between group members
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    diff = Int(f"diff_{group[i]}_{group[j]}")
                    solver.add(diff >= assigned_vars[group[i]] - assigned_vars[group[j]])
                    solver.add(diff >= assigned_vars[group[j]] - assigned_vars[group[i]])
                    solver.add(diff <= 1)

        else:
            # Add fairness penalty terms
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    fairness_terms.append(Abs(assigned_vars[group[i]] - assigned_vars[group[j]]))

            # Extent deviation penalty
            for p in group:
                target = int(np.round(group_target))
                deviation = Int(f"deviation_{p}")
                solver.add(deviation >= assigned_vars[p] - target)
                solver.add(deviation >= target - assigned_vars[p])
                extent_terms.append(deviation)

            # Memory = total number of shifts
            memory_terms.extend([assigned_vars[p] for p in group])

    if optimize:
        # Preferences
        for (p, s), var in x.items():
            val = preference[p][s]
            if val == 1:
                preference_terms.append(-var)
            elif val == -1:
                preference_terms.append(var)

        if lambdas is None:
            lambdas = {'pref': 100, 'fair': 2, 'memory': 3, 'extent': 2}

        total_cost = (
            lambdas['pref'] * Sum(preference_terms) +
            lambdas['fair'] * Sum(fairness_terms) +
            lambdas['memory'] * Sum(memory_terms) +
            lambdas['extent'] * Sum(extent_terms)
        )
        solver.minimize(total_cost)

    # Solve
    solve_start = time.time()
    result = solver.check()
    solve_end = time.time()
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    print(f"Z3 solver time: {solver_time:.4f} seconds")
    print(f"Z3 overall time: {overall_time:.4f} seconds")

    if result == sat:
        model = solver.model()
        schedule = {p: [] for p in physicians}
        for (p, s), var in x.items():
            if model.eval(var).as_long() == 1:
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
