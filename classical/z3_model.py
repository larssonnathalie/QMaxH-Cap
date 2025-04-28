from z3 import *
import numpy as np
import pandas as pd
from datetime import datetime
import time

def create_z3_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    solver = Optimize()
    set_param("parallel.enable", True)

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5, 'memory': 3, 'extent': 2}

    x = {(p, s): Int(f"x_{p}_{s}") for p in physicians for s in shifts if preference[p][s] != -2}
    for var in x.values():
        solver.add(Or(var == 0, var == 1))

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')

    # Compute days passed per shift
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {}
    for s in shifts:
        day_str = s.split(' ')[0]
        shift_date = datetime.strptime(day_str, "%Y-%m-%d")
        days_passed = (shift_date - start_date).days
        shift_days_passed[s] = days_passed

    preference_terms = []
    fairness_terms = []
    memory_terms = []
    extent_terms = []

    if cl >= 1:
        for s in shifts:
            relevant_vars = [x[(p, s)] for p in physicians if (p, s) in x]
            if relevant_vars:
                solver.add(Sum(relevant_vars) == demand[s])

        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))

        for p in physicians:
            assigned_vars = [x[(p, s)] for s in shifts if (p, s) in x]
            if assigned_vars:
                assigned_sum = Sum(assigned_vars)
                u_p = Int(f"u_{p}")
                solver.add(u_p >= assigned_sum - avg_assignments)
                solver.add(u_p >= avg_assignments - assigned_sum)
                fairness_terms.append(u_p)

    if cl >= 2:
        for (p, s), var in x.items():
            val = preference[p][s]
            if val == 1:
                preference_terms.append(-var)
            elif val == -1:
                preference_terms.append(var)

        shift_days = [s.split(' ')[0] for s in shifts]
        for i in range(1, len(shifts)):
            if shift_days[i] == shift_days[i-1]:
                for p in physicians:
                    if (p, shifts[i]) in x and (p, shifts[i-1]) in x:
                        memory_terms.append(x[(p, shifts[i-1])] * x[(p, shifts[i])])

        for idx_p, p in enumerate(physicians):
            work_rate_p = physician_df['work rate'].iloc[idx_p]
            for s in shifts:
                if (p, s) in x:
                    days_passed = shift_days_passed[s]
                    extent_priority = min(days_passed / 7, 1)
                    priority_p = abs(extent_priority * (1 - float(work_rate_p)))

                    if work_rate_p < 1:
                        extent_terms.append(-priority_p * x[(p, s)])
                    else:
                        extent_terms.append(priority_p * x[(p, s)])

    total_cost = (
        lambdas['pref'] * Sum(preference_terms) +
        lambdas['fair'] * Sum(fairness_terms) +
        lambdas['memory'] * Sum(memory_terms) +
        lambdas['extent'] * Sum(extent_terms)
    )
    solver.minimize(total_cost)

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
