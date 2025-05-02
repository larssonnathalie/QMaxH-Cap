from z3 import *
import numpy as np
import pandas as pd
from datetime import datetime
import time

def create_z3_model(physicians, shifts, demand, preference, lambdas=None):
    overall_start = time.time()
    solver = Optimize()
    set_param("parallel.enable", True)

    if lambdas is None:
        lambdas = {'demand': 3, 'fair': 10, 'pref': 5, 'unavail': 10, 'extent': 8, 'rest': 0, 'titles': 5, 'memory': 3}

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    work_rate = dict(zip(physician_df['name'], physician_df['work rate']))

    # Precompute days passed per shift
    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {
        s: (datetime.strptime(s.split(' ')[0], "%Y-%m-%d") - start_date).days
        for s in shifts
    }

    # Create decision variables only for available assignments
    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:
                x[p, s] = Int(f"x_{p}_{s}")
                solver.add(Or(x[p, s] == 0, x[p, s] == 1))

    # Hard constraint: demand satisfaction
    for s in shifts:
        relevant_vars = [x[p, s] for p in physicians if (p, s) in x]
        solver.add(Sum(relevant_vars) == demand[s])

    # Fairness constraints
    total_demand = sum(demand.values())
    avg_assignments = int(np.ceil(total_demand / len(physicians)))

    fairness_terms = []
    for p in physicians:
        assigned_vars = [x[p, s] for s in shifts if (p, s) in x]
        total_assigned = Sum(assigned_vars)
        u_p = Int(f"u_{p}")
        solver.add(u_p >= total_assigned - avg_assignments)
        solver.add(u_p >= avg_assignments - total_assigned)
        fairness_terms.append(u_p)

    # Preference penalties
    preference_terms = []
    for (p, s), var in x.items():
        val = preference[p][s]
        if val == 1:
            preference_terms.append(-1 * var)
        elif val == -1:
            preference_terms.append(1 * var)

    # Memory penalty: total number of assigned shifts per physician
    memory_terms = [
        Sum([x[p, s] for s in shifts if (p, s) in x])
        for p in physicians
    ]

    # Extent penalty: based on work_rate over time
    extent_terms = []
    for p in physicians:
        wr = work_rate[p]
        for s in shifts:
            if (p, s) in x:
                days_passed = shift_days_passed[s]
                extent_priority = min(days_passed / 7, 1)
                priority = abs(extent_priority * (1 - float(wr)))
                term = RealVal(priority) * x[p, s]
                if wr < 1:
                    extent_terms.append(-term)
                else:
                    extent_terms.append(term)

    # Objective function
    total_cost = (
        lambdas['pref'] * Sum(preference_terms) +
        lambdas['fair'] * Sum(fairness_terms) +
        lambdas['memory'] * Sum(memory_terms) +
        lambdas['extent'] * Sum(extent_terms)
    )
    solver.minimize(total_cost)

    # Solve the model
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
