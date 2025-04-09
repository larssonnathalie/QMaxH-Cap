from z3 import *
import numpy as np
import time

def create_z3_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    x = {(p, s): Bool(f"x_{p}_{s}") for p in physicians for s in shifts}
    solver = Optimize()

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 1, 'unavail': 5}

    preference_penalties = []
    fairness_terms = []

    if cl >= 1:
        for s in shifts:
            solver.add(Sum([If(x[p, s], 1, 0) for p in physicians]) == demand[s])
        for p in physicians:
            for i in range(len(shifts) - 1):
                s1, s2 = shifts[i], shifts[i + 1]
                solver.add(Or(Not(x[p, s1]), Not(x[p, s2])))
        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))
        for p in physicians:
            assigned = Sum([If(x[p, s], 1, 0) for s in shifts])
            fairness_terms.append(Abs(assigned - avg_assignments))

    if cl >= 2:
        for p in physicians:
            for s in shifts:
                val = preference[p][s]
                if val == 1:
                    preference_penalties.append(-lambdas['pref'] * If(x[p, s], 1, 0))
                elif val == -1:
                    preference_penalties.append(lambdas['pref'] * If(x[p, s], 1, 0))
                elif val == -2:
                    solver.add(Not(x[p, s]))

    solver.minimize(Sum(preference_penalties) + lambdas['fair'] * Sum(fairness_terms))

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
        for (p, s) in x:
            if is_true(model.evaluate(x[p, s])):
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time