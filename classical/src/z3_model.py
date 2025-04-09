from z3 import *
import numpy as np
import time

def create_z3_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    start_time = time.time()
    solver = Optimize()
    x = {(p, s): Bool(f"x_{p}_{s}") for p in physicians for s in shifts}

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

    if solver.check() == sat:
        model = solver.model()
        schedule = {p: [] for p in physicians}
        for (p, s) in x:
            if is_true(model.evaluate(x[p, s])):
                schedule[p].append(s)
        end_time = time.time()
        print(f"Z3 solve time: {end_time - start_time:.4f} seconds")
        return schedule
    else:
        end_time = time.time()
        print(f"Z3 solve time (no solution): {end_time - start_time:.4f} seconds")
        return None