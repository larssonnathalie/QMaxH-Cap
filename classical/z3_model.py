from z3 import *
import numpy as np
import time

def create_z3_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    overall_start = time.time()
    solver = Optimize()
# TODO Lambda har ingen påverkan
    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 100, 'unavail': 5}

    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:
                x[(p, s)] = Bool(f"x_{p}_{s}")

    preference_penalties = []
    fairness_penalties = []

    if cl >= 1:
        # Demand constraints
        for s in shifts:
            summands = [If(x[(p, s)], 1, 0) for p in physicians if (p, s) in x]
            solver.add(Sum(summands) == IntVal(demand[s]))

        # No consecutive shifts
        for p in physicians:
            for i in range(len(shifts) - 1):
                s1, s2 = shifts[i], shifts[i + 1]
                if (p, s1) in x and (p, s2) in x:  # === Optimization 3 ===
                    solver.add(Or(Not(x[(p, s1)]), Not(x[(p, s2)])))

        # Fairness penalty using auxiliary variables (no Abs)
        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))
        for p in physicians:
            assigned = Sum([If(x[(p, s)], 1, 0) for s in shifts if (p, s) in x])
            u_p = Int(f"u_{p}")
            solver.add(u_p >= assigned - IntVal(avg_assignments))  # === Optimization 1 ===
            solver.add(u_p >= IntVal(avg_assignments) - assigned)
            fairness_penalties.append(u_p)

    if cl >= 2:
        for p in physicians:
            for s in shifts:
                if (p, s) not in x:
                    continue  # already marked as unavailable
                val = preference[p][s]
                if val == 1:
                    preference_penalties.append(-lambdas['pref'] * If(x[(p, s)], 1, 0))
                elif val == -1:
                    preference_penalties.append(lambdas['pref'] * If(x[(p, s)], 1, 0))
                # -2 case already skipped above

    fairness_cost = Sum(fairness_penalties)
    preference_cost = Sum(preference_penalties)
    solver.minimize(preference_cost + lambdas['fair'] * fairness_cost)

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
            if is_true(model.evaluate(var)):
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
