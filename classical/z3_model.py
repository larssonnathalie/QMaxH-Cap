import matplotlib.pyplot as plt
from z3 import *
import numpy as np
import pandas as pd
from datetime import datetime
from .memory_monitor import *

def create_z3_model(physicians, shifts, demand, preference, lambdas=None, plot_memory=False):
    overall_start = time.time()


    solver = Optimize()
    set_param("parallel.enable", True)

    if lambdas is None:
        lambdas = {'fair': 10, 'pref': 5, 'extent': 8, 'memory': 3}

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    work_rate = dict(zip(physician_df['name'], physician_df['work rate']))

    start_date = datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {
        s: (datetime.strptime(s.split(' ')[0], "%Y-%m-%d") - start_date).days
        for s in shifts
    }

    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:
                x[p, s] = Int(f"x_{p}_{s}")
                solver.add(Or(x[p, s] == 0, x[p, s] == 1))

    for s in shifts:
        solver.add(Sum([x[p, s] for p in physicians if (p, s) in x]) == demand[s])

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

    preference_terms = []
    for (p, s), var in x.items():
        val = preference[p][s]
        if val == 1:
            preference_terms.append(-1 * var)
        elif val == -1:
            preference_terms.append(1 * var)

    memory_terms = [
        Sum([x[p, s] for s in shifts if (p, s) in x])
        for p in physicians
    ]

    extent_terms = []
    for p in physicians:
        wr = work_rate[p]
        for s in shifts:
            if (p, s) in x:
                days_passed = shift_days_passed[s]
                extent_priority = min(days_passed / 7, 1)
                priority = abs(extent_priority * (1 - float(wr)))
                term = RealVal(priority) * x[p, s]
                extent_terms.append(-term if wr < 1 else term)

    total_cost = (
        lambdas['pref'] * Sum(preference_terms) +
        lambdas['fair'] * Sum(fairness_terms) +
        lambdas['memory'] * Sum(memory_terms) +
        lambdas['extent'] * Sum(extent_terms)
    )
    solver.minimize(total_cost)

    # === Solve with monitored memory ===
    monitor = MemoryMonitor()
    monitor.start()
    solve_start = time.time()
    result = solver.check()
    solve_end = time.time()
    monitor.stop()

    solver_time = solve_end - solve_start
    overall_time = time.time() - overall_start
    peak_memory = monitor.peak_memory

    print(f"Z3 solver time: {solver_time:.4f} seconds")
    print(f"Z3 overall time: {overall_time:.4f} seconds")
    print(f"Peak RSS memory used during solve: {peak_memory:.2f} MB")

    if plot_memory:
        trace = monitor.get_trace()
        times, mems = zip(*trace)
        plt.figure(figsize=(8, 4))
        plt.plot(times, mems, label="Memory (MB)")
        plt.xlabel("Time (s)")
        plt.ylabel("RSS Memory (MB)")
        plt.title("Z3 Memory Usage Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if result == sat:
        model = solver.model()
        schedule = {p: [] for p in physicians}
        for (p, s), var in x.items():
            if model.eval(var).as_long() == 1:
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
