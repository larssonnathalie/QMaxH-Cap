import matplotlib.pyplot as plt
from gurobipy import *
import numpy as np
import pandas as pd
import datetime
from .memory_monitor import *

def create_gurobi_model(physicians, shifts, demand, preference, lambdas=None, plot_memory=False):
    overall_start = time.time()

    model = Model("PhysicianScheduling")
    model.setParam("OutputFlag", 0)

    if lambdas is None:
        lambdas = {'fair': 10, 'pref': 5, 'extent': 8, 'memory': 3}

    physician_df = pd.read_csv('data/intermediate/physician_data.csv')
    work_rate = dict(zip(physician_df['name'], physician_df['work rate']))

    start_date = datetime.datetime.strptime(shifts[0].split(' ')[0], "%Y-%m-%d")
    shift_days_passed = {
        s: (datetime.datetime.strptime(s.split(' ')[0], "%Y-%m-%d") - start_date).days
        for s in shifts
    }

    x = {}
    for p in physicians:
        for s in shifts:
            if preference[p][s] != -2:
                x[p, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{s}")

    for s in shifts:
        model.addConstr(
            quicksum(x[p, s] for p in physicians if (p, s) in x) == demand[s],
            name=f"demand_{s}"
        )

    total_demand = sum(demand.values())
    avg_assignments = int(np.ceil(total_demand / len(physicians)))

    u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")
    for p in physicians:
        assigned_vars = [x[p, s] for s in shifts if (p, s) in x]
        total_assigned = quicksum(assigned_vars)
        model.addConstr(total_assigned - avg_assignments <= u[p])
        model.addConstr(avg_assignments - total_assigned <= u[p])
    fairness_expr = quicksum(u[p] for p in physicians)

    preference_expr = quicksum(
        -x[p, s] if preference[p][s] == 1 else x[p, s]
        for (p, s) in x if preference[p][s] in [1, -1]
    )

    memory_expr = 0
    shift_days = [s.split(' ')[0] for s in shifts]
    for i in range(1, len(shifts)):
        if shift_days[i] == shift_days[i - 1]:
            for p in physicians:
                if (p, shifts[i]) in x and (p, shifts[i - 1]) in x:
                    memory_expr += x[p, shifts[i]] * x[p, shifts[i - 1]]

    extent_expr = 0
    for (p, s) in x:
        days_passed = shift_days_passed[s]
        extent_priority = min(days_passed / 7, 1)
        rate_diff = 1 - float(work_rate[p])
        weight = abs(extent_priority * rate_diff)
        extent_expr += (-weight if work_rate[p] < 1 else weight) * x[p, s]

    total_cost = (
        lambdas['pref'] * preference_expr +
        lambdas['fair'] * fairness_expr +
        lambdas['memory'] * memory_expr +
        lambdas['extent'] * extent_expr
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # === Solve with monitored memory ===
    monitor = MemoryMonitor()
    monitor.start()
    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    monitor.stop()

    solver_time = solve_end - solve_start
    overall_time = time.time() - overall_start
    peak_memory = monitor.peak_memory

    print(f"Gurobi solver time: {solver_time:.4f} seconds")
    print(f"Gurobi overall time: {overall_time:.4f} seconds")
    print(f"Peak RSS memory used during solve: {peak_memory:.2f} MB")

    if plot_memory:
        trace = monitor.get_trace()
        times, mems = zip(*trace)
        plt.figure(figsize=(8, 4))
        plt.plot(times, mems, label="Memory (MB)")
        plt.xlabel("Time (s)")
        plt.ylabel("RSS Memory (MB)")
        plt.title("Gurobi Memory Usage Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for (p, s), var in x.items():
            if var.X > 0.5:
                schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
