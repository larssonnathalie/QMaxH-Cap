from gurobipy import *
import numpy as np
import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

def create_gurobi_model(physicians, shifts, demand, preference, lambdas=None):
    overall_start = time.time()
    model = Model("Feasible_Scheduling")
    model.setParam("OutputFlag", 0)  # Suppress console output

    # Load physician data
    physician_df = pd.read_csv('data/intermediate/physician_data.csv')

    # Decision variables: x[p,s] ∈ {0,1}
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # === Constraint 1: Demand satisfaction ===
    for s in shifts:
        model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

    # === Constraint 2: Unavailability ===
    for p in physicians:
        for s in shifts:
            if preference[p][s] == -2:
                model.addConstr(x[p, s] == 0, name=f"unavail_{p}_{s}")

    # === Constraint 3: Extent & Memory ===
    total_shifts = len(shifts)
    extent_map = {row["name"]: row["extent"] for _, row in physician_df.iterrows()}

    # Group physicians by extent value
    extent_groups = defaultdict(list)
    for p in physicians:
        extent_groups[extent_map[p]].append(p)

    for extent, group in extent_groups.items():
        target = total_shifts * (extent / 100) / len(group)
        min_shifts = int(np.floor(target * 0.8))
        max_shifts = int(np.ceil(target * 1.2))

        assigned = {}
        for p in group:
            assigned[p] = model.addVar(vtype=GRB.INTEGER, name=f"assigned_{p}")
            model.addConstr(assigned[p] == quicksum(x[p, s] for s in shifts))
            model.addConstr(assigned[p] >= min_shifts, name=f"min_extent_{p}")
            model.addConstr(assigned[p] <= max_shifts, name=f"max_extent_{p}")

        # Fairness within group (memory): max ±1 shift diff
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                p1, p2 = group[i], group[j]
                diff = model.addVar(vtype=GRB.INTEGER, name=f"diff_{p1}_{p2}")
                model.addConstr(diff >= assigned[p1] - assigned[p2])
                model.addConstr(diff >= assigned[p2] - assigned[p1])
                model.addConstr(diff <= 1, name=f"fair_diff_{p1}_{p2}")

    # Dummy objective (Gurobi requires one even for feasibility)
    model.setObjective(0, GRB.MINIMIZE)

    # === Solve ===
    solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    overall_end = time.time()

    solver_time = solve_end - solve_start
    overall_time = overall_end - overall_start

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for p in physicians:
            for s in shifts:
                if x[p, s].X > 0.5:
                    schedule[p].append(s)
        return schedule, solver_time, overall_time
    else:
        return None, solver_time, overall_time
