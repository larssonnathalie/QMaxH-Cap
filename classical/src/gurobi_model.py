from gurobipy import *
import numpy as np
import time

def create_gurobi_model(physicians, shifts, demand, preference, cl=1, lambdas=None):
    start_time = time.time()
    model = Model("Linear_Scheduling")
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 1, 'unavail': 5}

    preference_expr = LinExpr()
    fairness_expr = LinExpr()

    if cl >= 1:
        for s in shifts:
            model.addConstr(quicksum(x[p, s] for p in physicians) == demand[s], name=f"demand_{s}")

        for p in physicians:
            for i in range(len(shifts) - 1):
                model.addConstr(x[p, shifts[i]] + x[p, shifts[i+1]] <= 1, name=f"no_consec_{p}_{i}")

        total_demand = sum(demand.values())
        avg_assignments = int(np.ceil(total_demand / len(physicians)))
        u = model.addVars(physicians, vtype=GRB.INTEGER, name="u")
        for p in physicians:
            assigned = quicksum(x[p, s] for s in shifts)
            model.addConstr(assigned - avg_assignments <= u[p], name=f"fair_upper_{p}")
            model.addConstr(avg_assignments - assigned <= u[p], name=f"fair_lower_{p}")
        fairness_expr = lambdas['fair'] * quicksum(u[p] for p in physicians)

    if cl >= 2:
        for p in physicians:
            for s in shifts:
                val = preference[p][s]
                if val == 1:
                    preference_expr += -lambdas['pref'] * x[p, s]
                elif val == -1:
                    preference_expr += lambdas['pref'] * x[p, s]
                elif val == -2:
                    model.addConstr(x[p, s] == 0, name=f"unavail_{p}_{s}")

    model.setObjective(preference_expr + fairness_expr, GRB.MINIMIZE)
    model.optimize()
    end_time = time.time()

    print(f"Gurobi solve time: {end_time - start_time:.4f} seconds")

    if model.status == GRB.OPTIMAL:
        schedule = {p: [] for p in physicians}
        for p in physicians:
            for s in shifts:
                if x[p, s].X > 0.5:
                    schedule[p].append(s)
        return schedule
    else:
        return None