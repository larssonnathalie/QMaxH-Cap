from gurobipy import *

def create_gurobi_model(physicians, shifts, demand, preference):
    """Creates a constraint-based Gurobi model for physician scheduling"""
    model = Model("Physician_Scheduling")
    model.setParam("OutputFlag", 0)  # Silent mode

    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # Constraint 1: No consecutive shifts for a physician
    for p in physicians:
        for i in range(len(shifts) - 1):
            model.addConstr(x[p, shifts[i]] + x[p, shifts[i + 1]] <= 1)

    # Constraint 2: Demand match for each shift
    for s in shifts:
        model.addConstr(sum(x[p, s] for p in physicians) == demand[s])

    # Constraint 3: Availability/preferences
    for p in physicians:
        for s in shifts:
            if preference[p][s] == 0:
                model.addConstr(x[p, s] == 0)

    # Objective: minimize dissatisfaction (same idea as Z3)
    dissatisfaction = quicksum(
        (1 - preference[p][s]) * x[p, s]
        for p in physicians for s in shifts
    )
    model.setObjective(dissatisfaction, GRB.MINIMIZE)

    return model, x
