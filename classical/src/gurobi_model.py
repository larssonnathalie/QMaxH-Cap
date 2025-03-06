from gurobipy import *

def create_gurobi_model(physicians, shifts, demand, preference):
    """Creates and returns the gurobi model for physician scheculing"""
    model = Model("Physician_Scheduling")

    # Decision variable: x[p,s] = 1 of physician p is assigned to shift s
    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # Constraint 1: No consecutive shifts for the same physician
    for p in physicians:
        for s in range(len(shifts) - 1):
            model.addConstr(x[p, shifts[s]] + x[p, shifts[s+1]] <= 1)

    # Constraint 2: Number of scheduled physicians matches demand for each shift
    for s in shifts:
        model.addConstr(sum(x[p,s] for p in physicians) == demand[s])

    # Constraint 3: A physician can only be scheduled when available
    for p in physicians:
        for s in shifts:
            if preference[p][s] == 0:
                model.addConstr(x[p,s] == 0)

    # Objective: min dissatisfaction (preference score)
    model.setObjective(sum(preference[p][s] * x[p,s] for p in physicians for s in shifts), GRB.MINIMIZE)

    return model, x