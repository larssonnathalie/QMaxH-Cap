from gurobipy import *

def create_gurobi_model(physicians, shifts, demand, preference, Q):
    """Creates and returns the Gurobi model using QUBO matrix"""
    model = Model("QUBO_Scheduling")

    x = model.addVars(physicians, shifts, vtype=GRB.BINARY, name="x")

    # Convert QUBO into Gurobi quadratic objective
    qubo_objective = quicksum(
        Q[i, j] * x[p1, s1] * x[p2, s2]
        for i, (p1, s1) in enumerate(x.keys())
        for j, (p2, s2) in enumerate(x.keys())
    )

    model.setObjective(qubo_objective, GRB.MINIMIZE)

    return model, x