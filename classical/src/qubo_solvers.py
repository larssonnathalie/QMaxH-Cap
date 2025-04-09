import os
import pandas as pd
import gurobipy as gp
import z3

def load_qubo_matrix(cl=3):
    """Load QUBO matrix from file for a given complexity level."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "intermediate",
        f"Qubo_matrix_cl{cl}.csv"
    )
    path = os.path.abspath(path)
    return pd.read_csv(path, header=None).values

def solve_qubo_gurobi(Q, timeout_sec=50):
    """Solve QUBO problem using Gurobi."""
    n = Q.shape[0]
    model = gp.Model("QUBO_Gurobi")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", timeout_sec)

    x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n)), gp.GRB.MINIMIZE)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        solution = [int(x[i].x > 0.5) for i in range(n)]
        print(solution)
        print("----")
        print(model.objVal)
        return solution, model.objVal

    return None, None

def solve_qubo_z3(Q, timeout_ms=50000):
    """Solve QUBO problem using Z3."""
    n = Q.shape[0]
    solver = z3.Optimize()
    solver.set("timeout", timeout_ms)

    x = [z3.Bool(f"x_{i}") for i in range(n)]

    cost = z3.Sum([
        Q[i][j] * z3.If(x[i], 1, 0) * z3.If(x[j], 1, 0)
        for i in range(n) for j in range(n)
    ])

    solver.minimize(cost)

    if solver.check() == z3.sat:
        model = solver.model()
        solution = [1 if z3.is_true(model[x[i]]) else 0 for i in range(n)]

        # Safe way to evaluate the cost using the model
        evaluated_cost = model.evaluate(cost, model_completion=True)
        value = int(str(evaluated_cost))
        print(solution)
        print("----")
        print(value)
        return solution, value

    return None, None