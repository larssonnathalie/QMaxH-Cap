from z3 import *

def create_z3_model(physicians, shifts, demand, preference, Q):
    """Creates and returns a Z3 model using QUBO matrix"""
    x = {(p, s): Bool(f"x_{p}_{s}") for p in physicians for s in shifts}
    solver = Optimize()

    # Convert QUBO into Z3 sum
    qubo_objective = Sum([
        Q[i, j] * If(x[p1, s1], 1, 0) * If(x[p2, s2], 1, 0)
        for i, (p1, s1) in enumerate(x.keys())
        for j, (p2, s2) in enumerate(x.keys())
    ])

    solver.minimize(qubo_objective)

    return solver, x, qubo_objective