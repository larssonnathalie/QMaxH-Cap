from z3 import *

def create_z3_model(physicians, shifts, demand, preference):
    """Creates and returns the z3 model for physician scheduling"""
    x = {(p,s): Bool(f"x_{p}_{s}") for p in physicians for s in shifts}
    solver = Optimize()

    # Constraint 1: No consecutive shifts for the same physician
    for p in physicians:
        for s in range(len(shifts)-1):
            solver.add(Or(Not(x[(p, shifts[s])]), Not(x[(p, shifts[s+1])])))

    # Constraint 2: Number of scheduled physicians matches demand for each shift
    for s in shifts:
        solver.add(Sum([If(x[p,s], 1, 0) for p in physicians]) == demand[s])

    # Constraint 3: A physician can only be scheduled when available
    for p in physicians:
        for s in shifts:
            if preference[p][s] == 0:
                solver.add(Not(x[(p,s)]))

    # Objective function: minimise dissatisfaction (preference score)
    dissatisfaction = Sum([preference[p][s] * If(x[(p,s)], 1, 0) for p in physicians for s in shifts])
    solver.minimize(dissatisfaction)

    return solver, x, dissatisfaction