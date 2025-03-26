from z3 import *

def create_z3_model(physicians, shifts, demand, preference):
    """Creates a constraint-based Z3 model for physician scheduling"""
    x = {(p, s): Bool(f"x_{p}_{s}") for p in physicians for s in shifts}
    solver = Optimize()

    # Constraint 1: No consecutive shifts for a physician
    for p in physicians:
        for i in range(len(shifts) - 1):
            s1, s2 = shifts[i], shifts[i + 1]
            solver.add(Or(Not(x[(p, s1)]), Not(x[(p, s2)])))

    # Constraint 2: Demand match for each shift
    for s in shifts:
        solver.add(Sum([If(x[(p, s)], 1, 0) for p in physicians]) == demand[s])

    # Constraint 3: Availability/preferences
    for p in physicians:
        for s in shifts:
            if preference[p][s] == 0:
                solver.add(Not(x[(p, s)]))

    # Objective: minimize dissatisfaction (assigning non-preferred shifts)
    dissatisfaction = Sum([
        (1 - preference[p][s]) * If(x[(p, s)], 1, 0)
        for p in physicians for s in shifts
    ])
    solver.minimize(dissatisfaction)

    return solver, x, dissatisfaction
