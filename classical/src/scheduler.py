from .z3_model import create_z3_model
from .gurobi_model import create_gurobi_model
from .data_handler import load_data_from_intermediate
from z3 import is_true, sat
from gurobipy import GRB

def solve_and_save_results(solver_type="z3", source="intermediate"):
    """
    Solves the physician scheduling problem using Z3 or Gurobi (constraint-based).
    Returns: dict of physician -> assigned shifts, or None.
    """

    solver_type = solver_type.lower()
    if solver_type not in ["z3", "gurobi"]:
        raise ValueError(f"Unsupported solver type '{solver_type}'.")

    print(f"Loading data from {source} source...")
    physicians, shifts, demand, preference = load_data_from_intermediate()

    if solver_type == "z3":
        print(f"Running classical Z3 solver...")
        solver, x, dissatisfaction = create_z3_model(physicians, shifts, demand, preference)

        if solver.check() == sat:
            model = solver.model()
            solution = {p: [s for s in shifts if is_true(model.evaluate(x[p, s]))] for p in physicians}
            score = model.evaluate(dissatisfaction).as_long()
            print(f"Z3: Optimal solution found with dissatisfaction = {score}")
            return solution
        else:
            print("Z3: No feasible solution found.")
            return None

    elif solver_type == "gurobi":
        print(f"Running classical Gurobi solver...")
        model, x = create_gurobi_model(physicians, shifts, demand, preference)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution = {p: [s for s in shifts if x[p, s].x > 0.5] for p in physicians}
            print("Gurobi: Optimal solution found.")
            return solution
        else:
            print("Gurobi: No feasible solution found.")
            return None