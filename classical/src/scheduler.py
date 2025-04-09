from .z3_model import create_z3_model
from .gurobi_model import create_gurobi_model
from .data_handler import load_data_from_intermediate
from z3 import is_true, sat
from gurobipy import GRB

def solve_and_save_results(solver_type="z3", source="intermediate", cl=2, lambdas=None):
    """
    Solves the physician scheduling problem using Z3 or Gurobi (constraint-based).
    Returns: dict of physician -> assigned shifts, or None.
    """

    solver_type = solver_type.lower()
    if solver_type not in ["z3", "gurobi"]:
        raise ValueError(f"Unsupported solver type '{solver_type}'.")

    print(f"Loading data from {source} source...")
    physicians, shifts, demand, preference = load_data_from_intermediate()

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 1, 'unavail': 5}

    if solver_type == "z3":
        print("Running classical Z3 solver...")
        solution = create_z3_model(physicians, shifts, demand, preference, cl=cl, lambdas=lambdas)

        if solution is not None:
            print("Z3: Optimal solution found.")
        else:
            print("Z3: No feasible solution found.")
        return solution

    elif solver_type == "gurobi":
        print("Running classical Gurobi solver...")
        solution = create_gurobi_model(physicians, shifts, demand, preference, cl=cl, lambdas=lambdas)

        if solution is not None:
            print("Gurobi: Optimal solution found.")
        else:
            print("Gurobi: No feasible solution found.")
        return solution