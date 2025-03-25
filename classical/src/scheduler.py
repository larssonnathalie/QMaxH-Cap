import csv
import sys
from .z3_model import create_z3_model
from .gurobi_model import create_gurobi_model
from .data_handler import load_data
from z3 import Or, is_true, sat
from gurobipy import GRB
import os

def solve_and_save_results(shifts_file, physicians_file, solver_type="z3", output_file=None, Q=None):
    """Solves the scheduling problem using Z3 or Gurobi and saves results"""

    solver_type = solver_type.strip().lower()

    if solver_type not in ["z3", "gurobi"]:
        print(f"Error: Unsupported solver type '{solver_type}'. Choose 'z3' or 'gurobi'.")
        sys.exit(1)

    # Load data
    physicians, shifts, demand, preference = load_data(shifts_file, physicians_file)

    if solver_type == "z3":
        print(f"Solving with Z3 optimizer...")
        solver, x, dissatisfaction = create_z3_model(physicians, shifts, demand, preference, Q)

        if solver.check() == sat:
            model = solver.model()
            solution = {p: [] for p in physicians}

            for (p, s) in x:
                if is_true(model.evaluate(x[(p, s)])):
                    solution[p].append(s)

            dissatisfaction_score = model.evaluate(dissatisfaction).as_long()
            print(f"Optimal Z3 solution found with dissatisfaction score: {dissatisfaction_score}")
            return solution
        else:
            print("No feasible solution found with Z3.")
            return None

    elif solver_type == "gurobi":
        print(f"Solving with Gurobi optimizer...")
        model, x = create_gurobi_model(physicians, shifts, demand, preference, Q)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution = {p: [] for p in physicians}

            for p in physicians:
                for s in shifts:
                    if x[p, s].x > 0.5:
                        solution[p].append(s)

            print(f"Optimal Gurobi solution found.")
            return solution
        else:
            print("No feasible solution found with Gurobi.")
            return None
