import csv
import sys
from .z3_model import create_z3_model
from .gurobi_model import create_gurobi_model
from .data_handler import load_data
from z3 import Or, is_true, sat
from gurobipy import GRB
import os

def solve_and_save_results(shifts_file, physicians_file, solver_type="z3", output_file=None):
    """Solves the scheduling problem using either Z3 or Gurobi and saves results to a CSV file"""

    # Normalize solver_type input
    solver_type = solver_type.strip().lower()

    # Validate solver type
    if solver_type not in ["z3", "gurobi"]:
        print(f"Error: Unsupported solver type '{solver_type}'. Choose 'z3' or 'gurobi'.")
        sys.exit(1)

    # Set output file dynamically
    if output_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
        results_dir = os.path.join(base_dir, "..", "results")  # Move results to /results/

        if solver_type == "z3":
            output_file = os.path.join(results_dir, "schedules_z3.csv")
        elif solver_type == "gurobi":
            output_file = os.path.join(results_dir, "schedules_gurobi.csv")

    # Load data
    physicians, shifts, demand, preference = load_data(shifts_file, physicians_file)

    # Use Z3 Solver
    if solver_type == "z3":
        print(f"Solving with Z3 optimizer... Results will be saved in '{output_file}'")
        solver, x, dissatisfaction = create_z3_model(physicians, shifts, demand, preference)

        all_schedules = []
        solution_count = 0

        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Solution", "Dissatisfaction Score"] + shifts)

            while solver.check() == sat:
                model = solver.model()
                schedule = {p: [] for p in physicians}

                # Extract schedule
                for (p, s) in x:
                    if is_true(model.evaluate(x[(p, s)])):
                        schedule[p].append(s)

                dissatisfaction_score = model.evaluate(dissatisfaction).as_long()
                all_schedules.append((dissatisfaction_score, schedule))
                solution_count += 1

                # Save solution to CSV
                row = [solution_count, dissatisfaction_score]
                for s in shifts:
                    assigned_physicians = [p for p in physicians if s in schedule[p]]
                    row.append(", ".join(assigned_physicians))
                writer.writerow(row)

                # Block the current solution to find a new one
                solver.add(Or([x[p, s] != model.evaluate(x[p, s]) for p in physicians for s in shifts]))

                print(f"Solutions found: {solution_count}")

        print(f"\nTotal feasible solutions found with Z3: {solution_count}")
        return all_schedules

    # Use Gurobi Solver
    elif solver_type == "gurobi":
        print(f"Solving with Gurobi optimizer... Results will be saved in '{output_file}'")
        model, x = create_gurobi_model(physicians, shifts, demand, preference)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution_count = 1
            all_schedules = []

            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Solution", "Dissatisfaction Score"] + shifts)

                # Extract solution
                schedule = {p: [] for p in physicians}
                for p in physicians:
                    for s in shifts:
                        if x[p, s].x > 0.5:  # If assigned in optimal solution
                            schedule[p].append(s)

                # Compute dissatisfaction score
                dissatisfaction_score = model.objVal
                all_schedules.append((dissatisfaction_score, schedule))

                # Save solution to CSV
                row = [solution_count, dissatisfaction_score]
                for s in shifts:
                    assigned_physicians = [p for p in physicians if s in schedule[p]]
                    row.append(", ".join(assigned_physicians))
                writer.writerow(row)

            print(f"\nTotal feasible solutions found with Gurobi: {solution_count}")
            return all_schedules

        else:
            print("No feasible solution found with Gurobi.")
            return []
