import sys
import os
from src.scheduler import solve_and_save_results
from src.plot_results import plot_schedule, compare_solutions
from src.qubo_solvers import load_qubo_matrix, solve_qubo_z3, solve_qubo_gurobi
from src.data_handler import load_data_from_intermediate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# File paths
shifts_file = "data/intermediate/shift_data.csv"
physicians_file = "data/intermediate/physician_data.csv"
cl = 3  # Complexity level for QUBO loading

print("Solving w/ classical solvers...")

# Classical solvers (constraint-based)
solution_z3 = solve_and_save_results(solver_type="z3", source="intermediate")
print("Z3 classical completed")
solution_gurobi = solve_and_save_results(solver_type="gurobi", source="intermediate")
print("Gurobi classical completed")

physicians, shifts, _, _ = load_data_from_intermediate()

# Plot classical schedules
if solution_z3:
    plot_schedule(solution_z3, physicians, shifts, title="Z3 (Classical)")
if solution_gurobi:
    plot_schedule(solution_gurobi, physicians, shifts, title="Gurobi (Classical)")

# QUBO-based solvers
print("Loading QUBO matrix")
Q = load_qubo_matrix(cl=cl)
print("Solving QUBO with Z3...")
qubo_z3_sol, qubo_z3_val = solve_qubo_z3(Q, timeout_ms=10000)  # 10 sec timeout
if qubo_z3_val is not None:
    print(f"Z3 (QUBO) completed with value: {qubo_z3_val}")
else:
    print("Z3 (QUBO) failed or timed out")

print("Solving QUBO with Gurobi...")
qubo_gurobi_sol, qubo_gurobi_val = solve_qubo_gurobi(Q)
if qubo_gurobi_val is not None:
    print(f"Gurobi (QUBO) completed with value: {qubo_gurobi_val}")
else:
    print("Gurobi (QUBO) failed")

# Show comparison bar plot
print("Plotting comparison...")
compare_solutions({
    "Z3 Classical": sum(len(v) for v in solution_z3.values()) if solution_z3 else None,
    "Gurobi Classical": sum(len(v) for v in solution_gurobi.values()) if solution_gurobi else None,
    "Z3 QUBO": qubo_z3_val,
    "Gurobi QUBO": qubo_gurobi_val
})

print("All scheduling runs completed.")