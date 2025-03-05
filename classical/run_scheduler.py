from scheduler import solve_and_save_results
from data_handler import load_data
from plot_results import plot_schedule

# Define paths
shifts_file = "shifts.csv"
physicians_file = "Physician_Data.csv"

# Solve and get schedules
all_schedules = solve_and_save_results(shifts_file, physicians_file)

# Plot the best schedule
if all_schedules:
    best_schedule = all_schedules[0][1] # Get schedule with lowest dissatisfaction
    physicians, shifts, _, _ = load_data(shifts_file, physicians_file)
    plot_schedule(best_schedule, physicians, shifts)