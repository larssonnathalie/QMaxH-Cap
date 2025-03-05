import sys
import os

# Add /src to the Python module search path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.scheduler import solve_and_save_results
from src.data_handler import load_data
from src.plot_results import plot_schedule

# Define paths
shifts_file = "data/shifts_test.csv"
physicians_file = "data/physicians_test.csv"

# Solve and get schedules
all_schedules = solve_and_save_results(shifts_file, physicians_file)

# Plot the best schedule
if all_schedules:
    best_schedule = all_schedules[0][1] # Get schedule with lowest dissatisfaction
    physicians, shifts, _, _ = load_data(shifts_file, physicians_file)
    plot_schedule(best_schedule, physicians, shifts)