from z3 import *  # Install using pip install z3-solver
import pandas as pd

def classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p, prints=True):
    n_physicians = physician_df.shape[0]
    n_shifts = demand_df.shape[0]

    # Z3 Solver
    solver = Solver()

    # Create binary decision variables: x[p][s] = 1 if physician p is assigned to shift s
    x = [[Bool(f"x_{p}_{s}") for s in range(n_shifts)] for p in range(n_physicians)]

    # Constraint 1: Each shift must be covered by exactly the required demand
    for s in range(n_shifts):
        solver.add(Sum([If(x[p][s], 1, 0) for p in range(n_physicians)]) == demand_df.loc[s, "demand"])

    # Constraint 2: No physician can exceed the maximum allowed number of shifts
    for p in range(n_physicians):
        solver.add(Sum([If(x[p][s], 1, 0) for s in range(n_shifts)]) <= max_shifts_per_p)

    # Solve the problem
    if solver.check() == sat:
        model = solver.model()

        # Create the schedule from the model
        schedule = []
        for s in range(n_shifts):
            assigned_physicians = [p for p in range(n_physicians) if model.evaluate(x[p][s], model_completion=True)]
            schedule.append({"date": demand_df.loc[s, "date"], "staff": assigned_physicians})

        result_schedule_df = pd.DataFrame(schedule)
        result_schedule_df.to_csv("result_schedule_classical_z3.csv", index=False)

        if prints:
            print("Optimized Schedule:")
            print(result_schedule_df)

        return result_schedule_df
    else:
        print("No feasible solution found.")
        return None


# Main Function
if __name__ == "__main__":
    start_date = '2025-02-10'
    end_date = '2025-02-17'

    # Create an example empty calendar
    empty_calendar_df = pd.DataFrame({
        "date": pd.date_range(start=start_date, end=end_date),
        "is_holiday": [False, False, False, False, False, True, True, False]
    })

    # Create example demand and physician data
    demand_df = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date), "demand": [4, 3, 2, 4, 3, 1, 1, 3]})
    physician_df = pd.DataFrame({"physician_id": ["p1", "p2", "p3", "p4"]})

    # Maximum shifts per physician
    max_shifts_per_p = (demand_df["demand"].sum() + len(physician_df) - 1) // len(physician_df)

    # Run the classical optimization
    result_schedule_df = classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p)
