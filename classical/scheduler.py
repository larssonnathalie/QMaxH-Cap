from .z3_model import create_z3_model
from .gurobi_model import create_gurobi_model
from .data_handler import load_data_from_intermediate
import pandas as pd

def solve_and_save_results(solver_type="z3", source="intermediate", cl=2, lambdas=None):
    """
    Solves the physician scheduling problem using Z3 or Gurobi (constraint-based).
    Returns: dict of physician -> assigned shifts, or None.
    """
    solver_type = solver_type.lower()
    physicians, shifts, demand, preference = load_data_from_intermediate()

    if lambdas is None:
        lambdas = {'demand': 5, 'fair': 2, 'pref': 1, 'unavail': 5}

    if solver_type == "z3":
        solution = create_z3_model(physicians, shifts, demand, preference, cl=cl, lambdas=lambdas)

        if solution[0] is not None:
            print("Z3: Optimal solution found.")
        else:
            print("Z3: No feasible solution found.")
        return solution

    elif solver_type == "gurobi":
        solution = create_gurobi_model(physicians, shifts, demand, preference, cl=cl, lambdas=lambdas)

        if solution[0] is not None:
            print("Gurobi: Optimal solution found.")
        else:
            print("Gurobi: No feasible solution found.")
        return solution

def schedule_dict_to_df(schedule, shifts_df):
        n_shifts = len(shifts_df)
        df = pd.DataFrame({'date': shifts_df['date'], 'staff': [[] for _ in range(n_shifts)]})
        for p, shift_list in schedule.items():
            pid = int(p.replace('physician', '')) if p.startswith('physician') else p
            for s in shift_list:
                s_index = shifts_df.index[shifts_df['date'] == s][0]
                df.at[s_index, 'staff'].append(str(pid))
        return df

