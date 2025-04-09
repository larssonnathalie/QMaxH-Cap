# run_scheduler.py
import pandas as pd
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt
from src.z3_model import create_z3_model
from src.gurobi_model import create_gurobi_model
from src.data_handler import load_data_from_intermediate
from z3 import is_true, sat
from gurobipy import GRB

def solve_and_save_results(solver_type="gurobi", source="intermediate", cl=2, lambdas=None):
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

def bitstringIndexToPS(idx, n_vars, n_shifts):
    idx=int((n_vars-1)-idx)
    p = int(idx/n_shifts)
    s = idx%n_shifts
    return p,s

def psVarNamesToI(xp_s, n_shifts):
    p,s = xp_s.lstrip('x').split('_')
    i = int(p) * n_shifts + int(s)
    return f'x{i}'

def bitstringToSchedule(bitstring:str, empty_calendar_df, cl, n_shifts, prints=True) -> pd.DataFrame:
    staff_col = [[] for _ in range(empty_calendar_df.shape[0])]

    n_vars = len(bitstring)
    for i in range(n_vars): 
        bit = bitstring[i]
        if bit == '1':
            p,s = bitstringIndexToPS(i, n_vars=n_vars, n_shifts=n_shifts)
            staff_col[s].append(str(p))

    result_schedule_df = empty_calendar_df.copy()
    result_schedule_df['staff'] = staff_col
    return result_schedule_df

def controlSchedule(result_schedule_df, shift_data_df, cl, prints=True):
    combined_df = shift_data_df.merge(result_schedule_df, on='date', how='outer')
    ok_col = []
    for i in range(combined_df.shape[0]):
        if combined_df.loc[i,'demand'] == len(combined_df.loc[i,'staff']):
            ok_col.append('ok')
        else:
            ok_col.append('NOT ok!')
    combined_df['shift covered'] = ok_col
    if prints:
        print(combined_df)
    combined_df.to_csv(f'data/results/result_and_demand_cl{cl}.csv', index=False)
    return combined_df

def controlPlot(result_df):
    physician_df = pd.read_csv('data/intermediate/physician_data.csv', index_col=False)
    n_physicians = len(physician_df)
    n_shifts = len(result_df)

    result_matrix = np.zeros((n_physicians, n_shifts))
    for s in range(n_shifts):
        workers_s = result_df['staff'].iloc[s]
        for p in workers_s:
            result_matrix[int(p)][s] = 1 

    prefer_matrix = np.zeros((n_physicians, n_shifts))
    for p in range(n_physicians):
        prefer_p = physician_df['prefer'].iloc[p]
        if prefer_p != '[]':
            prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')
            for s in prefer_shifts_p:
                prefer_matrix[p][int(s)] = 1

        prefer_not_p = physician_df['prefer not'].iloc[p]
        if prefer_not_p != '[]':
            prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')
            for s in prefer_not_shifts_p:
                prefer_matrix[p][int(s)] = -1

        unavail_p = physician_df['unavailable'].iloc[p]
        if unavail_p != '[]':
            unavail_shifts_p = unavail_p.strip('[').strip(']').split(',')
            for s in unavail_shifts_p:
                prefer_matrix[p][int(s)] = -2

    ok_row = np.zeros((1, n_shifts))
    ok_row[0, :] = result_df['shift covered'] == 'ok'

    x_size = 5
    y_size = n_physicians / n_shifts * x_size
    plt.figure(figsize=(x_size, y_size))
    prefer_colors = np.where(prefer_matrix.flatten() == 1, 'lightgreen', prefer_matrix.flatten())
    prefer_colors = np.where(prefer_matrix.flatten() == -1, 'pink', prefer_colors)
    prefer_colors = np.where(prefer_matrix.flatten() == 0, 'none', prefer_colors)
    prefer_colors = np.where(prefer_matrix.flatten() == -2, 'red', prefer_colors)

    plt.pcolor(np.arange(n_shifts + 1) - 0.5, np.arange(n_physicians + 1) - 0.5, result_matrix, cmap="Greens")
    x, y = np.meshgrid(np.arange(n_shifts), np.arange(n_physicians))
    plt.scatter(x.ravel(), y.ravel(), s=(50 * (x_size / n_shifts))**2, c='none', marker='s', linewidths=9, edgecolors=prefer_colors)
    plt.pcolor(np.arange(n_shifts + 1) - 0.5, [n_physicians - 0.5, n_physicians - 0.4], ok_row, cmap='RdYlGn', vmin=0, vmax=1)

    plt.xticks(ticks=np.arange(n_shifts), labels=[date for date in result_df['date']])
    yticks = [i for i in np.arange(n_physicians)] + [n_physicians - 0.4]
    plt.yticks(ticks=yticks, labels=[phys[-1] for phys in physician_df['name']] + ['OK n.o.\nworkers'])
    plt.show()

# Configuration
cl = 2
solver_type = "gurobi"
lambdas = {'demand': 5, 'fair': 2, 'pref': 1, 'unavail': 5}

# Run solver
schedule = solve_and_save_results(solver_type=solver_type, cl=cl, lambdas=lambdas)

if schedule is None:
    print("No schedule generated.")
else:
    print("\nSchedule:")
    for p, shifts in schedule.items():
        print(f"{p}: {shifts}")

    # Load calendar and shift info
    empty_calendar_df = pd.read_csv("data/intermediate/empty_calendar.csv")
    shift_data_df = pd.read_csv("data/intermediate/shift_data.csv")
    n_shifts = len(shift_data_df)

    # Build schedule matrix from schedule dict
    bitstring = ''.join(['1' if s in schedule[p] else '0'
                         for p in range(len(schedule))
                         for s in range(n_shifts)])

    schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl=cl, n_shifts=n_shifts)
    result_df = controlSchedule(schedule_df, shift_data_df, cl=cl)
    controlPlot(result_df)