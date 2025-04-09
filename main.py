from qaoa.qaoa import *
from classical.src.scheduler import solve_and_save_results, schedule_dict_to_df
from postprocessing.postprocessing import bitstringToSchedule, controlSchedule, controlPlot
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
    # best angles depend heavily on initialization, COBYLA finds very local optima
        # fixed for now by comparing many random initialization outcomes & taking the best
        # should we change to a global optimizer instead? Or is something wrong?
    # Implement on real IBM backend
    # How maximize fairness when workers have different percentages?
        # Focus on fairness of shift type/weekday/holidays?
        # How adapt demand to extent?
            # Soften demand-constraint so it is a minimum but more is ok? Impossible for constant Q-matrix (without slack vars)
    # Competence constraint
    # Fix universal way of storing list-like objects in csv
    # QAOA class instead of functions
    # Bugfix in postprocessing, missing rows in output schedules!

# Parameters
start_date = '2025-04-01' # including this date
end_date = '2025-04-06' # including this date
weekday_demand = 2
holiday_demand = 1
n_physicians = 5   #TODO should depend on al
cl = 2 # complexity level:
# cl1: demand, fairness
# cl2: demand, fairness, preferences, unavailable
# cl3: demand, fairness, preferences, unavailable, titles, competence
# cl4: demand, fairness, preferences, unavailable, titles, competence, shift_type, rest
# cl5: demand, fairness, preferences, unavailable, titles, competence, shift_type, rest,  side_tasks

prints = True
plots = True
classical = True
draw_circuit = False

n_layers = 2
search_iterations = 30
estimation_iterations = n_layers * 100
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions

# lambdas = penalties (how hard a constraint is)
lambdas = {'demand':5, 'fair':2, 'pref':1, 'unavail':5, 'rest':3}  # NOTE Must be integers

# Construct empty calendar with holidays etc.
emptyCalendar(end_date, start_date, prints=False)
empty_calendar_df = pd.read_csv(f'data/intermediate/empty_calendar.csv') # reading from file converts Datetime objects to str dates

# Automatically generate 'shift_data.csv'
generateShiftData(empty_calendar_df, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)
generatePhysicianData(empty_calendar_df,n_physicians,seed=True)
phys_df = pd.read_csv('data/intermediate/physician_data.csv')
#print('\nPHYS DF',phys_df)
convertPreferences(empty_calendar_df)


shifts_df = pd.read_csv(f'data/intermediate/shift_data.csv')
n_shifts=len(shifts_df)
n_dates = empty_calendar_df.shape[0]
n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

if prints:
    print('\nn physicians:', n_physicians)
    print('n days:', n_dates)
    print('n shifts', n_shifts)
    print('n variables:', n_physicians*n_shifts)

# Translate unprefered dates to unprefered shift-numbers
#convertPreferences(empty_calendar_df)

# Make sum of all objective functions and enforce penatlies (lambdas)
all_objectives, x_symbols = makeObjectiveFunctions(n_demand, cl, lambdas=lambdas)

# Extract Qubo Q-matrix from objectives           Y = x^T Qx
Q = objectivesToQubo(all_objectives, x_symbols, cl, mirror=False)


# Solve using classical solvers
if classical:
    print("\nSolving with Z3 (Classical)...")
    z3_schedule, z3_solver_time, z3_overall_time = solve_and_save_results(solver_type="z3", cl=cl, lambdas=lambdas)
    if z3_schedule:
        print("Z3 schedule:")
        for p, s in z3_schedule.items():
            print(f"{p}: {s}")
        z3_schedule_df = schedule_dict_to_df(z3_schedule, shifts_df)
        z3_checked_df = controlSchedule(z3_schedule_df, shifts_df, cl=cl)

    print("\nSolving with Gurobi (Classical)...")
    gurobi_schedule, gurobi_solver_time, gurobi_overall_time = solve_and_save_results(solver_type="gurobi", cl=cl, lambdas=lambdas)
    if gurobi_schedule:
        print("Gurobi schedule:")
        for p, s in gurobi_schedule.items():
            print(f"{p}: {s}")
        gurobi_schedule_df = schedule_dict_to_df(gurobi_schedule, shifts_df)
        gurobi_checked_df = controlSchedule(gurobi_schedule_df, shifts_df, cl=cl)

    print("\n--- Timing Comparison ---")
    print(f"Z3 solver time:     {z3_solver_time:.4f} s")
    print(f"Z3 overall time:    {z3_overall_time:.4f} s")
    print(f"Gurobi solver time: {gurobi_solver_time:.4f} s")
    print(f"Gurobi overall time:{gurobi_overall_time:.4f} s")

    print("\n--- Relative Difference ---")
    print(f"Solver time difference:  {z3_solver_time - gurobi_solver_time:.4f} s")
    print(f"Overall time difference: {z3_overall_time - gurobi_overall_time:.4f} s")

    if plots:
        controlPlotDual(z3_checked_df, gurobi_checked_df)

'''
# Q-matrix --> pauli operators --> cost hamiltonian (Hc)
b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
Hc = QToHc(Q, b)

# Set up hardware
backend = AerSimulator()

# Make initial circuit
circuit = QAOAAnsatz(cost_operator=Hc, reps=n_layers) # Using a standard mixer hamiltonian
circuit.measure_all()
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend) # pass manager transpiles circuit
circuit = pass_manager.run(circuit)

# Use estimator and COBYLA to find best ß and gammas, with Hc
# initial_parameters =  np.concatenate([initial_gammas,initial_betas])
best_parameters = findParameters(n_layers, circuit, backend, Hc, estimation_iterations, search_iterations, seed=True, prints=True, plots=plots)
#pd.DataFrame(best_parameters).to_csv('data/intermediate/saved_betas_and_gammas.csv',index=False, header=False)

best_circuit = circuit.assign_parameters(parameters=best_parameters)
if draw_circuit:
    #plt.figure(figsize=(10,10))
    print(best_circuit.decompose().draw()) # output='mpl' requires pip install pylatexenc, horizontal: fold=-1
    #plt.show()

# Use sampler to find solution bitstrings
sampling_distribution = sampleSolutions(best_circuit, backend, sampling_iterations, plots=plots)
best_bitstring = findBestBitstring(sampling_distribution, Hc, n_candidates, prints=True, worst_solutions=False)

result_schedule_df = bitstringToSchedule(best_bitstring, empty_calendar_df, cl, n_shifts)
if len(result_schedule_df)!= n_shifts:
    print('\n\nERROR!!!!, row missing in solution\n')
controled_result_df = controlSchedule(result_schedule_df, shifts_df, cl, prints=True)

controlPlot(controled_result_df)

# (Evaluate & compare solution to classical methods)'''