from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.testQandH import *
import json
# General TODO:s
    # Decide lambdas
    # Evaluate each constraint from bitstring

    # results
        # Classical(constr.) vs quantum(qubo)
        # quantum simulator vs quantum ibm
        # quantum sim vs quantum ibm vs "random guess" for many qubits

use_qaoa = True
use_classical = False

increasing_qubits = False

# Parameters
start_date = '2025-06-01' 
end_date = '2025-06-03'
n_physicians = 4
backend = 'aer'
cl = 3               # complexity level: 
cl_contents = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable, extent',
'cl3: demand, fairness, preferences, unavailable, extent, titles']

skip_unavailable_and_prefer_not = False 
only_fulltime = False
draw_circuit = False
preference_seed = 10
plot_width = 20

time_period = 'day'
if use_classical:
    time_period = 'all'

# Quantum params
n_layers = 2
search_iterations = 20
estimation_iterations = 1000
sampling_iterations = 4000
n_candidates = 50 # compare top X most common solutions
init_seed = True
estimation_plots = False

if increasing_qubits:
    estimation_iterations = 4000
    time_period = 'all'
    start_date = '2025-06-22'
    end_date = '2025-06-28'
    sampling_iterations = 100000
    n_physicians = 3            # 3, 4, 5, 6, 7, 10, 14, 17, 21

# LAMBDAS = penalties (how hard a constraint is)
lambdas = {'demand':3, 'fair':10, 'pref':5, 'unavail':15, 'extent':8, 'rest':0, 'titles':8, 'memory':3}  # NOTE Must be integers

# Construct empty CALENDAR with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)
all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
full_solution = []

# PHYSICIAN
# preferences, titles etc.
generatePhysicianData(all_dates_df, n_physicians, cl, seed=preference_seed, only_fulltime=only_fulltime)  
physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')

# DEMAND  
if shiftsPerWeek(cl)==7:    
    # Set from amount of workers and their extent
    target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) 
    target_n_shifts_total = target_n_shifts_total_per_week * (len(all_dates_df) / shiftsPerWeek(cl))

    demand_hd = max(target_n_shifts_total_per_week//12, 1)
    demand_wd = max((target_n_shifts_total_per_week - 2*demand_hd)//5, 1)
    demands = {'weekday': demand_wd, 'holiday': demand_hd}  
    print('demands:', demands)

# SHIFTS
generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)



# CLASSICAL
if use_classical: # TODO Store the results from classical
    from classical.scheduler import * 
    from classical.gurobi_model import * 
    from classical.data_handler import *
    from classical.z3_model import *

    print()
    print(cl_contents[cl])
    print('Lambdas:', lambdas)
    print('\nPhysicians:\t', n_physicians)
    print('Days:\t\t', n_days)
    print('Seed preference:', preference_seed)    
    
    print('\nOptimizing schedule using Classical methods')

    demands = {'weekday':2, 'holiday':1} # TODO make same demands for classical & Q when extent works
    generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
    all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)

    t= 0 # Only 1 optimization
    convertPreferences(all_shifts_df, t, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers

    shifts_df = all_shifts_df
    plots = True
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

    # Hc COST OF SOLUTIONS
    z3_bitstring = scheduleToBitstring(z3_checked_df, n_physicians)
    gurobi_bitstring = scheduleToBitstring(gurobi_checked_df, n_physicians)

    Hc_full = generateFullHc(demands, cl, lambdas, all_shifts_df, makeObjectiveFunctions, objectivesToQubo, QToHc)

    z3_Hc_cost = computeHcCost(z3_bitstring, Hc_full, costOfBitstring)
    gurobi_Hc_cost = computeHcCost(gurobi_bitstring, Hc_full, costOfBitstring)

    # TODO USE EVALUATOR class 
    # (something like this:)
    '''
    z3_evaluator = Evaluator(z3_checked_df, cl, time_period, lambdas)
    z3_evaluator.makeResultMatrix()
    z3_constraint_scores = z3_evaluator.evaluateConstraints(T)
    print(z3_constraint_scores) # Save these in some file
    fig = z3_evaluator.controlPlot(width=10)

    # and same for gurobi...
    '''

    # SAVE RESULT DATA
    timestamp = time.time()
    incr_str = '/increasing_qubits' if increasing_qubits else ''
    gurobi_data = {'Hc full':gurobi_Hc_cost, 'bitstring':gurobi_bitstring, 'demands':demands,'lambdas':str(lambdas)}
    with open(f'data/results{incr_str}/schedules/gurobi_{n_physicians}phys_cl{cl}_time{timestamp}.json', "w") as f:
            json.dump(gurobi_data, f)
            f.close()

    z3_data = {'Hc full':z3_Hc_cost, 'bitstring':z3_bitstring, 'demands':demands, 'lambdas':str(lambdas)}
    with open(f'data/results{incr_str}/schedules/z3_{n_physicians}phys_cl{cl}_time{timestamp}.json', "w") as f:
            json.dump(z3_data, f)
            f.close()

    #Save prefs and extents etc
    physician_df.to_csv(f'data/results{incr_str}/physician/classical_time{int(timestamp)}.csv', index=None)

all_sampler_ids, all_times = [], []



# QUANTUM OPTIMIZATION: QAOA
if use_qaoa:
    from qaoa.qaoa import *
    from qaoa.testQandH import *
    
    print()
    print(cl_contents[cl])
    print('Lambdas:', lambdas)
    print('\nPhysicians:\t', n_physicians)
    print('Days:\t\t', n_days)
    print('Seed preference:', preference_seed) 
    print('\nOptimizing schedule using QAOA')
    if increasing_qubits:
        print('Increasing qubits')
    print('Estimation initializations:', search_iterations)
    print('Initialization seed:', init_seed)
    print(f'comparing top {n_candidates} most common solutions')
    print(f't:s ({time_period}:s)\t', T)
    print('Layers\t\t', n_layers)

    start_time = time.time()
    if shiftsPerWeek(cl)==7:    
        # DEMAND  # TODO make same demands for classical & Q
        # set from amount of workers and their extent
        target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) 
        target_n_shifts_total = target_n_shifts_total_per_week * (len(all_dates_df) / shiftsPerWeek(cl))

        demand_hd = max(target_n_shifts_total_per_week//12, 1)
        demand_wd = max((target_n_shifts_total_per_week - 2*demand_hd)//5, 1)
        demands = {'weekday': demand_wd, 'holiday': demand_hd}  
        print('demands:', demands)

    generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
    all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)
    

    shifts_per_t = getShiftsPerT(time_period, cl, n_shifts=len(all_shifts_df))   

    for t in range(T):
        calendar_df_t = all_dates_df.iloc[t*shifts_per_t: min((t+1)*shifts_per_t, len(all_shifts_df))]

        print(f'\nt:\t{t}/{T}')

        shifts_df = pd.read_csv(f'data/intermediate/shift many t/shift_data_t{t}.csv')
        n_shifts = len(shifts_df)
        n_dates = calendar_df_t.shape[0] 
        #n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

        if cl >=2:
            convertPreferences(shifts_df, t, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers
        
        # OBJECTIVES
        all_objectives, x_symbols = makeObjectiveFunctions(demands, t, T, cl, lambdas, time_period, prints=False)
        n_vars = n_physicians*len(calendar_df_t)
        subs0 = assignVariables('0'*n_vars, x_symbols)

        # QUBO MATRIX          Y = x^T Qx
        Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False, prints = False)

        #if t==0:
        #   print('\nVariables:',Q.shape[0])

        # COST HAMILTONIAN
        # Q-matrix --> pauli operators --> Hc
        b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
        Hc = QToHc(Q, b) 
        #for i in range(len(Hc.coeffs)):
        # print(Hc.paulis[i], Hc.coeffs[i])

        # RUN QAOA
        qaoa = Qaoa(t, Hc, n_layers, plots=estimation_plots, seed=init_seed, backend=backend, instance='premium')
        qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
        best_bitstring_t = qaoa.samplerSearch(sampling_iterations, n_candidates, return_worst_solution=False)
        if increasing_qubits:
            plt.figure()
            counts, bins = qaoa.costCountsDistribution(start_time, n_physicians)
            plt.show()
        print('chosen bs',best_bitstring_t[::-1])
        
        # SAVE RUNS
        #all_sampler_ids.append(qaoa.sampler_id)
        all_times.append(qaoa.end_time - qaoa.start_time)
        
        #print('Hc(best)', costOfBitstring(best_bitstring_t, Hc))
        #print('xT Q x(best)', get_xT_Q_x(best_bitstring_t, Q))

        #print('Hc(0000)', costOfBitstring('0'*n_vars, Hc))
        #print('xT Q x(0000)', get_xT_Q_x('0'*n_vars, Q))

        # GET SCHEDULE
        result_schedule_df_t = bitstringToSchedule(best_bitstring_t, calendar_df_t)
        full_solution.append(result_schedule_df_t)
        controled_result_df_t = controlSchedule(result_schedule_df_t, shifts_df, cl)
        #print('result schedule')
        #print(controled_result_df_t)

        if cl>=2:
            recordHistory(controled_result_df_t, t,cl, time_period)
        

    all_shifts_df = pd.read_csv('data/intermediate/shift_data_all_t.csv', index_col=None)
    n_shifts = len(all_shifts_df)

    # GET FULL SCHEDULE    
    full_schedule_df = full_solution[0]
    for t in range(1,T):
        full_schedule_df = pd.concat([full_schedule_df, full_solution[t]],axis=0)
    ok_full_schedule_df = controlSchedule(full_schedule_df, all_shifts_df, cl)
    print(ok_full_schedule_df)

    end_time = time.time()
    incr_str = '/increasing_qubits' if increasing_qubits else ''

    # EVALUATE
    qaoa_evaluator = Evaluator(ok_full_schedule_df, cl, time_period, lambdas)
    qaoa_evaluator.makeResultMatrix()
    constraint_scores = qaoa_evaluator.evaluateConstraints(T)
    print(constraint_scores)
    fig = qaoa_evaluator.controlPlot(width=10, show_plot=False)

    # PLOT SCHEDULE
    #fig = controlPlot(ok_full_schedule_df, range(T), cl, time_period, lambdas, width=plot_width) 
    fig.savefig(f'data/results{incr_str}/plots/{backend}_{n_physicians}phys_time{int(start_time)}.png')

    # Hc full
    qaoa_bitstring = scheduleToBitstring(full_schedule_df,n_physicians)
    Hc_full = generateFullHc(demands, cl, lambdas, all_shifts_df, makeObjectiveFunctions, objectivesToQubo, QToHc)
    qaoa_Hc_cost = computeHcCost(qaoa_bitstring, Hc_full, costOfBitstring)
    
    # SAVE RUNS
    #run_data_per_t = pd.DataFrame({'sampler id:s':all_sampler_ids, 'time':all_times })
    #run_data_per_t.to_csv(f'data/results{incr_str}/runs/{backend}_{n_physicians}phys_cl{cl}_time{int(start_time)}.csv', index=None)
    if not increasing_qubits:
        run_data_full_dict = {'full time':end_time-start_time, 'Hc full':qaoa_Hc_cost, 'bitstring':qaoa_bitstring, 'demands':demands, 'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas, 'constraints':constraint_scores}
    if increasing_qubits:
        run_data_full_dict = {'best params':qaoa.params_best[0], 'best params cost':qaoa.params_best[1], 'demands':demands,'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas}

    with open(f'data/results{incr_str}/runs/{backend}_full_{n_physicians}phys_time{int(start_time)}.json', "w") as f:
        json.dump(run_data_full_dict, f)
        f.close()


    # SAVE EXTENT & PREF
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    physician_df.to_csv(f'data/results{incr_str}/physician/{backend}_time{int(start_time)}.csv', index=None)

    # SAVE RESULTS
    schedule_data = pd.DataFrame({'date':full_schedule_df['date'], 'staff':full_schedule_df['staff']})
    schedule_data.to_csv(f'data/results{incr_str}/schedules/{backend}_{n_physicians}phys_time{int(start_time)}.csv', index=None)

    # PLOT SATISFACTION
    '''if lambdas['pref'] != 0 and T>1:
        satisfaction_plot = np.array(satisfaction_plot)
        plt.figure()
        plt.title('Preference satisfaction per time period')
        physician_df = pd.read_csv('data/intermediate/physician_data.csv', index_col=None)
        n_physicians = len(physician_df)

        for p in range(n_physicians):
            plt.plot(satisfaction_plot[:,p], label=str(p))
        plt.legend()
        plt.show()'''


