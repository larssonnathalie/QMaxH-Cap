from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.testQandH import *

# General TODO:s
    # Decide lambdas

    # results
        # Classical(constr.) vs quantum(qubo)
        # quantum simulator vs quantum ibm
        # quantum sim vs quantum ibm vs "random guess" for many qubits


# Parameters
start_date = '2025-06-01' 
end_date = '2025-06-28'
n_physicians = 10
backend = 'aer'
cl = 3               # complexity level: 
cl_contents = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable, extent',
'cl3: demand, fairness, preferences, unavailable, extent, titles']

skip_unavailable_and_prefer_not = False 
only_fulltime = False
use_qaoa = True
use_classical = False
draw_circuit = False
preference_seed = True
init_seed = False
estimation_plots = False
compare_Hc_costs = True

time_period = 'day' # NOTE work extent constraint is very different if t = 'week' 

n_layers = 2
search_iterations = 15
estimation_iterations = n_layers * 500
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions
plot_width = 20

# LAMBDAS = penalties (how hard a constraint is)
lambdas = {'demand':2, 'fair':10, 'pref':5, 'unavail':10, 'extent':5, 'rest':0, 'titles':5, 'memory':3}  # NOTE Must be integers

# Construct empty CALENDAR with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
full_solution = []

# PHYSICIAN
# preferences, titles etc.
generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed, only_fulltime=only_fulltime)  
physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')


# TODO SET SAME DEMAND FOR CLASSIC AND QUANTUM
if shiftsPerWeek(cl)==7:    
    # DEMAND  # TODO make same demands for classical & Q
    # set from amount of workers and their extent
    target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) 
    target_n_shifts_total = target_n_shifts_total_per_week * (len(all_dates_df) / shiftsPerWeek(cl))

    demand_hd = max(target_n_shifts_total_per_week//12, 1)
    demand_wd = max((target_n_shifts_total_per_week - 2*demand_hd)//5, 1)
    demands = {'weekday': demand_wd, 'holiday': demand_hd}  
    print('demands:', demands)

# SHIFTS
generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)

# MAKE Hc FOR FULL PROBLEM TO EVALUATE CLASSICAL 
all_hamiltonians_full_T, x_symbols_full_T = makeObjectiveFunctions(demands, 0, 1, cl, lambdas, time_period='all')
Q_full_T = objectivesToQubo(all_hamiltonians_full_T, len(all_shifts_df),x_symbols_full_T, cl, mirror=False )
b_full_T = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
Hc_full_T = QToHc(Q_full_T, b_full_T)

print()
print(cl_contents[cl])
print('Lambdas:', lambdas)
print('\nPhysicians:\t', n_physicians)
print('Days:\t\t', n_days)
print('Seed preference:', preference_seed)    


# TODO Store the results from classical
# Solve using classical solvers
if use_classical:
    from classical.scheduler import * 
    from classical.gurobi_model import * 
    from classical.data_handler import *
    from classical.z3_model import *
    
    print('\nOptimizing schedule using Classical methods')

    #demands = {'weekday':2, 'holiday':1} # TODO make same demands for classical & Q when extent works
    #generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
    #all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)

    t=0 # Only 1 optimization
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

if use_qaoa:
    from qaoa.qaoa import *
    from qaoa.testQandH import *

    print('\nOptimizing schedule using QAOA')
    print('Estimation initializations:', search_iterations)
    print('Initialization seed:', init_seed)
    print(f'comparing top {n_candidates} most common solutions')
    print(f't:s ({time_period}:s)\t', T)
    print('Layers\t\t', n_layers)

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
    shifts_per_t = getShiftsPerT(time_period, cl)   

    for t in range(T):
        #empty_calendar_df_t = pd.read_csv(f'data/intermediate/empty_calendar_t{t}.csv')
        calendar_df_t = all_dates_df.iloc[t*shifts_per_t: min((t+1)*shifts_per_t, len(all_shifts_df))]

        print(f'\nt:\t{t}/{T}')

        shifts_df = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
        n_shifts = len(shifts_df)
        n_dates = calendar_df_t.shape[0] 
        n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

        if cl >=2:
            convertPreferences(calendar_df_t, t, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers
        # Make sum of all objective functions and enforce penatlies (lambdas)
        all_objectives, x_symbols = makeObjectiveFunctions(demands, t, T, cl, lambdas, time_period, prints=False)
        n_vars = n_physicians*len(calendar_df_t)
        subs0 = assignVariables('0'*n_vars, x_symbols)

        # Extract Qubo Q-matrix from objectives           Y = x^T Qx
        Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False, prints = False)

        #if t==0:
        #   print('\nVariables:',Q.shape[0])


        # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
        b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
        Hc = QToHc(Q, b) 
        #for i in range(len(Hc.coeffs)):
        # print(Hc.paulis[i], Hc.coeffs[i])

        qaoa = Qaoa(t, Hc, n_layers, plots=estimation_plots, seed=init_seed, backend=backend, instance='premium')
        qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
        best_bitstring_t = qaoa.samplerSearch(sampling_iterations, n_candidates, return_worst_solution=False)
        print('chosen bs',best_bitstring_t[::-1])

        print('Hc(best)', costOfBitstring(best_bitstring_t, Hc))
        print('xT Q x(best)', get_xT_Q_x(best_bitstring_t, Q))

        print('Hc(0000)', costOfBitstring('0'*n_vars, Hc))
        print('xT Q x(0000)', get_xT_Q_x('0'*n_vars, Q))

        result_schedule_df_t = bitstringToSchedule(best_bitstring_t, calendar_df_t)
        full_solution.append(result_schedule_df_t)
        controled_result_df_t = controlSchedule(result_schedule_df_t, shifts_df, cl)
        #TODO check if result is ok (unavailable & demand met) rerun until ok if not
        #print('result schedule')
        #print(controled_result_df_t)

        if cl>=2:
            recordHistory(controled_result_df_t, t,cl, time_period)
        

    all_shifts_df = pd.read_csv('data/intermediate/shift_data_all_t.csv', index_col=None)
    n_shifts = len(all_shifts_df)    
    full_schedule_df = full_solution[0]
    for t in range(1,T):
        full_schedule_df = pd.concat([full_schedule_df, full_solution[t]],axis=0)
    ok_full_schedule_df = controlSchedule(full_schedule_df, all_shifts_df, cl)
    print(ok_full_schedule_df)
    #ok_full_schedule_df = pd.read_csv('data/results/result_and_demand_cl2.csv') # Use saved result
    fig = controlPlot(ok_full_schedule_df, range(T), cl, time_period, lambdas, width=plot_width) 
    fig.savefig('data/results/schedule.png')


    if lambdas['pref'] != 0 and T>1:
        satisfaction_plot = np.array(satisfaction_plot)
        plt.figure()
        plt.title('Preference satisfaction per time period')
        physician_df = pd.read_csv('data/intermediate/physician_data.csv', index_col=None)
        n_physicians = len(physician_df)

        for p in range(n_physicians):
            plt.plot(satisfaction_plot[:,p], label=str(p))
        plt.legend()
        plt.show()


    # (Evaluate & compare solution to classical methods)'''


if compare_Hc_costs:
    if use_qaoa:
        bitstring_qaoa = scheduleToBitstring(full_schedule_df, n_physicians)
        print('Hc for QAOA:', costOfBitstring(bitstring_qaoa))
    if use_classical:
        try:
            gurobi_bitstring = scheduleToBitstring(gurobi_checked_df, n_physicians) 
            print('Hc for Gurobi:',costOfBitstring(gurobi_bitstring))
        except:
            print('No gurobi bitstring')
        try:
            z3_bitstring = scheduleToBitstring(z3_checked_df, n_physicians) 
            print('Hc for z3:',costOfBitstring(z3_bitstring))
        except:
            print('No z3 bitstring')
