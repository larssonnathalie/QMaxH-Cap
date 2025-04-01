from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
    # Handle titles & assignments
        # might need even shorter periods than week
        # "chef" title has special constraints & not included in fairness
    # n_weeks --> T,    w -> t
    # Fix ibm sampler
    # Merge branches
    # Reduce Qubits: Work-type in constraints similar to long term fairness, OR: remove side-task work-types
    # Define "fairness", considering different titles have different types of work 
    # Memorize fairness externally and make new qubo-matrix for each week to make long term fair schedules with limited n.o. qubits
        # add more factors than preference satisfaction, ex. weekend shifts, night shifts etc
        # Optimize w.r.t. current week, not just previous OR day-to-day opt?
    # Simulator for finding candidate angles, compare candidates with ibm estimator
    # Extent: workers have different percentages 
    # Decide lambdas
    # Demand for 3-shift
    # cl1 compatibility
    # dont assign unavailable
    # Extent --> target over time
    # shift type preferences
    # (remove extent)

# Parameters
start_date = '2025-03-24' # for now this should be an [int] number of weeks
end_date = '2025-03-30'
weekday_demand = 2
holiday_demand = 1
n_physicians = 4
cl = 2                      # complexity level: 
cl_contents = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable, extent',
'cl3: demand, fairness, preferences, unavailable, extent, shift_type, rest',
'cl4: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles',
'cl5: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks',
'cl6: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks, competence']

skip_unavailable_and_prefer_not = True
prints = True
plots = True
classical = False
draw_circuit = False
preference_seed = False
init_seed = False

time_period = 'day'
n_layers = 1
search_iterations = 10
estimation_iterations = n_layers * 100 
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions

# lambdas = penalties (how hard a constraint is)
lambdas = {'demand':10, 'fair':1, 'pref':1, 'unavail':1, 'extent':10, 'rest':5}  # NOTE Must be integers

# Construct empty calendar with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
full_solution = []

# Generate random preferences
generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed)  # NOTE TESTING WITH NO PREFER NOT & NO UNAVAIL

# Demand & attractiveness for each shift
generateShiftData(all_dates_df, T, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, time_period=time_period)
all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)
shifts_per_t = getShiftsPerT(time_period, cl)

print()
print(cl_contents[cl])
print('\nPhysicians:\t', n_physicians)
print('Days:\t\t', n_days)
print('Shifts:\t', len(all_shifts_df))
print(time_period+':s\t', T)
print('Shifts per t:', shifts_per_t)
print('Layers\t', n_layers)
print('Seeds:')
print('\tPreference\t', preference_seed)
print('\tInitialization\t', init_seed)
print()
print('search iterations:', search_iterations)
print(f'comparing top {n_candidates} most common solutions')


for t in range(T):
    #empty_calendar_df_t = pd.read_csv(f'data/intermediate/empty_calendar_t{t}.csv')
    calendar_df_t = all_dates_df.iloc[t*shifts_per_t: min((t+1)*shifts_per_t, len(all_shifts_df))]

    print('\ttime period (t):\t', t)

    shifts_df = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
    n_shifts = len(shifts_df)
    n_dates = calendar_df_t.shape[0] 
    n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

    if cl >=2:
        convertPreferences(calendar_df_t, t, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers

    # Make sum of all objective functions and enforce penatlies (lambdas)
    all_objectives, x_symbols = makeObjectiveFunctions(n_demand, t, T, cl, lambdas, time_period) 
   
    # Extract Qubo Q-matrix from objectives           Y = x^T Qx
    Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False)

    # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
    b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
    Hc = QToHc(Q, b) 

    qaoa = Qaoa(t, Hc, n_layers, plots=False, seed=init_seed, backend='aer', instance='premium')
    qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
    best_bitstring_t = qaoa.sampleSolutions(sampling_iterations, n_candidates, return_worst_solution=False)

    result_schedule_df_t = bitstringToSchedule(best_bitstring_t, calendar_df_t)
    full_solution.append(result_schedule_df_t)
    controled_result_df_t = controlSchedule(result_schedule_df_t, shifts_df, cl)
    #TODO check if result is ok (unavailable & demand met) rerun until ok if not
    print('result schedule')
    print(controled_result_df_t)

    if cl>=2:
        preferenceHistory(controled_result_df_t, t)
    

all_shifts_df = pd.read_csv('data/intermediate/shift_data_all_t.csv', index_col=None)
n_shifts = len(all_shifts_df)    
full_schedule_df = full_solution[0]
for w in range(1,T):
    full_schedule_df = pd.concat([full_schedule_df, full_solution[w]],axis=0)
ok_full_schedule_df = controlSchedule(full_schedule_df, all_shifts_df, cl)
#ok_full_schedule_df = pd.read_csv('data/results/result_and_demand_cl2.csv') # Use saved result
controlPlot(ok_full_schedule_df, range(T), cl, time_period, width=4) 

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

