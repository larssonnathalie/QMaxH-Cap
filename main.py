from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *


# General TODO:s
    # Handle titles & assignments
        # might need even shorter periods than week
        # "chef" title has special constraints & not included in fairness
    # Fix ibm sampler
    # Merge branches
    # Reduce Qubits: Work-type in constraints similar to long term fairness, OR: remove side-task work-types
    # Define "fairness", considering different titles have different types of work 
    # Memorize fairness externally and make new qubo-matrix for each week to make long term fair schedules with limited n.o. qubits
        # add more factors than preference satisfaction, ex. weekend shifts, night shifts etc.
        # Handle confusion of fairness & pref
    # Simulator for finding candidate angles, compare candidates with ibm estimator
    # Extent: workers have different percentages? 
    # Bugfix in postprocessing, missing rows in output schedules!
    # Decide lambdas
    # (remove extent)

# Parameters
start_date = '2025-03-24' # for now this should be an [int] number of weeks
end_date = '2025-04-20'
weekday_demand = 2
holiday_demand = 1
n_physicians = 3
cl = 2                      # complexity level: 
complexity_levels = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable',
'cl3: demand, fairness, preferences, unavailable, shift_type, rest',
'cl4: demand, fairness, preferences, unavailable, shift_type, rest, titles',
'cl5: demand, fairness, preferences, unavailable, shift_type, rest, titles, side_tasks',
'cl6: demand, fairness, preferences, unavailable, shift_type, rest, titles, side_tasks, competence']

prints = True
plots = True
classical = False
draw_circuit = False
preference_seed = False
init_seed = False

n_layers = 1
search_iterations = 10
estimation_iterations = n_layers * 100 
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions

# lambdas = penalties (how hard a constraint is)
lambdas = {'demand':10, 'fair':10, 'pref':1, 'unavail':10, 'rest':3}  # NOTE Must be integers

# Construct empty calendar with holidays etc.
n_weeks, total_holidays = emptyCalendar(end_date, start_date)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
full_solution = []

# Generate random preferences
#generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed)  # NOTE TESTING WITH NO PREFER NOT & NO UNAVAIL

# Deterministic generation, used multiple times for now
generateShiftData(all_dates_df, 'all', cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)

print()
print(complexity_levels[cl])
print('Physicians:\t', n_physicians)
print('Days:\t\t', len(all_dates_df))
print('Weeks\t', n_weeks)
print('Layers\t', n_layers)
print('Seeds:')
print('\tPreference\t', preference_seed)
print('\tInitialization\t', init_seed)
print()
print('search iterations:', search_iterations)
print(f'comparing top {n_candidates} most common solutions')

for week in range(n_weeks):
    #empty_calendar_df_week = pd.read_csv(f'data/intermediate/empty_calendar_week{week}.csv')
    calendar_df_week = all_dates_df.iloc[week*7:(week+1)*7]

    print('\nWEEK:\t', week)
    # Automatically generate 'shift_data.csv'
    generateShiftData(calendar_df_week, week, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)

    shifts_df = pd.read_csv(f'data/intermediate/shift_data_w{week}.csv')
    n_shifts = len(shifts_df)
    n_dates = calendar_df_week.shape[0] 
    n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

    if cl >=2:
        convertPreferences(calendar_df_week, week)   # Dates to shift-numbers

    # Make sum of all objective functions and enforce penatlies (lambdas)
    all_objectives, x_symbols = makeObjectiveFunctions(n_demand, week, n_weeks, cl, lambdas=lambdas) 
   
    # Extract Qubo Q-matrix from objectives           Y = x^T Qx
    Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False)

    # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
    b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
    Hc = QToHc(Q, b) 

    qaoa = Qaoa(week, Hc, n_layers, plots=False, seed=init_seed, backend='aer', instance='premium')
    qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
    best_bitstring_w = qaoa.sampleSolutions(sampling_iterations, n_candidates, return_worst_solution=False)
    result_schedule_df_w = bitstringToSchedule(best_bitstring_w, calendar_df_week, n_shifts)
    full_solution.append(result_schedule_df_w)
    controled_result_df_w = controlSchedule(result_schedule_df_w, shifts_df, cl)
    #print()
    #print(controled_result_df_w)
    #TODO check if result is ok (unavailable & demand met) rerun until ok if not

    if cl>=2:
        preferenceHistory(controled_result_df_w, week)
    

all_shifts_df = pd.read_csv('data/intermediate/shift_data_all_w.csv', index_col=None)
n_shifts = len(all_shifts_df)    
full_schedule_df = full_solution[0]
for w in range(1,n_weeks):
    full_schedule_df = pd.concat([full_schedule_df, full_solution[w]],axis=0)
ok_full_schedule_df = controlSchedule(full_schedule_df, all_shifts_df, cl)
#ok_full_schedule_df = pd.read_csv('data/results/result_and_demand_cl2.csv') # Use saved result

#print(ok_full_schedule_df)
controlPlot(ok_full_schedule_df, weeks=range(n_weeks), cl=cl, width=20) 

#physician_df = pd.read_csv('data/intermediate/physician_data.csv', index_col=None)
#physician_df_print = physician_df.drop(columns=['name', 'competence', 'extent', 'prefer', 'prefer not', 'unavailable'])
#print('\nFinal satisfactions:')
#print(physician_df['satisfaction'])
satisfaction_plot = np.array(satisfaction_plot)
plt.figure()
for p in range(n_physicians):
    plt.plot(satisfaction_plot[:,p], label=str(p))
plt.legend()
plt.show()
# (Evaluate & compare solution to classical methods)'''

