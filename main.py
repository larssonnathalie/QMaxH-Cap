from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.testQandH import *

# General TODO:s
    # Handle titles & assignments
        # might need even shorter periods than week
        # "chef" title has special constraints & not included in fairness
    # Merge branches
    # Reduce Qubits: Work-type in constraints similar to long term fairness, OR: remove side-task work-types
    # Define "fairness", considering different titles have different types of work 
        # add more factors than preference satisfaction, ex. weekend shifts, night shifts etc
        # Optimize w.r.t. current week, not just previous OR day-to-day opt?
    # Decide lambdas
    # dont assign unavailable
    # shift type preferences

    # antal variabler max för aer
        # bestämma vilka resultat vi vill ha
            # Jämföra klassisk och kvant
            # (Jämföra klassisk(qubo) och klassisk(constraints))
            # Jämföra klassisk(qubo lång) med klassisk(qubo många korta)
            # jämför klassisk(qubo korta) med klassisk(rolling horizon)

# Parameters
start_date = '2025-03-24' 
end_date = '2025-03-26'
n_physicians = 6
backend = 'aer'
cl = 3                    # complexity level: 
cl_contents = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable, extent',
'cl3: demand, fairness, preferences, unavailable, extent, shift_type, rest',
'cl4: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles',
'cl5: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks',
'cl6: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks, competence']

skip_unavailable_and_prefer_not = True 
only_fulltime =True
prints = True
plots = True
classical = False
draw_circuit = False
preference_seed = True
init_seed = False
estimation_plots = False

time_period = 'shift'
demands = {'weekday': 2, 'holiday': 0}   # weekday should be > holiday 
if cl>= 3:    # {(shift, is_holiday): num_workers_needed, ...} Demand should be decided in consideration of the amount of workers and their extent
    demands = {('dag', False):2, ('kväll',False):2, ('natt',False):2,('dag',True):1, ('kväll',True):1, ('natt',True):1} 
n_layers = 2
search_iterations = 20
estimation_iterations = n_layers * 1000
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions
plot_width = 10

# lambdas = penalties (how hard a constraint is)
lambdas = {'demand':5, 'fair':0, 'pref':0, 'unavail':0, 'extent':10, 'rest':20}  # NOTE Must be integers

# Construct empty calendar with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
full_solution = []

# Generate random preferences
generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed, only_fulltime=only_fulltime)  

# Demand & attractiveness for each shift
generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)
shifts_per_t = getShiftsPerT(time_period, cl)

# TESTSTRING
TESTSTRING0 = '000000000'
TESTSTRING1 = '100100100'



print()
print(cl_contents[cl])
print('\nPhysicians:\t', n_physicians)
print('Days:\t\t', n_days)
print('Shifts:\t', len(all_shifts_df))
print(f't:s ({time_period}:s)\t', T)
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

    print('\nt:\t', t)

    shifts_df = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
    n_shifts = len(shifts_df)
    n_dates = calendar_df_t.shape[0] 
    n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

    if cl >=2:
        convertPreferences(calendar_df_t, t, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers

    # Make sum of all objective functions and enforce penatlies (lambdas)
    all_objectives, x_symbols = makeObjectiveFunctions(n_demand, t, T, cl, lambdas, time_period)
    n_vars = n_physicians*len(calendar_df_t)
    #print(n_vars, 'vars')
    subs0 = assignVariables('0'*n_vars, x_symbols)
    #subs1 = assignVariables(TESTSTRING1[t*n_vars:(t+1)*n_vars], x_symbols) 

    #print('Objectives(0000.. ):')
    #print(sp.simplify(all_objectives.subs(subs0)))
    #print(sp.simplify(all_objectives.subs(subs1)))
    # Extract Qubo Q-matrix from objectives           Y = x^T Qx
    Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False, prints = False)
    if t==0:
        print('\nVariables:',Q.shape[0])
    # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
    b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
    Hc = QToHc(Q, b) 

    qaoa = Qaoa(t, Hc, n_layers, plots=estimation_plots, seed=init_seed, backend=backend, instance='premium')
    qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
    best_bitstring_t = qaoa.sampleSolutions(sampling_iterations, n_candidates, return_worst_solution=False)

    print('chosen bs',best_bitstring_t[::-1])
    subs2 = assignVariables(best_bitstring_t[::-1], x_symbols)
    #print(all_objectives.subs(subs2))
    #print('Hc(0000)', costOfBitstring('0'*n_vars, Hc))
    print('Hc(best)', costOfBitstring(best_bitstring_t, Hc))


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
for w in range(1,T):
    full_schedule_df = pd.concat([full_schedule_df, full_solution[w]],axis=0)
ok_full_schedule_df = controlSchedule(full_schedule_df, all_shifts_df, cl)
#ok_full_schedule_df = pd.read_csv('data/results/result_and_demand_cl2.csv') # Use saved result
controlPlot(ok_full_schedule_df, range(T), cl, time_period, lambdas, width=plot_width) 

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

