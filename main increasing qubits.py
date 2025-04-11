from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.testQandH import *

all_Hc = []
all_n_vars = []

# Parameters
start_date = '2025-03-24' 
end_date = '2025-03-30'
#n_physicians = 3
backend = 'aer'
cl = 2                 # complexity level: 
cl_contents = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable, extent, (one shift per day)',
'cl3: demand, fairness, preferences, unavailable, extent, shift_type, rest, (3 shifts per day)',
'cl4: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, (3 shifts per day)',
'cl5: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks, (3 shifts per day)']

skip_unavailable_and_prefer_not = False 
only_fulltime = False
preference_seed = True
init_seed = True
estimation_plots = True

time_period = 'week' # NOTE work extent constraint is very different if t = 'week' 

n_layers = 2
search_iterations = 20
estimation_iterations = n_layers * 500
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions
plot_width = 10

# lambdas = penalties (how hard a constraint is) 
lambdas = {'demand':5, 'fair':8, 'pref':5, 'unavail':10, 'extent':0, 'rest':10}  # NOTE Must be integers
# NOTE 'fair' -> 'pref' if T =1                              if t='week': extent is not fit for n_days < 7

# Construct empty calendar with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)


print()
print(cl_contents[cl])
#print(f't:s ({time_period}:s)\t', T)
print('Layers\t', n_layers)
print('Seeds:\t\tPreference:', preference_seed,'\t\tInitialization:', init_seed)
print('Initializations:', search_iterations)
print(f'comparing top {n_candidates} most common solutions')

for n_physicians in [3,4]:
    # loop START
    print('\nPhysicians:\t', n_physicians)
    print('dates:\t', start_date, 'to:', end_date)
    print('Days:\t\t', n_days)

    # PHYSICIAN
    generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed, only_fulltime=only_fulltime)  
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')

    # DEMAND
    target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) #NOTE assuming 7 days!!
    print('target per week',target_n_shifts_total_per_week)
    target_n_shifts_total = target_n_shifts_total_per_week*(n_days/7)
    print('target total',target_n_shifts_total)
    print(physician_df['extent'])

    demand_hd = max(target_n_shifts_total_per_week//12,1)
    demand_wd = max((target_n_shifts_total_per_week - 2*demand_hd)//5,1)
    demands = {'weekday': demand_wd, 'holiday': demand_hd}   # weekday should be > holiday 
    print('demands:', demands)

    # SHIFT
    generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
    all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)
    shifts_per_t = getShiftsPerT(time_period, cl)
    calendar_df = all_dates_df

    shifts_df = all_shifts_df
    n_shifts = len(shifts_df)
    n_dates = calendar_df.shape[0] 
    n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

    if cl >=2:
        convertPreferences(calendar_df, 0, only_prefer=skip_unavailable_and_prefer_not)   # Dates to shift-numbers

    # Make sum of all objective functions and enforce penatlies (lambdas)
    all_objectives, x_symbols = makeObjectiveFunctions(n_demand, 0, T, cl, lambdas, time_period, prints=False)
    n_vars = n_physicians*len(calendar_df)
    subs0 = assignVariables('0'*n_vars, x_symbols)

    # Extract Qubo Q-matrix from objectives           Y = x^T Qx
    Q = objectivesToQubo(all_objectives, n_shifts, x_symbols, cl, mirror=False, prints = False)
    n_vars = Q.shape[0]
    print('\nVariables:',n_vars)
    all_n_vars.append(n_vars)

    # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
    b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
    Hc = QToHc(Q, b) 
    #for i in range(len(Hc.coeffs)):
        #print(Hc.paulis[i], Hc.coeffs[i])

    qaoa = Qaoa(0, Hc, n_layers, plots=estimation_plots, seed=init_seed, backend=backend, instance='premium')
    qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
    best_bitstring_t = qaoa.sampleSolutions(sampling_iterations, n_candidates, return_worst_solution=False)
    print('chosen bs',best_bitstring_t[::-1])
    final_cost = costOfBitstring(best_bitstring_t, Hc)
    print('Hc:', final_cost)
    all_Hc.append(final_cost)

    result_schedule_df = bitstringToSchedule(best_bitstring_t, calendar_df)
    controled_result_df = controlSchedule(result_schedule_df, shifts_df, cl)
    print(controled_result_df)  
    recordHistory(result_schedule_df, 0, cl, time_period)
    fig = controlPlot(controled_result_df, range(T), cl, time_period, lambdas, width=plot_width) 
    fig.savefig(f'data/results/increasing_qubits/{backend}-backend_{n_vars}vars_cl{cl}.png')

save_results = pd.DataFrame({'Hc:s':all_Hc, 'variables':all_n_vars})
save_results.to_csv(f'data/results/increasing_qubits/{backend}-backend_{n_vars}vars_cl{cl}.csv')
