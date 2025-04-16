from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.testQandH import *
from collections import Counter
import time

all_counted_costs = []
all_n_vars = []
all_times = []
all_counted_costs_random =[]

# Parameters
start_date = '2025-06-04' 
end_date = '2025-06-06'
#n_physicians = 3
backend = 'aer'
cl = 3                 # complexity level: 

skip_unavailable_and_prefer_not = False 
only_fulltime = False
preference_seed = False
init_seed = True
estimation_plots = False

time_period = 'week' # NOTE work extent constraint is very different if t = 'week' 

n_layers = 2
search_iterations = 20
estimation_iterations = n_layers * 500
sampling_iterations = 20000
n_candidates = 20 # compare top X most common solutions
plot_width = 15

# lambdas = penalties (how hard a constraint is) 
lambdas = {'demand':5, 'fair':5, 'pref':5, 'unavail':5, 'extent':8, 'rest':0, 'titles':10}  # NOTE Must be integers
# NOTE 'fair' -> 'pref' if T =1                              if t='week': extent is not fit for n_days < 7

# Construct empty calendar with holidays etc.
T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)


print()
print('cl:', cl)
#print(f't:s ({time_period}:s)\t', T)
print('Layers\t', n_layers)
print('Seeds:\t\tPreference:', preference_seed,'\t\tInitialization:', init_seed)
print('Initializations:', search_iterations)
print(f'comparing top {n_candidates} most common solutions')

def generateRandomSolutions(n_vars, sampling_iterations):
    all_solutions = []
    for s in range(sampling_iterations):
        solution = ''
        for var in range(n_vars):
            if np.random.random()>0.5:
                solution += '1'
            else:
                solution += '0'
        all_solutions.append(solution)
    solution_distribution = dict(Counter(all_solutions))
    
    return solution_distribution

for n_physicians in [3]:
    start_time = time.time()

    # loop START
    print('\nPhysicians:\t', n_physicians)
    print('dates:\t', start_date, 'to:', end_date)
    print('Days:\t\t', n_days)

    # PHYSICIAN
    generatePhysicianData(all_dates_df, n_physicians,cl, seed=preference_seed, only_fulltime=only_fulltime)  
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')

    # DEMAND
    target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) #NOTE assuming 7 days!!
    #print('target per week',target_n_shifts_total_per_week)
    target_n_shifts_total = target_n_shifts_total_per_week*(n_days/7)
    #print('target total',target_n_shifts_total)
    #print(physician_df['extent'])

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
    best_bitstring_t = qaoa.samplerSearch(sampling_iterations, n_candidates, return_worst_solution=False)
    

    plt.subplot(2,1,1)
    plt.title('Sampling distributions \n# vars = '+str(n_vars))
    counted_costs = qaoa.costCountsDistribution() # vars ='+str(n_vars))
    plt.legend()

    all_counted_costs.append(counted_costs)

    plt.subplot(2,1,2)
    random_distribution = generateRandomSolutions(n_vars, sampling_iterations)
    counted_costs_random = qaoa.costCountsDistribution(sampling_distribution=random_distribution, random=True)
    all_counted_costs_random.append(counted_costs_random)

    plt.legend()
    plt.show()
    final_cost = costOfBitstring(best_bitstring_t, Hc)
    print('chosen bs',best_bitstring_t[::-1],'Hc:', final_cost)
    result_schedule_df = bitstringToSchedule(best_bitstring_t, calendar_df)
    controled_result_df = controlSchedule(result_schedule_df, shifts_df, cl)
    print(controled_result_df)  
    recordHistory(result_schedule_df, 0, cl, time_period)
    end_time = time.time()
    all_times.append(int(end_time - start_time))
    #fig = controlPlot(controled_result_df, range(T), cl, time_period, lambdas, width=plot_width) 
    #fig.savefig(f'data/results/increasing_qubits/{backend}-backend_{n_vars}vars_cl{cl}.png')

save_results = pd.DataFrame({'counted costs':all_counted_costs, 'random Hc':all_counted_costs_random, '# variables':all_n_vars, 'time':all_times})
save_results.to_csv(f'data/results/increasing_qubits/{backend}-backend_{n_vars}vars_cl{cl}.csv', index=None)
