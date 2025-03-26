from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
    # Handle titles & assignments
        # might need even shorter periods than week
    # Define "fairness", considering different titles have different types of work 
    # Memorize fairness externally and make new qubo-matrix for each week to make long term fair schedules with limited n.o. qubits
    # Simulator for finding candidate angles, compare candidates with ibm estimator
    # best angles depend heavily on initialization, COBYLA finds very local optima
        # fixed by comparing many random initialization outcomes & taking the best
    # How maximize fairness when workers have different percentages? 
        # Focus on fairness of shift type/weekday/holidays?
        # How adapt demand to extent?
            # Soften demand-constraint so it is a minimum but more is ok? Impossible for constant Q-matrix (without slack vars)
    # Competence constraint
    # Fix universal way of storing list-like objects in csv
    # Bugfix in postprocessing, missing rows in output schedules!

# Parameters
start_date = '2025-03-24' # for now this must be a [int] number of weeks
end_date = '2025-04-06'
weekday_demand = 2
holiday_demand = 1
n_physicians = 2
cl = 2 # complexity level:
complexity_leves = ['',
'cl1: demand, fairness',
'cl2: demand, fairness, preferences, unavailable',
'cl3: demand, fairness, preferences, unavailable, titles, competence',
'cl4: demand, fairness, preferences, unavailable, titles, competence, shift_type, rest',
'cl5: demand, fairness, preferences, unavailable, titles, competence, shift_type, rest,  side_tasks']

prints = True
plots = True
classical = False
draw_circuit = False

n_layers = 1
search_iterations = 30
estimation_iterations = n_layers * 100 
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions

# lambdas = penalties (how hard a constraint is)
lambdas = {'demand':5, 'fair':2, 'pref':1, 'unavail':5, 'rest':3}  # NOTE Must be integers

# Construct empty calendar with holidays etc.
n_weeks, total_holidays = emptyCalendar(end_date, start_date)

all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)
solution = []

# Generate random preferences
generatePhysicianData(all_dates_df, n_physicians,seed=True) 

# Deterministic generation, used multiple times for now
generateShiftData(all_dates_df, 'all', cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)

print()
print(complexity_leves[cl])
print('n physicians:\t', n_physicians)
print('n days:\t\t', len(all_dates_df))
print('n layers\t', n_layers)

for week in range(n_weeks):
    #empty_calendar_df_week = pd.read_csv(f'data/intermediate/empty_calendar_week{week}.csv')
    calendar_df_week = all_dates_df.iloc[week*7:(week+1)*7]

    print('\nWEEK:\t', week)
    print()
    # Automatically generate 'shift_data.csv'
    generateShiftData(calendar_df_week, week, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)

    shifts_df = pd.read_csv(f'data/intermediate/shift_data_w{week}.csv')
    n_shifts = len(shifts_df)
    n_dates = calendar_df_week.shape[0] 
    n_demand = sum(shifts_df['demand']) # sum of workers demanded on all shifts

    '''print('n physicians:\t', n_physicians)
    print('n days:\t', n_dates)
    print('n shifts\t', n_shifts)
    print('n variables:\t', n_physicians*n_shifts)
    print('n layers\t', n_layers)'''

    # Translate unprefered dates to unprefered shift-numbers
    convertPreferences(calendar_df_week, week)

    # Make sum of all objective functions and enforce penatlies (lambdas)
    all_objectives, x_symbols = makeObjectiveFunctions(n_demand, week, cl, lambdas=lambdas) 

    # Extract Qubo Q-matrix from objectives           Y = x^T Qx
    Q = objectivesToQubo(all_objectives,n_shifts, x_symbols, cl, mirror=False)

    # Q-matrix --> pauli operators --> cost hamiltonian (Hc)
    b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
    Hc = QToHc(Q, b) 

    qaoa = Qaoa(Hc, n_layers, True, True, backend='aer', instance='premium')
    qaoa.findOptimalCircuit(estimation_iterations=estimation_iterations, search_iterations=search_iterations)
    best_bitstring_w = qaoa.sampleSolutions(sampling_iterations, n_candidates, return_worst_solution=False)
    solution.append(best_bitstring_w)
    print()
    result_schedule_df_w = bitstringToSchedule(best_bitstring_w, calendar_df_week, n_shifts)
    controled_result_df_w = controlSchedule(result_schedule_df_w, shifts_df, cl)

    controlPlot(controled_result_df_w, week)

final_bitstring =''
for week in solution:
    for bit in week:
        final_bitstring += bit

all_shifts_df = pd.read_csv('data/intermediate/shift_data_all_w.csv', index_col=None)
n_shifts = len(all_shifts_df)    

full_result_schedule_df = bitstringToSchedule(final_bitstring, all_dates_df, n_shifts)
full_controled_result_df = controlSchedule(full_result_schedule_df, all_shifts_df, cl)
#controlPlot(controled_result_df, week) #TODO all weeks in controlPlot

# (Evaluate & compare solution to classical methods)'''

