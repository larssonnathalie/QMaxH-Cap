from qaoa.qaoa import *
#from classical.classical import * 
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *


# General TODO:s
    # best angles depend heavily on initialization, COBYLA finds very local optima
        # fixed for now by comparing many random initialization outcomes & taking the best
        # should we change to a global optimizer instead? Or is something wrong?
    # How maximize fairness when workers have different percentages? Focus on shift type/weekday/holidays?
    # Extent constraint: set target to correct n.o. hours/week
    # shift type, shift_data.csv as Nathalies, adapt generateDemand()
    # rest-time in objectives
    # Same data files & QUBO for classic & quantum
    # implement al: how many physicians
    # Fix universal way of storing list-like objects in csv
    # QAOA class instead of functions
    # Bugfix in postprocessing, missing rows in output schedules!

# Parameters
start_date = '2025-03-21' # including this date
end_date = '2025-03-23' # including this date
weekday_demand = 2
holiday_demand = 1
n_physicians = 3   #TODO should depend on al
#al = 1 # amount level {1: 5 physicians, 2: } #TODO decide numbers
cl = 3 # complexity level:
# cl1: demand, fairness
# cl2: demand, fairness, preferences
# cl3: demand, fairness, preferences, unavailable, extent 
# cl4: demand, fairness, preferences, unavailable, extent, shift_type
# cl5: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles
# cl6: demand, fairness, preferences, unavailable, extent, shift_type, rest, titles, side_tasks

prints = True
plots = False
classical = False
draw_circuit = False

n_layers = 2 
#np.random.seed(12)
#initial_betas = np.random.random(size=n_layers)*np.pi # Random initial angles [np.pi/2]*n_layers 
#initial_gammas = np.random.random(size=n_layers)*np.pi*2  #[np.pi]*n_layers  

#print(np.shape([np.pi/2]*n_layers ))
search_iterations = 20
estimation_iterations = n_layers * 2000 
sampling_iterations = 4000
n_candidates = 20 # compare top X most common solutions

lambda_demand = 5 # lambdas = penalties, should be integers
lambda_fair = 2
lambda_pref = 1 
lambda_extent=4

# Include relevant constraint penalties
all_constraints = ['demand', 'fair', 'pref', 'unavail', 'extent', 'shift_type', 'rest', 'titles', 'side_tasks']
all_penalties = [lambda_demand, lambda_fair, lambda_pref, lambda_extent,0,0,0,0,0]
include_idx_for_cl ={1:2, 2:3, 3:5, 4:6, 5:8, 6:9}
penalties = [0]*len(all_penalties)
penalties[:include_idx_for_cl[cl]] = all_penalties[:include_idx_for_cl[cl]]    # ensure all deactivated lambdas are 0 
lambdas = {all_constraints[i]:penalties[i] for i in range(len(all_constraints))}


# Construct empty calendar with holidays etc.
emptyCalendar(end_date, start_date, cl, prints=False)
empty_calendar_df = pd.read_csv(f'data/intermediate/empty_calendar.csv') # reading from file converts Datetime objects to str dates

# Automatically generate demand per day based on weekday/holiday --> 'demand.csv'
generateDemandData(empty_calendar_df, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)
generatePhysicianData(empty_calendar_df,n_physicians,seed=False)
# Get n.o. workers, shifts & total demand
demand_df = pd.read_csv(f'data/intermediate/demand.csv')
#physician_df = pd.read_csv(f'data/input/physician_data.csv') # TODO add "usecols=" depending on cl 
#physician_df = physician_df.iloc[:n_physicians,:] # only use n_physician rows 
#physician_df.to_csv(f'data/intermediate/physician_data.csv', index=None)

n_shifts = empty_calendar_df.shape[0] # NOTE assuming 1 shift per row
n_demand = sum(demand_df['demand']) # sum of workers demanded on all shifts
#max_shifts_per_p = int((n_demand/n_physicians)+0.9999)  # fair distribution of shifts

if prints:
    print('\nn physicians:', n_physicians)
    print('n shifts:', n_shifts)
    print('n variables:', n_physicians*n_shifts)
    #print('Max shifts per p:', max_shifts_per_p,'\n')

# Classical optimization (BILP, solver: z3), for comparison
if classical:
    #result_classical = classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p, prints=False)
    #print('Classical (z3):\n',result_classical)
    pass

# Translate unprefered dates to unprefered shift-numbers
convertPreferences(empty_calendar_df)

# Make sum of all objective functions and enforce penatlies (lambdas)
all_objectives, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambdas=lambdas) 

# Extract Qubo Q-matrix from objectives           Y = x^T Qx
Q = objectivesToQubo(all_objectives, n_physicians, n_shifts, x_symbols, cl, output_type='np', mirror=False)

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
#initial_parameters =  np.concatenate([initial_gammas,initial_betas])
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
#best_cost = costOfBitstring(best_bitstrings[0], Hc) # TODO take best solution instead of first
#print('\nHc', best_cost)

result_schedule_df = bitstringToSchedule(best_bitstring, empty_calendar_df, cl, n_shifts)
if len(result_schedule_df)!= n_shifts:
    print('ERROR!!!!, row missing in solution')
result_ok_df = controlSchedule(result_schedule_df, demand_df, cl, prints=True)
controlPlot(result_ok_df)
# (Evaluate & compare solution to classical methods)'''

