from qaoa.qaoa import *
from classical.classical import *
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
    # Hc differs & all are wrong. Find error & compare to theory (Q, b etc)
        # Handle complex numbers correctly?
        # Q, b as input to p func correct?
        # balance in lambdas?
        # mirror Q and Hc or not? double values in conversion?
    # pref. levels 1-5
    # implement al: how many physicians
    # less loops, more paralellism
    # Remove unused code
    # decide universal way of storing list-like objects in csv
    # QAOA class instead of functions

# Parameters
start_date = '2025-03-07' # including this date
end_date = '2025-03-12' # including this date
weekday_demand = 2
holiday_demand = 1
al = 1 # amount level {1: 5 physicians, 2: } #TODO decide numbers
cl = 1 # complexity level:
# cl1: demand, fairness
# cl2: demand, fairness, preferences
# cl3: demand, fairness, preferences, time off 
# cl4: demand, fairness, preferences, time off, shift type
# cl5: demand, fairness, preferences, time off, shift type, rest

prints = True
plots = False
classical = False
draw_circuit = False

lambda_demand = 2 # penalties, should be integers
lambda_fair = 1
lambda_pref = 1 
n_layers = 3
initial_betas = [np.pi/2]*n_layers 
initial_gammas = [np.pi]*n_layers  

estimation_iterations = n_layers * 1000 #  seems to stop after ~100 iterations. Adjust "tol"=tolerance in findParameters
sampling_iterations = 4000

# Construct empty calendar with holidays etc.
emptyCalendar(end_date, start_date, cl, prints=False)
empty_calendar_df = pd.read_csv(f'data/intermediate/empty_calendar_cl{cl}.csv') # reading from file converts Datetime objects to str dates

# Automatically generate demand per day based on weekday/holiday --> 'demand.csv'
generateDemandData(empty_calendar_df, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=False)

# Get n.o. workers, shifts & total demand
demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
physician_df = pd.read_csv(f'data/input/physician_cl{cl}.csv')

n_physicians = physician_df.shape[0]
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
    result_classical = classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p, prints=False)
    print('Classical (z3):\n',result_classical)

# Translate unprefered dates to unprefered shift-numbers
if cl>1:
    generatePreferences(empty_calendar_df, cl)
else:
    physician_df.to_csv(f'data/intermediate/physician_cl{cl}.csv', index=None)

    
# Make, sum and simplify all hamiltonians and enforce penatlies (lambdas)
all_objectives, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambda_demand=lambda_demand, lambda_fair=lambda_fair) # NOTE does not handle preferences yet

# Extract Qubo Q-matrix from objectives           Y = x^T Qx
Q = makeQuboNew(all_objectives, n_physicians, n_shifts, x_symbols, cl, output_type='np', mirror=False)

# Q-matrix --> pauli operators --> cost hamiltonian (Hc)
# 
b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
Hc = QToHc(Q, b) 

# Set up hardware
backend = AerSimulator() 

# Make initial circuit
circuit = QAOAAnsatz(cost_operator=Hc, reps=n_layers) # Using a standard mixer hamiltonian 
circuit.measure_all() 
pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend) # pass manager transpiles circuit # TODO replace copied settings 
circuit = pass_manager.run(circuit) 

# Use estimator and COBYLA  to find best ß and gammas, with Hc
initial_parameters =  initial_gammas + initial_betas 
best_parameters = findParameters(initial_parameters, circuit, backend, Hc, estimation_iterations, prints=True, plots=plots)

best_circuit = circuit.assign_parameters(parameters=best_parameters)

# Use sampler to find solution bitstrings
sampling_distribution = sampleSolutions(best_circuit, backend, sampling_iterations, plots=plots)
best_bitstrings = findBestBitstring(sampling_distribution, prints=True)
best_cost = costOfBitstring(best_bitstrings[0], Hc)
print('\nHc',best_cost)

for bitstring in best_bitstrings:
    result_schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl, n_shifts)
    controlSchedule(result_schedule_df, demand_df, cl, prints=True)

# (Evaluate & compare solution to classical methods)

