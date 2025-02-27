from qaoa.qaoa import *
from classical.classical import *
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
#DONE less look-ups, define ex. n_shifts, n_physicians only once
#DONE Debug estimator & sampler (why 1 of each bitstring)
    # pref. levels 1-5
    # Sampler seems to choose only-0-bitstrings, error in Hc, & findParameters?
    # implement al: how many physicians
    # less loops, more paralellism
    # Remove unused code
    # decide universal way of storing list-like objects in csv
    # QAOA class instead of functions
#DONE solve deprecation warning on sampler
#DONE divide data to input, intermediate, output 

# Parameters
start_date = '2025-03-03' # including this date
end_date = '2025-03-05' # including this date
weekday_demand = 2
holiday_demand = 1
al = 1 # amount level {1: 5 physicians, 2: } #TODO decide 
cl = 1 # complexity level:
# cl1: demand, fairness
# cl2: demand, fairness, preferences
# cl3: demand, fairness, preferences, time off 
# cl4: demand, fairness, preferences, time off, shift type
# cl5: demand, fairness, preferences, time off, shift type, rest

prints = True
plots = True
classical = False
draw_circuit = False

lambda_fair = 0.5
lambda_pref = 0.5
n_layers = 3
initial_betas = [np.pi/2]*n_layers # TODO change?
initial_gammas = [np.pi/2]*n_layers  # change?

estimation_iterations = n_layers * 1000 #  seems to stop after ~100 iterations. Adjust "tol"=tolerance in findParameters
sampling_iterations = 1000

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
max_shifts_per_p = int((n_demand/n_physicians)+0.9999)  # fair distribution of shifts
print('\nMax shifts per p:', max_shifts_per_p)

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
all_hamiltonians, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambda_fair, lambda_pref) # NOTE does not handle preferences yet

# Extract Qubo Q-matrix from hamiltonians           Y = x^T Qx
Q = hamiltoniansToQuboMatrix(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='np')

# Q-matrix --> pauli operators --> cost hamiltonian (Hc)
Hc = QToHc(Q)

# Set up hardware
backend = AerSimulator() 

# Make initial circuit
circuit = QAOAAnsatz(cost_operator=Hc, reps=n_layers)
circuit.measure_all() # not needed?
pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend) # TODO replace copied settings
circuit = pass_manager.run(circuit) # -||-

# Use estimator and COBYLA  to find best ß and gammas, with Hc
initial_parameters = initial_betas + initial_gammas
best_parameters = findParameters(initial_parameters, circuit, backend, Hc, estimation_iterations, prints=True, plots=plots)

best_circuit = circuit.assign_parameters(parameters=best_parameters)

# Use sampler to find solution bitstrings
sampling_distribution = sampleSolutions(best_circuit, backend, sampling_iterations)
best_bitstring = findBestBitstring(sampling_distribution, prints=True)
result_schedule_df = bitstringToSchedule(best_bitstring, empty_calendar_df, cl, n_shifts)
controlSchedule(result_schedule_df, demand_df, cl, prints=True)





# Transpile
    # using function in qiskit
    # adapted to specific hardware
#quantumCircuit = transpileAnsatz(ansatz, backend, prints=prints)


# Use best gamma & ß to sample solutions using sampler primitive
#distribution = sampleSolutions(bestParameters, prints=prints, plots=plots)

# (Find most probable solution)
#bitstringSolution =[] 

# (Evaluate & compare solution to classical methods)

# Decode bitstring output to schedule
#schedule_df = bitstringToSchedule(bitstringSolution, encoding, prints=prints)

# Export schedule as .csv or similar
