from qaoa.qaoa import *
from classical.classical import *
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
#DONE less look-ups, define ex. n_shifts, n_physicians only once
    # move Quantum code from main to qaoa.py
    # Debug estimator & sampler (why 1 of each bitstring)
    # implement al: how many physicians
    # find error in H -> Q or Q -> QP (H seems to work)
    # less loops, more paralellism
    # decision var. names --> handle more digits 
    # decide universal way of storing list-like objects in csv
    # (QAOA class instead of functions?)
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
n_layers = 2
initial_betas = [np.pi/2]*n_layers # TODO change?
initial_gammas = [np.pi/2]*n_layers  # change?

estimation_iterations = n_layers * 1000
sampling_iterations = n_


# Construct empty calendar with holidays etc.
emptyCalendar(end_date, start_date, cl, prints=False)
empty_calendar_df = pd.read_csv(f'data/intermediate/empty_calendar_cl{cl}.csv') # reading from file converts Datetime objects to str dates

# Automatically generate demand per day based on weekday/holiday --> 'demand.csv'
generateDemandData(empty_calendar_df, cl, weekday_workers=weekday_demand, holiday_workers=holiday_demand, prints=prints)

# Get n.o. workers, shifts & total demand
demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
physician_df = pd.read_csv(f'data/input/physician_cl{cl}.csv')
n_physicians = physician_df.shape[0]
n_shifts = empty_calendar_df.shape[0] # NOTE assuming 1 shift per row
n_demand = sum(demand_df['demand']) # sum of workers demanded on all shifts
max_shifts_per_p = int((n_demand/n_physicians)+0.9999)  # fair distribution of shifts

# Classical optimization (BILP, solver: z3), for comparison
if classical:
    result_classical = classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p, prints=False)
    print('Classical:\n',result_classical)

# Translate unprefered dates to unprefered shift-numbers
if cl>1:
    generatePreferences(empty_calendar_df, cl)
else:
    physician_df.to_csv(f'data/intermediate/physician_cl{cl}.csv', index=None)


# Make, sum and simplify all hamiltonians and enforce penatlies (lambdas)
all_hamiltonians, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambda_fair, lambda_pref) # NOTE does not handle preferences yet

# Extract Qubo Q-matrix from hamiltonians           Y = x^T Qx
Q = hamiltoniansToQuboMatrix(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='np')

# Automatically make Hc from Q if type(Q) = QP
#Hc, offset = qubo_qp.to_ising()

# Q-matrix --> pauli operators --> cost hamiltonian (Hc)
Hc = QToHc(Q)

# Use estimator and classical solver to find best ß and gammas
backend = AerSimulator() 
estimator = Estimator(mode=backend,options={"default_shots": 4_000})
initial_parameters = initial_betas + initial_gammas
ansatz = QAOAAnsatz(cost_operator=Hc, reps=n_layers)
ansatz.measure_all() # not needed?
pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend) # TODO replace copied settings
circuit = pass_manager.run(ansatz) # -||-

bounds = [(0, np.pi) for _ in range(n_layers*2)] # TODO replace copied settings

result = minimize(  
    cost_func_estimator,
    initial_parameters,
    args=(circuit, Hc, estimator),
    method="COBYLA", # COBYLA is a classical OA: Constrained Optimization BY Linear Approximations
    bounds=bounds,  
    tol=1e-3,                  # TODO replace copied settings
    options={"rhobeg": 1e-1}   # -||-
)
parameters = result.x
best_circuit = circuit.assign_parameters(parameters=parameters)

# Use sampler to find best bitstrings, given the found ßs and gammas
sampler = Sampler(mode=backend, options={"default_shots": 400}) # TODO replace copied settings

# TODO circuit not transpiled? Check if needed
pub = (best_circuit,)
job = sampler.run([pub])
counts = job.result()[0].data.meas.get_counts()

# Plot parameter optimization
plt.plot(Hc_values)
plt.title('Hc costs while optimizing ßs and gammas')
plt.figure(figsize=(20,10))

# Plot solution frequency
plt.title('Solution distribution')
plt.bar([i for i in range(len(counts))], counts.values())
plt.xticks(ticks = [i for i in range(len(counts))], labels=counts.keys())
plt.xticks(rotation=90)
plt.show()
#transpiled_circuit = qiskit.transpile(qaoa_circuit, backend=backend)#, basis_gates=['u3', 'cx'])
'''if draw_circuit:
    print(transpiled_circuit.draw())

qaoa = QAOA(sampler=sampler, optimizer=cybola)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qubo_qp)
bitstring = result.variables_dict # Seems to output constraints as variables if some is broken?
result_schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl, n_shifts)

demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
controlSchedule(result_schedule_df, demand_df, cl)

# Check costs of solution (possibly an error in Q-matrix, but I can't re-generate the error)
quantum_solution = {}
classical_solution = {}
x_quantum = np.zeros(n_physicians * n_shifts)
x_classic = np.zeros(n_physicians * n_shifts)
for count, variable in enumerate(bitstring.keys()):
    x_quantum[count] = bitstring[variable]
    p,s = variableNameToPS(variable, n_shifts)
    quantum_solution[f'x{p}_{s}'] = bitstring[variable]
    if p ==0:
        if s == 0 or s==3:
            value = 1
        else:
            value=0
    elif p == 1:
        if s ==2 or s==3:
            value =1
        else:
            value=0
    elif p==2:
        if s ==0 or s==1:
            value=1
        else:
            value =0
    else:
        print('wrong p', p)
    classical_solution[f'x{p}_{s}'] = value
    x_classic[count] = value
    
print('\n Hc COST quantum =', all_hamiltonians.subs(quantum_solution))
print('\n Hc COST classical solution (test)=', all_hamiltonians.subs(classical_solution))

# print x^T Q x
Q = pd.read_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv', header=None).to_numpy()
xc, xcT = x_classic.reshape((12,1)), x_classic.reshape((1,12))
print('xTQx classic:', np.matmul(np.matmul(xcT,Q),xc))
x, xT = x_quantum.reshape((12,1)), x_quantum.reshape((1,12))
print('xTQx quantum:', np.matmul(np.matmul(xT,Q),x))'''


# Hc --> QAOA Circuit (Ansatz)
#ansatz = makeAnsatz(Hc, prints=prints)

# Set up hardware 
    # (IBM or other)
#backend = hardwareSetup(prints=prints)

# Transpile
    # using function in qiskit
    # adapted to specific hardware
#quantumCircuit = transpileAnsatz(ansatz, backend, prints=prints)

# Use estimator primitive to find best gamma and ß
    # Using initial parameter values for gamma and ß
    # Evaluate using Hc
#bestParameters = findParameters(initial_betas, initial_gammas, quantumCircuit, prints=prints, plots=plots)

# Use best gamma & ß to sample solutions using sampler primitive
#distribution = sampleSolutions(bestParameters, prints=prints, plots=plots)

# (Find most probable solution)
#bitstringSolution =[] 

# (Evaluate & compare solution to classical methods)

# Decode bitstring output to schedule
#schedule_df = bitstringToSchedule(bitstringSolution, encoding, prints=prints)

# Export schedule as .csv or similar
