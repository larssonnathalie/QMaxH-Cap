from qaoa.qaoa import *
from classical.classical import *
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *

# General TODO:s
#DONE less look-ups, define ex. n_shifts, n_physicians only once
    # find error in H -> Q or Q -> QP (H seems to work)
    # less loops, more paralellism
    # decision var. names --> handle more digits 
    # decide universal way of storing list-like objects in csv
    # (QAOA class instead of functions?)
    # solve deprecation warning on sampler
#DONE divide data to input, intermediate, output 

# Parameters
start_date = '2025-02-14' # including this date
end_date = '2025-02-17' # including this date
weekday_demand = 2
holiday_demand = 1
cl = 1                 # complexity level  
# cl1: demand, holidays, fairness, preferences
# cl2: 

automaticQuboTranslation = False
prints = True
plots = True
preferences = False
classical = True
draw_circuit = False

# Not used yet:
lambda_fair = 0.5
lambda_pref = 0.5
n_layers = 5
initial_betas = [np.pi/2]*n_layers
initial_gammas = [np.pi/2]*n_layers

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
print('demand =',n_demand)
max_shifts_per_p = int((n_demand/n_physicians)+0.9999)  # fair distribution of shifts

# Classical optimization (BILP, solver: z3), for comparison
if classical:
    result_classical = classical_optimization_z3(empty_calendar_df, demand_df, physician_df, max_shifts_per_p, prints=False)
    print('Classical:\n',result_classical)

# Translate unprefered dates to unprefered shift-numbers
if preferences:
    generatePreferences(empty_calendar_df, cl)
else:
    physician_df.to_csv(f'data/intermediate/physician_cl{cl}.csv', index=None)

if automaticQuboTranslation:
    
    # Import problem data as objective functions
    objectives = constructObjectives(cl, n_physicians, n_shifts, max_shifts_per_p, preferences, prints=True)

    qubo = objectives
    backend = Aer.get_backend('qasm_simulator') # QasmSimulator, aer_simulator, statevector_simulator
    sampler = Sampler()
    #sampler.set_options(backend)

    operator, offset = qubo.to_ising()
    cybola = COBYLA(maxiter=200)
    ansatz = QAOAAnsatz(cost_operator=operator)

    qaoa_circuit = ansatz.assign_parameters(parameters=[0.5] * ansatz.num_parameters)
    transpiled_circuit = qiskit.transpile(qaoa_circuit, backend=backend, basis_gates=['u3', 'cx'])
    if draw_circuit:
        print(transpiled_circuit.draw())

    qaoa = QAOA(sampler=sampler, optimizer=cybola )
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qubo)
    bitstring = result.variables_dict # Seems to output constraints as variables if some is broken?
    result_schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl, n_shifts)

    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
    controlSchedule(result_schedule_df, demand_df, cl)

else:
    # Make, sum and simplify all hamiltonians and enforce penatlies (lambdas)

    all_hamiltonians, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, preferences, lambda_fair, lambda_pref) # NOTE does not handle preferences yet

    # Extract Qubo Q-matrix from hamiltonians           Y = x^T Qx
    qubo_qp = hamiltoniansToQuboMatrix(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl) 
    
    # Automatically make Hc from Q
    Hc, offset = qubo_qp.to_ising()

    # Hc = makeCostHamiltonian(q_matrix, prints=prints) #NOTE used .to_ising() instead

    # TEST Q AND Hc_____________________
    backend = Aer.get_backend('qasm_simulator') # QasmSimulator, aer_simulator, statevector_simulator
    sampler = Sampler()    
    cybola = COBYLA(maxiter=200)
    ansatz = QAOAAnsatz(cost_operator=Hc)
    qaoa_circuit = ansatz.assign_parameters(parameters=[0.5] * ansatz.num_parameters)
    transpiled_circuit = qiskit.transpile(qaoa_circuit, backend=backend, basis_gates=['u3', 'cx'])
    if draw_circuit:
        print(transpiled_circuit.draw())

    qaoa = QAOA(sampler=sampler, optimizer=cybola )
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
    print('xTQx quantum:', np.matmul(np.matmul(xT,Q),x))

    #___________END TEST________________
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
