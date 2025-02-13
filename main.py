from qaoa.functions import *
from classical.classical import *
from preprocessing.functions import *
from postprocessing.functions import *

# General TODO:s
    # less look-ups, define ex. n_shifts, n_physicians only once
    # less loops, more paralellism
    # decision var. names --> handle more digits 
    # decide universal way of storing list-like objects in csv
    # divide data to input, intermediate, output DONE

# Parameters
start_date = '2025-02-14' # including this date
end_date = '2025-02-17' # including this date
cl = 1 # complexity level
weekday_demand = 2
holiday_demand = 1

prints = True
plots = True
preferences = False
classical = True
draw_circuit = False

# Not used yet:
lamda_fair = 0.5
lamda_pref = 0.5
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

# Import problem data as objective functions
objectives = constructObjectives(cl, n_physicians, n_shifts, max_shifts_per_p, preferences, prints=True)

if cl==1:
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
    encoding_not_used = 1 # TODO: encoding?
    result_schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl, encoding_not_used)

    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
    controlSchedule(result_schedule_df, demand_df, cl)

if cl>1:
    # Encode problem to QUBO  Y = x^T Qx
    qubo, encoding = makeQubo(objectives, lamda_fair=lamda_fair, lamda_pref=lamda_pref, prints=prints)

    # QUBO --> Cost Function Hamiltonian Hc      Y = z^T Qz + b^T z
        # x {0, 1}  --> z {-1, 1}
        # built from 
            # pauli-Z operators
            # and pauli-ZZ operators (enabling entanglement)
    Hc = makeCostHamiltonian(qubo, prints=prints)

    # Hc --> QAOA Circuit (Ansatz)
    ansatz = makeAnsatz(Hc, prints=prints)

    # Set up hardware 
        # (IBM or other)
    backend = hardwareSetup(prints=prints)

    # Transpile
        # using function in qiskit
        # adapted to specific hardware
    quantumCircuit = transpileAnsatz(ansatz, backend, prints=prints)

    # Use estimator primitive to find best gamma and ß
        # Using initial parameter values for gamma and ß
        # Evaluate using Hc
    bestParameters = findParameters(initial_betas, initial_gammas, quantumCircuit, prints=prints, plots=plots)

    # Use best gamma & ß to sample solutions using sampler primitive
    distribution = sampleSolutions(bestParameters, prints=prints, plots=plots)

    # (Find most probable solution)
    bitstringSolution =[] 

    # (Evaluate & compare solution to classical methods)

    # Decode bitstring output to schedule
    schedule_df = bitstringToSchedule(bitstringSolution, encoding, prints=prints)

    # Export schedule as .csv or similar
