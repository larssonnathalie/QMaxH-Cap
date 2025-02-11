from qaoa.functions import *
from classical.functions import *
from preprocessing.functions import *
from postprocessing.functions import *


# Parameters
start_date = '2025-02-15' # including this date
end_date = '2025-02-17' # including this date
prints = True
plots = True
preferences = False
lamda_fair = 0.5
lamda_pref = 0.5
n_layers = 5
initial_betas = [np.pi/2]*n_layers
initial_gammas = [np.pi/2]*n_layers
cl = 1 # complexity level

# Construct empty calendar with holidays etc.
emptyCalendar(end_date, start_date, cl, prints=False)
empty_calendar_df = pd.read_csv(f'data/empty_calendar_cl{cl}.csv')
# Automatically generate demand per day based on weekday/holiday --> 'demand.csv'
generateDemandData(empty_calendar_df, cl, prints=prints)

# Import problem data as objective functions
objectives = constructObjectives(empty_calendar_df, cl, preferences=False, prints=prints)

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
    #print(transpiled_circuit.draw())

    qaoa = QAOA(sampler=sampler, optimizer=cybola )
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qubo)
    #print(result)
    bitstring = result.variables_dict # Seems to output constraints as variables if some is broken?
    print(bitstring)
    encoding_not_used = 1 # TODO: encoding?
    result_schedule_df = bitstringToSchedule(bitstring, empty_calendar_df, cl, encoding_not_used)

    demand_df = pd.read_csv('data/demand_cl1.csv')
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

    # (QUBO --> classical optimization algorithm, to compare)

    # (Evaluate & compare solution to classical methods)

    # Decode bitstring output to schedule
    schedule_df = bitstringToSchedule(bitstringSolution, encoding, prints=prints)

    # Export schedule as .csv or similar
    # schedule_df.to_csv('data/final_schedule')
