from qaoa.functions import *
from Classical.functions import *
from preprocessing.functions import *
from postprocessing.functions import *

# Parameters
end_date = '2025-12-31'
prints = True
plots = True
preferences = False
lamda_fair = 0.5
lamda_pref = 0.5
n_layers = 5
initial_betas = [np.pi/2]*n_layers
initial_gammas = [np.pi/2]*n_layers


# Construct empty calendar with holidays etc.
empty_calendar_df = emptyCalendar(end_date)

# Import problem data as objective functions
objectives = constructObjectives(empty_calendar_df, lamda_fair=lamda_fair, lamda_pref=lamda_pref, preferences=False, prints=prints)

# Encode problem to QUBO  Y = x^T Qx
qubo, encoding = makeQubo(objectives, prints=prints)

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
bestParameters = findParameters(quantumCircuit, prints=prints, plots=plots)

# Use best gamma & ß to sample solutions using sampler primitive
distribution = sampleSolutions(bestParameters, prints=prints, plots=plots)

# (Find most probable solution)
bitstringSolution =[] 

# Decode bitstring output to schedule
schedule_df = decodeBitstring(bitstringSolution, encoding, prints=prints)

# (QUBO --> classical optimization algorithm, to compare)

# (Evaluate & compare solution to classical methods)

# Export calendar as .csv or similar
