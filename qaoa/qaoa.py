#from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_ibm_runtime import QiskitRuntimeService
#from qiskit_algorithms.optimizers import COBYLA
#from qiskit_optimization import QuadraticProgram
#from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize, brute
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qaoa.converters import *
import matplotlib.pyplot as plt
import qiskit
import pandas as pd
import numpy as np
import sympy as sp

Hc_values = []

def QToHc(Q, b):
    # Takes qubo and returns ising hamiltonian

    # Expand Qubo:
    # x^T Q x  =     ∑ᵢ Qᵢᵢxᵢ   +    ∑ᵢ<ⱼ   Qᵢⱼxᵢxⱼ        (1)
                   # ^diagonal^       ^upper half^

    # Substitution x ∈ {0,1}  --> z ∈ {-1,1}:
    # xᵢ = (1-zᵢ)/2        (zᵢ = Pauli-Z operator acting on qubit i)
    
    # Put in eq. (1) -->   xᵢ = (1-zᵢ)/2
    #                -->   xᵢxⱼ = (1-zᵢ)(1-zⱼ)/4  = 1/4(1 -zᵢ -zⱼ +zᵢzⱼ)

    # eq (1) -->        Hc = ∑ᵢ cᵢZᵢ + ∑ᵢ<ⱼ cᵢⱼZᵢZⱼ
        # where:
        # cᵢ = - Qᵢᵢ/2  -  ∑ᵢ≠ⱼ Qᵢ/4
        # cᵢⱼ = Qᵢⱼ/4

    n_vars = Q.shape[0]
    #print('n vars:\t\t',n_vars)
    pauli_list = []

    # Upper half
    for i in range(n_vars-1):    # n_vars-1 bc. skip diagonal
        for j in range(i + 1, n_vars):
            if Q[i, j] != 0:
                pauli_string = ['I'] * n_vars
                pauli_string[i], pauli_string[j] = 'Z', 'Z'     # "Z" at positions i and j
                coeff = 2 * Q[i, j] / 4                         # "*2" compensates for exclusion of lower half
                pauli_string.reverse()
                pauli_list.append(("".join(pauli_string), coeff))

    # "b"terms
    for i in range(n_vars):
        if b[i] != 0: 
            pauli_string = ['I'] * n_vars
            pauli_string[i] = 'Z'
            coeff = b[i] / 4
            pauli_string.reverse()
            pauli_list.append(("".join(pauli_string), coeff))
    
    Hc = SparsePauliOp.from_list(pauli_list)

    return Hc


def estimateHc(parameters, ansatz, hamiltonian, estimator:Estimator): 
    #print('estimation start')
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, parameters)
    
    job = estimator.run([pub])
    print('start')
    results = job.result()[0] # This takes time
    print('job done')
    cost = results.data.evs
    Hc_values.append(cost) # NOTE global list

    return cost

def findParameters(n_layers, circuit, backend, Hc, estimation_iterations, search_iterations, backend_name, instance, seed=True, prints=True, plots=True): # TODO what job mode? (single, session, etc)
    
    bounds = [(0, np.pi/2) for _ in range(n_layers)] # gammas have period = 2 pi, given integer penalties
    bounds += [(0, np.pi) for _ in range(n_layers)] # betas have period = 1 pi

    candidates, costs = [],np.zeros(search_iterations)
    for i in range(search_iterations):
        if seed:
            np.random.seed(i*10)
        initial_betas = np.random.random(size=n_layers)*np.pi # Random initial angles 
        initial_gammas = np.random.random(size=n_layers)*np.pi*2   
        initial_parameters = np.concatenate([initial_gammas, initial_betas])

        # SIMULATOR TO FIND CANDIDATES
        estimator = Estimator(mode=backend,options={"default_shots": estimation_iterations})
        result = minimize(
            estimateHc,
            initial_parameters,
            args=(circuit, Hc, estimator),
            method="COBYLA", # COBYLA is a classical OA: Constrained Optimization BY Linear Approximations
            bounds=bounds,
            tol=1e-3, #NOTE should be 1e-3 or smaller
            options={"rhobeg": 1}   # Sets initial step size (manages exploration)
            )
        candidates.append(result.x)
        costs[i] = estimateHc(result.x, circuit, Hc, estimator)
    
    found_parameters = candidates[np.argmin(costs)]

    if plots:
        plt.figure()
        plt.plot(Hc_values)
        plt.title('Estimated Hc using simulator')
        plt.show()

    Hc_values.clear()

    if backend_name == 'ibm':
        # IBM HARDWARE TO OPTIMIZE FOUND PARAMETERS WITH NOISE
        print('simulator found parameters:', found_parameters)
        print('Now initialize ibm backend with them')

        token = open('../token.txt').readline().strip()

        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance=instance,
            token=token)
        backend = service.least_busy(min_num_qubits=127)
        
        pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend) # transpiles circuit
        circuit_ibm = pass_manager.run(circuit) 
        print('\ntranspiled')

        with Session(backend=backend) as session:
            estimator = Estimator(mode=session, options={"default_shots": 100}) #NOTE TESTVALUE
            result = minimize(
                estimateHc,
                found_parameters,
                args=(circuit_ibm, Hc, estimator),
                method="COBYLA",
                bounds=bounds,
                tol=1e-1,
                options={"rhobeg": 1e-1}  
            )
            found_parameters = result.x 
           
    if plots:
        plt.figure()
        plt.plot(Hc_values)
        plt.title('Hc estimations using IBM')
        plt.show()
    #if prints:
        #print('\nBest parameters (ß:s & gamma:s):', parameters)
        print('Estimated cost of best parameters', estimateHc(found_parameters, circuit, Hc, estimator))
        print('Estimator iterations', len(Hc_values))


    return found_parameters

def sampleSolutions(best_circuit, backend, sampling_iterations, prints=True, plots=True):
    # TODO Use single job-mode?
    sampler = Sampler(mode=backend, options={"default_shots": sampling_iterations})

    pub = (best_circuit,)
    job = sampler.run([pub])
    sampling_distribution = job.result()[0].data.meas.get_counts()

    if plots:
        plt.figure(figsize=(10,8))
        plt.title('Solution distribution')
        plt.bar([i for i in range(len(sampling_distribution))], sampling_distribution.values())
        plt.xticks(ticks = [i for i in range(len(sampling_distribution))], labels=sampling_distribution.keys())
        plt.xticks(rotation=90)
        plt.show()
    if prints:
        pass
        #print('\nSampling iterations:', sum(sampling_distribution.values()))
    return sampling_distribution


def costOfBitstring(bitstring:str, Hc:SparsePauliOp):
    bitstring_z = bitstringToPauliZ(bitstring)
    cost = 0
    for pauli, coeff in zip(Hc.paulis, Hc.coeffs):
        term_value = 1
        for i, p in enumerate(pauli.to_label()):  # Convert to string like "ZZ" or "Z "
            if p == "Z":  # ignore "I" terms
                term_value *= bitstring_z[i]
        cost += coeff * term_value
    return cost

def findBestBitstring(sampling_distribution:dict, Hc, n_candidates=20, prints=False, worst_solution=False): # No prints temporary
    reverse = (worst_solution==False)
    sorted_distribution = dict(sorted(sampling_distribution.items(), key=lambda item:item[1], reverse=reverse)) #NOTE sorting might be memory expensive
    frequent_bitstrings = list(sorted_distribution.keys())[:n_candidates]
    
    costs = [costOfBitstring(bitstring, Hc) for bitstring in frequent_bitstrings]
    if worst_solution:
        best_bitstring = frequent_bitstrings[np.argmax(costs)]
    else:
        best_bitstring = frequent_bitstrings[np.argmin(costs)]

    if prints:
        #print('\nBest bitstring:', best_bitstring)
        print('best cost', costOfBitstring(best_bitstring, Hc))
    return best_bitstring


class Qaoa:
    def __init__(self, t, Hc:np.ndarray, n_layers:int, seed:bool, plots:bool, backend:str='ibm', instance:str='open'):
        self.Hc = Hc
        self.n_layers = n_layers
        self.seed = seed
        self.plots = plots
        self.backend_name = backend
        self.instance='wacqt/partners/scheduling-of-me'
        if instance =='open':
            self.instance = 'ibm-q/open/main'

        self.backend = AerSimulator() 

        if t == 0:
            if self.backend_name == 'ibm':
                print('\nUsing both simulator and ibm hardware as quantum backend')
            else:
                print('\nUsing quantum simulator')

            
    def findOptimalCircuit(self, estimation_iterations=2000, search_iterations=20):
        # Make initial circuit
        circuit = QAOAAnsatz(cost_operator=self.Hc, reps=self.n_layers) # Using a standard mixer hamiltonian 
        circuit.measure_all() 
        pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend) # pass manager transpiles circuit
        circuit = pass_manager.run(circuit) 

        # Find best betas and gammas using estimator on initial circuit
        best_parameters = findParameters(self.n_layers, circuit, self.backend, self.Hc, estimation_iterations, search_iterations, self.backend_name, self.instance, seed=self.seed, plots=self.plots)
        print('assigning parameters:', best_parameters)
        best_circuit = circuit.assign_parameters(parameters=best_parameters)
        self.optimized_circuit = best_circuit
    
    def sampleSolutions(self, sampling_iterations=4000, n_candidates=20, return_worst_solution=False):
        sampling_distribution = sampleSolutions(self.optimized_circuit, self.backend, sampling_iterations, plots=self.plots)
        best_bitstring = findBestBitstring(sampling_distribution, self.Hc, n_candidates, worst_solution=return_worst_solution)
        return best_bitstring
