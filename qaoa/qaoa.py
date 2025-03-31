from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
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
    circuit, observable, params = pub  
    #job = estimator.run([circuit], observables=[observable], parameter_values=[params])
    job = estimator.run([pub])
    results = job.result()[0] # why does this take time
    cost = results.data.evs

    Hc_values.append(cost) # NOTE global list

    return cost

def findParameters(n_layers, circuit, backend, Hc, estimation_iterations, search_iterations, backend_name, service, seed=True, prints=True, plots=True): # TODO what job mode? (single, session, etc)
    if backend_name == 'aer':
        estimator = Estimator(mode=backend,options={"default_shots": estimation_iterations})
    bounds = [(0, 2*np.pi) for _ in range(n_layers)] # gammas have period = 2 pi, given integer penalties
    bounds += [(0, np.pi) for _ in range(n_layers)] # betas have period = 1 pi

    candidates, costs = [],[]
    for i in range(search_iterations):
        if seed:
            np.random.seed(i*10)
        initial_betas = np.random.random(size=n_layers)*np.pi # Random initial angles 
        initial_gammas = np.random.random(size=n_layers)*np.pi*2   
        initial_parameters = np.concatenate([initial_gammas, initial_betas])
        with Session(backend=backend) as session:
            if backend_name == 'ibm':
                estimator = Estimator(mode=session, options={"default_shots": estimation_iterations})
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
            costs.append(estimateHc(result.x, circuit, Hc, estimator))
    
    #print('costs',costs) 
    #print('min',costs[np.argmin(costs)])
    parameters = candidates[np.argmin(costs)]
    #print('COBYLA:', parameters)
    if plots:
        plt.figure()
        plt.plot(Hc_values)
        plt.title('Hc costs while optimizing ßs and gammas')
        plt.show()
    #if prints:
        #print('\nBest parameters (ß:s & gamma:s):', parameters)
        print('Estimated cost of best parameters', estimateHc(parameters, circuit, Hc, estimator))
        print('Estimator iterations', len(Hc_values))
    Hc_values.clear()

    return parameters

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
    def __init__(self, week, Hc:np.ndarray, n_layers:int, seed:bool, plots:bool, backend:str='ibm', instance:str='open'):
        self.Hc = Hc
        self.n_layers = n_layers
        self.seed = seed
        self.plots = plots
        self.backend_name = backend

        # Initialize backend
        if backend == 'ibm':
            if instance=='premium':
                instance='wacqt/partners/scheduling-of-me' # maybe add me"dical-doctors"
            else:
                instance = 'ibm-q/open/main'
            token = open('../token.txt').readline().strip()

            service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance=instance,
            token=token)
            self.service =service
            self.backend = service.least_busy(min_num_qubits=127)

        else:
            self.backend = AerSimulator() 
            self.service=''
        if week == 0:
            print('\nQuantum backend set to:', self.backend)
    
    def findOptimalCircuit(self, estimation_iterations=2000, search_iterations=20):
        # Make initial circuit
        circuit = QAOAAnsatz(cost_operator=self.Hc, reps=self.n_layers) # Using a standard mixer hamiltonian 
        circuit.measure_all() 
        pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend) # pass manager transpiles circuit
        circuit = pass_manager.run(circuit) 

        # Find best betas and gammas using estimator on initial circuit
        best_parameters = findParameters(self.n_layers, circuit, self.backend, self.Hc, estimation_iterations, search_iterations, self.backend_name, self.service, seed=self.seed, plots=self.plots)
        best_circuit = circuit.assign_parameters(parameters=best_parameters)
        self.optimized_circuit = best_circuit
    
    def sampleSolutions(self, sampling_iterations=4000, n_candidates=20, return_worst_solution=False):
        sampling_distribution = sampleSolutions(self.optimized_circuit, self.backend, sampling_iterations, plots=self.plots)
        best_bitstring = findBestBitstring(sampling_distribution, self.Hc, n_candidates, worst_solution=return_worst_solution)
        return best_bitstring
