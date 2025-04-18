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
                coeff = Q[i, j] / 4                         

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
    #print('start')
    results = job.result()[0] # This takes time
    #print('job done')
    cost = results.data.evs
    print(cost, parameters)
    Hc_values.append(cost) # NOTE global list
    return cost




def costOfBitstring(bitstring:str, Hc:SparsePauliOp):
    bitstring_z = bitstringToPauliZ(bitstring)
    cost = 0
    for pauli, coeff in zip(Hc.paulis, Hc.coeffs):
        term_value = 1
        for i, p in enumerate(pauli.to_label()):  # Convert to string like "ZZ" or "Z "
            #print(p)
            if p == "Z":  # ignore "I" terms
                term_value *= bitstring_z[i]
        cost += coeff * term_value
    return cost



class Qaoa:
    def __init__(self, t, Hc:np.ndarray, n_layers:int, seed:bool, plots:bool, backend:str='ibm', instance:str='open'):
        self.Hc = Hc
        self.n_layers = n_layers
        self.seed = seed
        self.plots = plots
        self.backend_name = backend
        self.instance='wacqt/partners/scheduling-of-me'
        self.n_vars = len(self.Hc.paulis[0])

        if instance =='open':
            self.instance = 'ibm-q/open/main'

        if t == 0:
            if self.backend_name == 'ibm':
                print('\nUsing ibm hardware as quantum backend')
                self.backend=None
            else:
                print('\nUsing "aer" quantum simulator')

                self.backend = AerSimulator() 


            
    def findOptimalCircuit(self, estimation_iterations=2000, search_iterations=20):
        # Make initial circuit
        self.circuit = QAOAAnsatz(cost_operator=self.Hc, reps=self.n_layers) # Using a standard mixer hamiltonian 
        self.circuit.measure_all() 


        if self.backend_name == 'aer':
            pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend) # pass manager transpiles circuit
            self.transpiled_circuit = pass_manager.run(self.circuit)

        # Find best betas and gammas using estimator on initial circuit
        best_parameters = self.findParameters(estimation_iterations, search_iterations)
        best_circuit = self.transpiled_circuit.assign_parameters(parameters=best_parameters)
        self.optimized_circuit = best_circuit
    
    def samplerSearch(self, sampling_iterations=4000, n_candidates=20, return_worst_solution=False):
        self.sampleSolutions(sampling_iterations, plots=self.plots)
        self.sampling_iterations = sampling_iterations
        best_bitstring = self.findBestBitstring(n_candidates, worst_solution=return_worst_solution)
        return best_bitstring

    def findParameters(self, estimation_iterations, search_iterations): 
        
        #NOTE current version of COBYLA does not support bounds
        bounds = [(0, np.pi/2) for _ in range(self.n_layers)] # gammas have period =  π/2, given integer penalties
        bounds += [(0, np.pi) for _ in range(self.n_layers)] # ß:s have period π
        
        # AER SIMULATOR 
        if self.backend_name=='aer':
            candidates, costs = [],np.zeros(search_iterations)
            for i in range(search_iterations):
                if self.seed:
                    np.random.seed(i*10)
                initial_betas = np.random.random(size=self.n_layers)*np.pi/2 # Random initial angles 
                initial_gammas = np.random.random(size=self.n_layers)*np.pi  
                initial_parameters = np.concatenate([initial_gammas, initial_betas])

                estimator = Estimator(mode=self.backend,options={"default_shots": estimation_iterations})
                result = minimize(
                    estimateHc,
                    initial_parameters,
                    args=(self.transpiled_circuit, self.Hc, estimator),
                    method="COBYLA", # COBYLA is a classical OA: Constrained Optimization BY Linear Approximations
                    bounds=bounds,
                    tol=1e-4, #NOTE should be 1e-3 or smaller
                    options={"rhobeg": 0.5}   # Sets initial step size (manages exploration)
                    )
                candidates.append(result.x) 
                costs[i] = result.fun 
            found_parameters = candidates[np.argmin(costs)]

            if self.plots:
                plt.figure()
                plt.plot(Hc_values)
                plt.title(f'Estimated Hc ufsing simulator. \n(with {search_iterations} random initializations)')
                plt.show()
            Hc_values.clear()
            

        # IBM HARDWARE
        elif self.backend_name == 'ibm':
            if self.seed:
                    np.random.seed(10)
            max_init_distance = 0.2
            initial_betas = [np.pi,0] #*self.n_layers # TESTING #0.2  +  max_init_distance * np.random.random(size=self.n_layers)  # initial ß ≈ π/2
            initial_gammas = [1.75,0] #* self.n_layers # TESTING (1.9-max_init_distance) + np.random.random(size=self.n_layers) * max_init_distance   # initial gamma ≈ 0
            initial_parameters = np.concatenate([initial_gammas, initial_betas])

            token = open('../token.txt').readline().strip()
            service = QiskitRuntimeService(
                channel='ibm_quantum',
                instance=self.instance,
                token=token)
            self.backend = service.least_busy(min_num_qubits=127)

            # Best circuit transpilation out of 10, compare n.o. 2 qubit gates
            circuit_candicates, circuit_n_doubles = [], [] 
            for i in range(10):
                pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend) 
                circuit_i = pass_manager.run(self.circuit) 
                circuit_candicates.append(circuit_i)
                two_qubit_gate_count = sum(1 for instr, qargs, _ in circuit_i.data if len(qargs) == 2)
                circuit_n_doubles.append(two_qubit_gate_count)

            best_idx = np.argmin(circuit_n_doubles)
            print('n doubles', circuit_n_doubles)
            self.transpiled_circuit = circuit_candicates[best_idx]

            print('\ntranspiled')

            #grid_bounds = [(np.pi/2-0.1, np.pi/2+0.1), (0,0.2), (np.pi-0.2, np.pi), (0,0.2)]

            with Session(backend=self.backend) as session:
                estimator = Estimator(mode=session, options={"default_shots": estimation_iterations})
                
                '''# PLOT ENERGY LANDSCAPE
                brute_result = brute(estimateHc, grid_bounds, args=(self.transpiled_circuit, self.Hc, estimator), Ns=2, disp=True, workers=1, full_output=True)

                x0, fval, grid, Jout =  brute_result

                plt.figure(figsize=(10,8))
                plt.imshow(Jout, origin='lower', extent=(0, np.pi/2, 0, np.pi/2))            
                plt.show()
                Hc_values.clear()
            print('DONE WITH GRID SEARCH')'''
                result = minimize(
                    estimateHc,
                    initial_parameters,
                    args=(self.transpiled_circuit, self.Hc, estimator),
                    method="COBYLA",
                    bounds=bounds,
                    tol=1e-3,
                    options={"rhobeg": 1e-1}  
                )
                found_parameters = result.x 
                print('Found params using ibm:', result.x, 'estimated Hc:', result.fun)

            if self.plots:
                plt.figure()
                plt.plot(Hc_values)
                print(Hc_values)
                plt.title('Hc estimations using IBM')
                plt.show()

        return found_parameters

    
    def sampleSolutions(self, sampling_iterations, plots=True):
        # TODO Use single job-mode?
        sampler = Sampler(mode=self.backend, options={"default_shots": sampling_iterations})

        pub = (self.optimized_circuit,)
        job = sampler.run([pub])
        self.sampling_distribution = job.result()[0].data.meas.get_counts()

        if plots:
            plt.figure(figsize=(10,8))
            plt.title('Solution distribution')
            plt.bar([i for i in range(len(self.sampling_distribution))], self.sampling_distribution.values())
            plt.xticks(ticks = [i for i in range(len(self.sampling_distribution))], labels=self.sampling_distribution.keys())
            plt.xticks(rotation=90)
            plt.show()

    def findBestBitstring(self, n_candidates, worst_solution=False): # No prints temporary
        reverse = (worst_solution==False)
        sorted_distribution = dict(sorted(self.sampling_distribution.items(), key=lambda item:item[1], reverse=reverse)) #NOTE sorting might be memory expensive
        frequent_bitstrings = list(sorted_distribution.keys())[:n_candidates]
        costs = [costOfBitstring(bitstring, self.Hc) for bitstring in frequent_bitstrings]

        if worst_solution:
            best_bitstring = frequent_bitstrings[np.argmax(costs)]
        else:
            best_bitstring = frequent_bitstrings[np.argmin(costs)]
        self.best_bitstring = best_bitstring
        #print('\nBest bitstring:', best_bitstring)
        #print('best cost', costOfBitstring(best_bitstring, Hc))
        return best_bitstring

    def costCountsDistribution(self, random_distribution=None, bins=50):
        # QUANTUM
        all_costs, x_min, x_max = [], np.inf, -np.inf
        for bitstring_i in self.sampling_distribution.keys():
            count_i = self.sampling_distribution[bitstring_i]
            cost_i = np.real(costOfBitstring(bitstring_i, self.Hc))
            if cost_i < x_min:
                x_min = cost_i
            elif cost_i > x_max:
                x_max = cost_i
        
        # RANDOM 
        if random_distribution is not None:
            plot_costs = []
            all_costs_random = []
            for bitstring_i in random_distribution.keys():
                count_i = random_distribution[bitstring_i]
                cost_i = np.real(costOfBitstring(bitstring_i, self.Hc))
                all_costs_random += [cost_i]*count_i
            plot_costs = all_costs_random
            print('RANDOM')
            print(len(all_costs), len(all_costs_random), len(plot_costs))

            label = 'Random solutions'  
            color = 'orange'
            self.x_min = min(min(all_costs_random), x_min) # ensure same x-lims for plots
            self.x_max = max(max(all_costs_random), x_max)
        

        else:
            # QUANTUM
            print('QUANTUM')
            all_costs = []
            for bitstring_i in self.sampling_distribution.keys():
                count_i = self.sampling_distribution[bitstring_i]
                cost_i = np.real(costOfBitstring(bitstring_i, self.Hc))
                all_costs += [cost_i]*count_i
            plot_costs = all_costs
            label = str(self.backend_name)+' quantum backend'
            color = 'skyblue'


        n, bins, bars = plt.hist(plot_costs, bins=bins, label=label, color=color, range=(self.x_min, self.x_max), alpha=0.8)
        print('summa', sum(n))
        plt.legend()
        plt.xlabel('Cost (Hc)')
        plt.ylabel('Probability [%]')
        plt.yticks(ticks=np.linspace(0,self.sampling_iterations,11), labels=['','10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        plt.xlim((self.x_min,self.x_max))
        plt.ylim((0,self.sampling_iterations))
        return n, bins
