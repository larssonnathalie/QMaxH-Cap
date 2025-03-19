from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
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

def makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambdas, prints=False):
    # Both objective & constraints formulated as Hamiltonians to be combined to QUBO form
    # Using sympy to simplify the H expressions

    demand_df = pd.read_csv(f'data/intermediate/demand.csv')
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    
    # define decision variables (a list of lists)
    x_symbols = []
    for p in range(n_physicians):
        x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
        x_symbols.append(x_symbols_p)
    
    H_fair = 0
    H_meet_demand = 0
    H_pref = 0
    H_unavail = 0
    H_extent = 0

    if True:#cl<=2:
        # Objective: minimize UNFAIRNESS
        # Hfair = ∑ᵢ₌₁ᴾ (∑ⱼ₌₁ˢ xᵢⱼ − S/P)²                 S = n_demand, P = n_physicians
        max_shifts_per_p = int((n_demand/n_physicians)+0.999 ) # fair distribution of shifts
        for p in range(n_physicians):
            H_fair_s_sum_p = sum(x_symbols[p][s] for s in range(n_shifts))   
            H_fair_p = (H_fair_s_sum_p - max_shifts_per_p)**2   
            H_fair += H_fair_p
    
    if False:#cl>2:
        # EXTENT (work percentage) constraint
        for p in range(n_physicians):
            percentage = physician_df['Extent'].iloc[p]/100
            n_shifts_target_p = int(n_demand*percentage) # TODO Change to correct n.o. hours per week
            H_extent_s_sum_p = sum(x_symbols[p][s] for s in range(n_shifts))   
            H_extent_p = (H_extent_s_sum_p - n_shifts_target_p)**2   
            H_extent += H_extent_p

    if cl>1:
        # Objective: minimize PREFERENCE dissatisfaction
        for p in range(n_physicians): 
            prefer_p = physician_df['Prefer'].iloc[p]
            if prefer_p != '[]':
                prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
                H_pref_p = sum(x_symbols[p][int(s)] for s in prefer_shifts_p) # Reward prefered shifts (negative penalties)
                H_pref -= H_pref_p 

            prefer_not_p = physician_df['Prefer Not'].iloc[p]
            if prefer_not_p != '[]':
                prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  
                H_pref_not_p = sum(x_symbols[p][int(s)] for s in prefer_not_shifts_p) # Penalize unprefered shifts
                H_pref += H_pref_not_p

            if cl>2:
                # UNAVAILABLE constraint
                unavail_shifts_p = physician_df['Unavailable'].iloc[p]
                if unavail_shifts_p != '[]':
                    unavail_shifts_p = unavail_shifts_p.strip('[').strip(']').split(',')  
                    H_unavail_p = sum(x_symbols[p][int(s)] for s in unavail_shifts_p)
                    H_unavail += H_unavail_p
                

    # Constraint: Meet DEMAND
    # ∑s=1 (demanded – (∑p=1  x_ps))^2
    for s in range(n_shifts): 
        demand_s = demand_df['demand'].iloc[s]
        workers_s = sum(x_symbols[p][s] for p in range(n_physicians))   
        H_meet_demand_s = (sp.Integer(demand_s)-workers_s)**2
        H_meet_demand += H_meet_demand_s
    

    # Combine to one single H
    # H = λ₁H_fair + λ₂H_pref + λ₃H_meetDemand
    if prints:
        print('Hdemand', sp.nsimplify(sp.expand(H_meet_demand*lambdas['demand'])))
        print('Hfair', sp.nsimplify(sp.expand(H_fair*lambdas['fair'])))
    all_hamiltonians = sp.nsimplify(sp.expand(H_meet_demand*lambdas['demand'] + H_fair*lambdas['fair'] + H_pref*lambdas['pref'] + H_unavail * lambdas['unavail'] + H_extent*lambdas['extent']))
    return all_hamiltonians, x_symbols

def objectivesToQubo(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='QP', mirror=True):
    x_list = [x_symbols[p][s] for p in range(n_physicians) for s in range(n_shifts)]
    n_vars = n_physicians*n_shifts
    Q = np.zeros((n_vars,n_vars))

    for term in all_hamiltonians.as_ordered_terms():
        coeff, variables = term.as_coeff_mul()

        if len(variables) == 1: # Linear terms
            var = variables[0]
            term_powers = term.as_powers_dict()
            if term_powers[var] ==0:  # Handle x^2 terms
                #print('=0',term.as_powers_dict()) 
                #print(var)
                var = list(term_powers.keys())[1] # TODO better solution
            idx = x_list.index(var) #TODO remove index
            Q[idx, idx] += coeff  

        elif len(variables) == 2: # Quadratic terms
            var1, var2 = variables

            if var1 in x_list and var2 in x_list:
                idx1 = x_list.index(var1)
                idx2 = x_list.index(var2)
                if idx1 != idx2:
                    if idx1>idx2: # upper triangular
                        idx1, idx2 = idx2, idx1
                    Q[idx1, idx2] += coeff  # Off-diagonal terms
                    if mirror:
                        Q[idx2, idx1] += coeff  # Symmetric QUBO matrix
                else:
                    print('THIS SHOULD NOT OCCUR, SOMETHING WRONG IN MAKEQUBO()')   #Q[idx1, idx1] += coeff  # Self-interaction terms

    # Save Q to csv
    Q_df = pd.DataFrame(Q, index=None)
    Q_df.to_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv', index=False, header=False)

    return Q


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


def cost_func_estimator(parameters, ansatz, hamiltonian, estimator): 

    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, parameters)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    Hc_values.append(cost) # NOTE global list

    return cost

def findParameters(n_layers, circuit, backend, Hc, estimation_iterations, search_iterations, seed=True, prints=True, plots=True): # TODO what job mode? (single, session, etc)
    estimator = Estimator(mode=backend,options={"default_shots": estimation_iterations})
    bounds = [(0, 2*np.pi) for _ in range(n_layers)] # gammas have period = 2 pi, given integer penalties
    bounds += [(0, np.pi) for _ in range(n_layers)] # betas have period = 1 pi

    # test plot of energy landscape
    '''brute_result = brute(cost_func_estimator, bounds, args=(circuit, Hc, estimator), Ns=30, disp=True, workers=1, full_output=True)
    plt.imshow(brute_result[2][0])
    plt.figure()
    plt.imshow(brute_result[2][1])
    plt.show()
    print('\nBRUTE:',brute_result[0])
    Hc_values.clear()'''

    candidates, costs = [],[]
    for i in range(search_iterations):
        if seed:
            np.random.seed(i*10)
        initial_betas = np.random.random(size=n_layers)*np.pi # Random initial angles [np.pi/2]*n_layers 
        initial_gammas = np.random.random(size=n_layers)*np.pi*2  #[np.pi]*n_layers  
        initial_parameters = np.concatenate([initial_gammas, initial_betas])
        result = minimize(  
            cost_func_estimator,
            initial_parameters,
            args=(circuit, Hc, estimator),
            method="COBYLA", # COBYLA is a classical OA: Constrained Optimization BY Linear Approximations
            bounds=bounds,  
            tol=1e-3,           
            options={"rhobeg": 2}   # Sets initial step size (manages exploration)
        )
        candidates.append(result.x)
        costs.append(cost_func_estimator(result.x, circuit, Hc, estimator))
    
    #print('costs',costs) 
    #print('min',costs[np.argmin(costs)])
    parameters = candidates[np.argmin(costs)]
    print('COBYLA:', parameters)
    if plots:
        plt.figure()
        plt.plot(Hc_values)
        plt.title('Hc costs while optimizing ßs and gammas')
        plt.show()
    if prints:
        print('\nBest parameters (ß:s & gamma:s):', parameters)
        print('Estimated cost', cost_func_estimator(parameters, circuit, Hc, estimator))
        print('\nEstimator iterations', len(Hc_values))

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
        print('\nSampling iterations:', sum(sampling_distribution.values()))
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

def findBestBitstring(sampling_distribution:dict, Hc, n_candidates=20, prints=True, worst_solutions=False):
    reverse= not worst_solutions
    sorted_distribution = dict(sorted(sampling_distribution.items(), key=lambda item:item[1], reverse=reverse)) #NOTE might be memory demanding
    best_bitstrings = list(sorted_distribution.keys())[:n_candidates]
    
    costs = [costOfBitstring(bitstring, Hc) for bitstring in best_bitstrings]
    best_bitstring = best_bitstrings[np.argmin(costs)]

    if prints:
        print('\nBest bitstring:', best_bitstring)
        print('cost', costOfBitstring(best_bitstring, Hc))
    return best_bitstring
