from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qaoa.converters import *
import matplotlib.pyplot as plt
import qiskit
import pandas as pd
import numpy as np
import sympy as sp

Hc_values = []

def makeQuboNathaliesSolution(n_demand, n_physicians, n_shifts, cl, lambda_demand, lambda_fair):
    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')

    Q = {}

    # Objective function: Minimize dissatisfaction + penalties
    for p in range(n_physicians):
        for s in range(n_shifts):
            Q[(p, s), (p, s)] = 1

    # Meet demand constraint
    for s in range(n_shifts):
        demand_s  = demand_df['demand'].iloc[s]
        for p in range(n_physicians):
            for p_prime in range(n_physicians):
                if p != p_prime:
                    Q[(p, s), (p_prime, s)] = lambda_demand  # Penalize assignment of different physicians on the same shift
        Q[(p, s), (p, s)] -= lambda_demand * demand_s  #
    
    # Convert dict to np
    n_vars = n_physicians * n_shifts
    Q_np = np.zeros((n_vars,n_vars))
    for key in Q.keys():
        i,j = xToQIndex(key[0],key[1], n_shifts=n_shifts)
        Q_np[i,j]=Q[key]
    print(Q_np)

    return Q_np



def makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambda_demand, lambda_fair, prints=False):
    # Both objective & constraints formulated as Hamiltonians to be combined to QUBO form
    # Using sympy to simplify the H expressions

    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')

    # define decision variables (a list of lists)
    x_symbols = []
    for p in range(n_physicians):
        x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
        x_symbols.append(x_symbols_p)

    # Objective: minimize unfairness, physicians work similar amount
    # Hfair = ∑ᵢ₌₁ᴾ (∑ⱼ₌₁ˢ xᵢⱼ − S/P)²                 S = n_demand, P = n_physicians
    max_shifts_per_p = int((n_demand/n_physicians)+0.999 ) # fair distribution of shifts
    H_fair = 0
    for p in range(n_physicians):
        H_fair_s_sum_p = sum(x_symbols[p][s] for s in range(n_shifts))   
        H_fair_p = (H_fair_s_sum_p - max_shifts_per_p)**2   
        H_fair += H_fair_p

    if cl>1:
        print('makeObjectiveFunctions does not handle preferences yet')
        # TODO: preferences
        # Objective: minimize preference dissatisfaction

    # Constraint: Meet demand
    # ∑s=1 (demanded – (∑p=1  x_ps))^2
    H_meet_demand = 0
    for s in range(n_shifts): 
        demand_s = demand_df['demand'].iloc[s]
        workers_s = sum(x_symbols[p][s] for p in range(n_physicians))   
        if prints:
            print(f'workers on shift {s}:',workers_s.subs(substitution),'demand',demand_s, '\tSum: ',(workers_s.subs(substitution)-demand_s)**2)
            print()
        H_meet_demand_s = (sp.Integer(demand_s)-workers_s)**2
        H_meet_demand += H_meet_demand_s
    #print(sp.expand(H_meet_demand))

    # Combine to one single H
    # H = λ₁Hfair + λ₂Hpref + λ₃HmeetDemand
    if prints:
        print('Hdemand', sp.nsimplify(sp.expand(H_meet_demand*lambda_demand)).subs(substitution))
        print('Hfair', sp.nsimplify(sp.expand(H_fair*lambda_fair)).subs(substitution))
    all_hamiltonians = sp.nsimplify(sp.expand(H_meet_demand*lambda_demand + H_fair*lambda_fair))
    return all_hamiltonians, x_symbols



def hamiltoniansToQuboMatrix(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='QP', mirror=True)->np.ndarray:
    # Extract Q matrix from terms in H
    n_vars = n_physicians * n_shifts
    Q = np.zeros((n_vars,n_vars)) #TODO remove steps, now we go from sp -> np -> QP. Should be possible to do sp -> QP

    for p in range(n_physicians):
        for s in range(n_shifts):                        # ps_1
            x_ps_1st = x_symbols[p][s]
            x_ps_1st_terms = sum(term/x_ps_1st for term in all_hamiltonians.args if term.has(x_ps_1st)) # extract the terms mutiplied by x_ps in H
            for term in x_ps_1st_terms.as_ordered_terms():           # ps_2
                coeff, variables = term.as_coeff_mul(rational=True)
                #print(f"Term: {term}, Coefficient: {coeff}, Variables: {variables}")
                #if len(variables)==0: # linear terms have no variable after /x_ps. Treated as diagonal terms
                    #x_ps_2nd = x_ps_1st  
                x_ps_2nd = x_ps_1st if len(variables) == 0 else variables[-1]

                #else:
                   # x_ps_2nd = variables[0]
                q_element = coeff      
                p2, s2 = str(x_ps_2nd).lstrip('x').split('_')
                #if (int(p2), int(s2)) < (p, s):  #  bc. Q is symmetric, we only need half the matrix = each pairwise relation once
                    #print('CONTINUE') # Not needed yet bc. not all pair-relations are covered by H-functions
                    #continue 
                q_i, q_j = xToQIndex([p, s], [int(p2), int(s2)], n_shifts)
                
                Q[q_i,q_j] += q_element  # NOTE not 100% if it should be += or =
                if mirror:
                    if q_i != q_j: # Mirror matrix, avoid diagonal
                        Q[q_j,q_i] = q_element

    # Save Q to csv
    Q_df = pd.DataFrame(Q, index=None)
    Q_df.to_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv', index=False, header=False)

    
    if output_type == 'QP':   # Convert Q  --> QuadraticProgram, without constraints (they are encoded in the objective)
        qp = QuadraticProgram()
        n_vars = Q.shape[0]
        for i in range(n_vars):
            qp.binary_var(f'x{i}') 
        linear = {f'x{i}': Q[i, i] for i in range(n_vars)} 
        quadratic = {(f'x{i}', f'x{j}'): Q[i, j] for i in range(n_vars) for j in range(i+1, n_vars) if Q[i, j] != 0}
        # set objective
        qp.minimize(linear=linear, quadratic=quadratic)
        q = qp
    else:
        q = Q

    return q

def makeQuboNew(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='QP', mirror=True):
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
                    #Q[idx1, idx1] = coeff  # Self-interaction terms #NOTE removed +=
                    print('THIS SHOULD NOT OCCUR')

    return Q


# NOTE: COPIED THIS FUNCTION TO GET ESTIMATOR WORKING 
def cost_func_estimator(parameters, ansatz, hamiltonian, estimator): #TODO: UNDERSTAND WHAT HAPPENS & rewrite

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits for the backend.
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, parameters)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    Hc_values.append(cost)
    #objective_func_vals.append(cost)

    return cost

# Replaced with QuboToIsing for now
'''def QToHc(q_matrix):   # add offset (b) as argument, as in nb  
    # Takes qubo and returns ising hamiltonian

    # Expand Qubo:
    # x^T Q x  =     ∑ᵢ Qᵢᵢxᵢ   +    ∑ᵢ<ⱼ   Qᵢⱼxᵢxⱼ        (1)
    #                 ^diagonal^     ^upper half^
    # Substitution:
    # x_i = (1-z_i)/2        (z_i = Pauli-Z operator acting on qubit i)
    #
    # Put in eq. (1) -->   xᵢ = (1-zᵢ)/2
    #                      xᵢxⱼ = (1-zᵢ)(1-zⱼ)/4  = 1/4(1 -zᵢ -zⱼ +zᵢzⱼ)
    # eq (1) -->        Hc = ∑ᵢ cᵢZᵢ + ∑ᵢ<ⱼ cᵢⱼZᵢZⱼ
    # where:
    # ci = - Qii/2  -  ∑ᵢ≠ⱼ Qij/4
    # cij = Qij/4
    # Convert to Ising Hamiltonian

    n_vars = q_matrix.shape[0]
    pauli_terms = []
    coeffs = []

    for i in range(n_vars):
        for j in range(n_vars):
            if q_matrix[i, j] != 0:
                if i == j: # linear (diagonal) terms
                    pauli_string = ['I'] * n_vars
                    pauli_string[i] = 'Z'
                    pauli_terms.append(''.join(pauli_string))
                    coeffs.append(q_matrix[i, j])
                else: # quadratic terms
                    pauli_string = ['I'] * n_vars
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    pauli_terms.append(''.join(pauli_string))
                    coeffs.append(q_matrix[i, j] / 2)

    Hc = SparsePauliOp(pauli_terms, coeffs)
    #print('Hc:', Hc)

    return Hc'''


def transpileAnsatz(ansatz:QAOAAnsatz, backend): # TODO do we need to transpile? Passmanager transpiles
    return []

def findParameters(initial_parameters, circuit, backend, Hc, estimation_iterations, prints=True, plots=True): # TODO what job mode? (single, session, etc)

    estimator = Estimator(mode=backend,options={"default_shots": estimation_iterations})

    bounds = [(0, 2*np.pi) for _ in range(len(initial_parameters))] 

    result = minimize(  
        cost_func_estimator,
        initial_parameters,
        args=(circuit, Hc, estimator),
        method="COBYLA", # COBYLA is a classical OA: Constrained Optimization BY Linear Approximations
        bounds=bounds,  
        tol=1e-6,           
        options={"rhobeg": 1e-1}   # TODO replace copied settings
    )
    parameters = result.x
    if prints:
        print('\nEstimator iterations', len(Hc_values))

    if plots:
        plt.figure()
        plt.plot(Hc_values)
        plt.title('Hc costs while optimizing ßs and gammas')
        plt.show()
    if prints:
        print('\nBest parameters (ß:s & gamma:s):', parameters)
    return parameters


def sampleSolutions(best_circuit, backend, sampling_iterations, prints=True, plots=True):
    # TODO Use single job-mode?
    sampler = Sampler(mode=backend, options={"default_shots": sampling_iterations})

    # TODO circuit not transpiled? Check if needed
    pub = (best_circuit,)
    job = sampler.run([pub])
    sampling_distribution = job.result()[0].data.meas.get_counts()

    if plots:
        plt.figure(figsize=(20,10))
        plt.title('Solution distribution')
        plt.bar([i for i in range(len(sampling_distribution))], sampling_distribution.values())
        plt.xticks(ticks = [i for i in range(len(sampling_distribution))], labels=sampling_distribution.keys())
        plt.xticks(rotation=90)
        plt.show()
    if prints:
        print('\nSampling iterations:', sum(sampling_distribution.values()))
    return sampling_distribution

def findBestBitstring(sampling_distribution:dict, prints=True):
    counts, bitstrings = list(sampling_distribution.values()), list(sampling_distribution.keys())
    counts_np = np.array(counts)
    max_count = np.max(counts_np)  
    max_idcs = np.where(counts_np == max_count)[0]
    print('max index = ', max_idcs, len(max_idcs))
    best_bitstrings = [bitstrings[idx] for idx in max_idcs]
    if prints:
        print('\nBest bitstring:', best_bitstrings)
    return best_bitstrings

def costOfBitstring(bitstring:str, Hc:SparsePauliOp):
    bitstring_z = bitstringToPauliZ(bitstring)
    cost = 0
    for pauli, coeff in zip(Hc.paulis, Hc.coeffs):
        term_value = 1
        for i, p in enumerate(pauli.to_label()):  # Convert to string like "ZZ" or "Z "
            if p == "Z":  # Only consider Z terms (ignore "I" terms)
                term_value *= bitstring_z[i]
        cost += coeff * term_value

    print("Cost of bitstring:", cost)
    return cost

def QuboToIsing(Q_qubo:np.ndarray): #TODO include complex values
    # Takes qubo matrix where variables are assumed to be (0,1) 
    # returns corresponding ising Q matrix where variables assumed to be (-1,1)
    # b is vector for linear elements, replacing diagonal in qubo matrix 

    print('Q qubo\n',Q_qubo)
    n_vars = Q_qubo.shape[0]
    Q_ising = np.zeros((n_vars, n_vars))
    b_ising = np.zeros(n_vars)

    for i in range(n_vars): # TODO check correct index? 
        j_sum_i = 0
        for j in range(i+1,n_vars):
            #if i ==j:
                #j_sum_i += Q_qubo[i,j]*2
            #else:  
                Q_ising[i,j] = Q_qubo[i,j]/4 #TODO add symmetry?
        #b_ising[i] = -j_sum_i/4
    b_ising = -np.sum(Q_qubo, axis=1) / 4 

    
    return Q_ising, b_ising


### THIS IS PONTUS'S FUNC TO COMPARE RESULTS:::
def generate_pauli_terms(Q: np.ndarray, b: np.ndarray) -> list[tuple[str, float]]:
    """Construct the cost Hamiltonian.

    Args:
        Q (ndarray): NxN symmetric matrix of coefficients for Z_i Z_j terms.
        b (ndarray): N-dimensional array coefficients for Z_i terms.

    Returns:
        pauli_list: List of (Pauli string, coefficient) pairs representing the Hamiltonian terms.
    """
    N = len(b) # number of qubits

    pauli_list = []

    # Two-qubit terms
    for i in range(N-1):                       #NOTE    why N-1????????????????
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                # Create a Pauli string with "Z" at positions i and j
                paulis = ["I"] * N
                paulis[i], paulis[j] = "Z", "Z"
                coeff = 2 * Q[i, j] / 4 # multiply by a factor 2 since we are only summing the upper triangular of Q
                pauli_list.append(("".join(paulis)[::-1], coeff))

    # Single-qubit terms
    for i in range(N):
        if b[i] != 0: 
            # Create a Pauli string with "Z" at position i
            paulis = ["I"] * N
            paulis[i] = "Z"
            coeff = b[i] / 4
            pauli_list.append(("".join(paulis)[::-1], coeff))
    
    return pauli_list
