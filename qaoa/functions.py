from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
import qiskit
import pandas as pd
import numpy as np
import sympy as sp


def xToQIndex(first_x, second_x, n_shifts, upper=True): # [[p,s],[p,s]] --> [i,j]    # TODO test function
    # Takes ps-indices of 2 x-variables that are combined in Q, 
    # Returns ij-index of q-element 
    p1, s1 = first_x
    p2, s2 = second_x
    i = s1 + p1 * n_shifts
    j = s2 + p2 * n_shifts

    if upper:  # Only cover upper half of matrix
        if i>j:
            i,j = j, i
    return [i, j]

def qToXIndex(q_index, n_shifts): #  [i,j] -->  [[p,s],[p,s]]   # TODO test function
    q_i, q_j = q_index
    first_xp = int(q_i/n_shifts)
    first_xs = q_i%n_shifts
    
    second_xp = int(q_j/n_shifts)
    second_xs = q_j%n_shifts
    return [[first_xp, first_xs], [second_xp, second_xs]]


def makeObjectiveFunctions(n_demand, n_shifts, n_physicians, cl, preferences, lambda_fair, lambda_pref):
    # Both objective & constraints formulated as Hamiltonians to be combined to QUBO form
    # TODO remove constructObjectives when this is working
    physician_df = pd.read_csv(f'data/intermediate/physician_cl{cl}.csv') 
    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')
    n_vars = n_shifts * n_physicians
    #x_matrix = np.zeros((n_physicians, n_shifts))  # decision variables

    # define decision variables (a list of lists)
    x_symbols = []
    for p in range(n_physicians):
        x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
        x_symbols.append(x_symbols_p)


    # Objective: minimize unfairness, physicians work similar amount
    # Hfair = ∑ᵢ₌₁ᴾ (∑ⱼ₌₁ˢ xᵢⱼ − S/P)²                 S = n_demand, P = n_physicians
    H_fair = 0
    for p in range(n_physicians):
        H_fair_jsum_i = sum(x_symbols[p][s] for s in range(n_shifts))   
        H_fair_i = (H_fair_jsum_i - (n_shifts/n_physicians))**2     #TODO use n_shifts_per_p? Find better expression?
        H_fair += H_fair_i
    #print(sp.expand(H_fair))


    if preferences:
        print('makeObjectiveFunctions does not handle preferences yet')
        # TODO: preferences
        # Objective: minimize preference dissatisfaction

    # Constraint: Meet demand
    # ∑s=1 (demanded – (∑p=1  x_ps))^2
    H_meet_demand = 0
    for s in range(n_shifts): 
        demand_s = demand_df['demand'].iloc[s]
        workers_s = sum(x_symbols[p][s] for p in range(n_physicians))    
        H_meet_demand_s = (demand_s-workers_s)**2
        H_meet_demand += H_meet_demand_s
    #print(sp.expand(H_meet_demand))

    # Combine to one single H
    # H = λ₁Hfair + λ₂Hpref + λ₃HmeetDemand
    H = sp.nsimplify(sp.expand(H_fair + H_meet_demand))
    #print(H)

    # Extract Q matrix from terms in H
    Q = np.zeros((n_vars,n_vars))
    for p in range(n_physicians):
        for s in range(n_shifts):
            x_ps = x_symbols[p][s]
            print(x_ps)    
            x_ps_terms = sum(term/x_ps for term in H.args if term.has(x_ps)) # extract the terms mutiplied by x_ps in H
            for term in x_ps_terms.as_ordered_terms():
                coeff, variables = term.as_coeff_mul(rational=True)
                #print(f"Term: {term}, Coefficient: {coeff}, Variables: {variables}")
                
                if len(variables)==0: # linear terms have no variable after /x_ps. Treated as diagonal terms
                    x_ps_2nd = x_ps  #NOTE assuming x_ps = x_ps^2 bc. binary
                else:
                    x_ps_2nd = variables[0]

                q_element = coeff      
                
                p2, s2 = str(x_ps_2nd).lstrip('x').split('_')

                if int(p2)<p: #  bc. Q is symmetric, we only need half the matrix = each pairwise relation once
                    if int(s2)<s:
                        print('CONTINUE') # Not needed yet bc. not all pair-relations are covered by H-functions 
                        continue 
                else:
                    pass
                    #print(int(p2), 'not <',p)
                
                q_i, q_j = xToQIndex([p, s], [int(p2), int(s2)], n_shifts)
                Q[q_i,q_j] += q_element
                if q_i != q_j: # Mirror matrix, avoid diagonal
                    Q[q_j,q_i] += q_element

    Q_df = pd.DataFrame(Q, index=None)
    Q_df.to_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv',index=False,header=False)

    return Q # type np.array


def makeCostHamiltonian(q_matrix, prints=True):
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
                if i == j:
                    pauli_string = ['I'] * n_vars
                    pauli_string[i] = 'Z'
                    pauli_terms.append(''.join(pauli_string))
                    coeffs.append(q_matrix[i, j])
                else:
                    pauli_string = ['I'] * n_vars
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    pauli_terms.append(''.join(pauli_string))
                    coeffs.append(q_matrix[i, j] / 2)

    hamiltonian = SparsePauliOp(pauli_terms, coeffs)

    # Print result
    print("Cost Hamiltonian:\n", hamiltonian)

    return hamiltonian

def makeAnsatz(Hc, prints=True)->QAOAAnsatz:
    # use QAOAAnsatz
    return []

def hardwareSetup(prints=True):
    return []

def transpileAnsatz(ansatz:QAOAAnsatz, backend, prints=True): 
    return []

def findParameters(initial_betas, initial_gammas, quantumCircuit, prints=True, plots=True):
    n_layers = len(initial_gammas)
    # Using estimator primitive
    # use parallelism job-mode
    return []


def sampleSolutions(bestParameters, prints=True, plots=True):
    # Using sampler primitive
    # Use single job-mode
    # returns distribution of solutions
    return []