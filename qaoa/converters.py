import numpy as np
from qiskit.quantum_info import SparsePauliOp


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

def bitstringToPauliZ(bitstring)->np.ndarray:
   # binary to ±1 eigenvalues of Z operators
    z_eigenvalues = np.array([1 if bit == "0" else -1 for bit in bitstring])
    return z_eigenvalues

def getShiftsPerT(time_period:str, cl:int) -> int:
    if time_period == 'week':
        if cl<3:
            shifts_per_t = 7
        else:
            shifts_per_t = 3*7
            
    elif time_period == 'day':
        if cl <3:
            shifts_per_t=1
        else:
            shifts_per_t =3

    elif time_period == 'shift':
        shifts_per_t=1 
    return shifts_per_t

def percentOfShifts(percentage, cl):
    # assuming shifts are 8~hrs
    shifts_per_week = 7
    if cl>=3:
        shifts_per_week = 3*7

    target_percent_of_shifts = {25: 1.25/shifts_per_week, 50:2.5/shifts_per_week, 75:3.75/shifts_per_week, 100:5/shifts_per_week} 
    return target_percent_of_shifts[percentage]

def targetShiftsPerWeek(percentage, cl):
    # assuming shifts are 8~hrs
    shifts_per_week = 7
    if cl>=3:
        shifts_per_week = 3*7

    target_percent_of_shifts = {25: 1.25/shifts_per_week, 50:2.5/shifts_per_week, 75:3.75/shifts_per_week, 100:5/shifts_per_week} 
    target_shifts = target_percent_of_shifts[percentage]*shifts_per_week
    return target_shifts
    
def getDaysPassed(t, time_period):
    if time_period=='shift':
        days = t//3
    elif time_period =='week':
        days = t*7
    elif time_period =='day':
        days = t
    return days

# Only for visualization, does not handle complex numbers
'''def HcPaulisToQ(Hc:SparsePauliOp)-> np.ndarray: 
    n_vars = len(Hc.paulis[0])
    Q=np.zeros((n_vars,n_vars))

    for string_nr, string in enumerate(Hc.paulis):
        z_idcs=[]
        coeff = Hc.coeffs[string_nr]
        for count, digit in enumerate(string):
            if str(digit) == 'Z':
                z_idcs.append(n_vars-(count+1))
            if len(z_idcs) == 2:
                Q[z_idcs[0], z_idcs[1]] = coeff
                Q[z_idcs[1], z_idcs[0]] = coeff # mirror symmetric matrix
            elif count == (n_vars-1):
                Q[z_idcs[0], z_idcs[0]] = coeff
    print(Q)
    return Q'''
