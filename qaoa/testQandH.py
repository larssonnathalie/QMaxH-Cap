import sympy as sp
import numpy as np

STRING = '101100110001011010'
n_physicians = 3
n_shifts=6
n_demand = 10
cl=1
lambda_demand = 2
lambda_fair = 1

# define decision variables (a list of lists)
'''x_symbols = []
for p in range(n_physicians):
    x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
    x_symbols.append(x_symbols_p)'''

def assignVariables(bitstring:str, x_symbols)->dict:
    substitution = {}
    i=0
    for p in x_symbols:
        for x_ps in p:
            substitution[x_ps] = int(bitstring[i])
            i+=1
    return substitution

def get_xT_Q_x(bitstring:str, Q):
    int_string =[]
    for bit in bitstring:
        int_string.append(int(bit))
    x = np.array(int_string)

    x.resize((len(bitstring),1))

    xT_Q_x = x.T @ Q @ x
    return xT_Q_x
    


'''substitution = assignVariables(STRING)

#print('Variables:', substitution)"""
    
# Make, sum and simplify all hamiltonians and enforce penatlies (lambdas)
all_hamiltonians, x_symbols = makeObjectiveFunctions(n_demand, n_physicians, n_shifts, cl, lambda_demand=lambda_demand, lambda_fair=lambda_fair) # NOTE does not handle preferences yet
#Q = objectivesToQubo(all_hamiltonians, n_physicians, n_shifts, x_symbols, cl, output_type='np', mirror=False)

hsum = all_hamiltonians.subs(substitution)
print('Hsum',hsum)
#print('H expression\n', all_hamiltonians)
# Extract Qubo Q-matrix from hamiltonians           Y = x^T Qx

x_string = np.zeros((len(STRING),1))
i=0
for s in STRING:
    x_string[i] = int(s)
    i+=1

x_sym= sp.Matrix(np.zeros((len(STRING),1)))
i=0
for p in range(n_physicians):
    for s in range(n_shifts):
        x_sym[i] = x_symbols[p][s]
        i+=1
#print('string', STRING, x_string)
xTQx_expr = sp.expand(sp.simplify(x_sym.T* sp.Matrix(Q)*x_sym))[0]
#print('\nxQx expr\n', xTQx_expr) #xtQx expr
#print((i[0],i[1]) for i in [term.as_coeff_mul() for term in hsum.as_ordered_terms()])
#print(i for i in [term.as_coeff_mul() for term in xTQx_expr.as_ordered_terms()])

new_h_expr = 0
for term in all_hamiltonians.as_ordered_terms(): # Shows the ignored constant term 
    coeff, variables = term.as_coeff_mul()
    if len(variables) == 0:
        print('0vars:\n',coeff, variables) #PRITNT C?
        new_h_expr += term
    else:
        #print(variables[0])
        #print('DICT',term.as_powers_dict()[variables[0]])
        if len(variables) == 1 and term.as_powers_dict()[variables[0]] !=0:  # Get dictionary of {variable: exponent}:
            new_h_expr += coeff*variables[0]**2 
        else:
            new_h_expr += term
difference_expr = sp.simplify(sp.expand(new_h_expr - xTQx_expr))
#print('\ndiff:\n', difference_expr)
if type(difference_expr) != int:
    print('Objective functions expressionn and x^TQx differs more than just a constant')
    print(type(difference_expr), difference_expr)

#print([f'{term1}\t{term2}\n' for (term1,term2) in zip(all_hamiltonians.as_ordered_terms(), xTQx_expr.as_ordered_terms())] )
xQx = np.matmul(np.matmul(x_string.T, Q),x_string)[0][0]
print('xQx', xQx) #xtQx
print(xQx + difference_expr, hsum)
print('xQx + C = Hsum:', int(xQx + difference_expr) == hsum)
#print(x_symbols, sp.Matrix(Q))'''