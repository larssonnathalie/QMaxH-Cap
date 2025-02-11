# from data Import shift time info, preferences, constraints etc.
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler#StatevectorSampler, Estimator



# construct empty calendar with work days, holidays etc
def emptyCalendar(end_date, start_date, prints=True):
    #current_year = datetime.now().year

    years = list(range(int(start_date[:4]),int(end_date[:4])+1))
    all_dates = pd.date_range(start=start_date, end=end_date)
    swedish_holidays = holidays.Sweden(years=years) 
    calendar_df = pd.DataFrame({'date': all_dates, 'is_holiday': all_dates.isin(swedish_holidays)}) # TODO add saturdays
    if prints:
        print(calendar_df)
    return calendar_df

# Automatically generate data in "demand_clX.csv", following repeating demand rules based on weekdays/holidays
# cl stands for complexity level
def generateDemandData(empty_calendar, cl, prints=True):
    if cl == 1: 
        demand_col = [int(empty_calendar.loc[i,'is_holiday']==False) for i in range(len(empty_calendar))] #[1]*len(empty_calendar)             
        df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col, 'is_holiday':empty_calendar['is_holiday']})

    else:
        print('generateDemandData is not coded yet for cl', str(cl))

    filename = 'data/demand_cl{}.csv'.format(str(cl))
    if prints:
        print(df)
        print('filename:', filename)
    df.to_csv(filename, index=False)




# construct objective functions for fairness, preference
# and constraints such as 1 person per shift & max shifts per week
def constructObjectives(empty_calendar_df, cl, preferences=True, prints=True):
    physician_df = pd.read_csv('data/physician_cl{}.csv'.format(str(cl))) 
    demand_df = pd.read_csv('data/demand_cl{}.csv'.format(str(cl)))
    n_physicians, n_shifts = physician_df.shape[0], demand_df.shape[0]    # NOTE assuming 1 shift per row
    n_vars = n_physicians * n_shifts # n.o. decision variables 

    def xToQIndex(x_index): # [[p,s],[p,s]] --> [i,j]    # TODO test function
        # Takes ps-indices of 2 x-variables that are combined in Q, 
        # Returns ij-index of q-element 
        first_x, second_x = x_index
        first_xp, first_xs, second_xp, second_xs = first_x, second_x

        i = first_xs + first_xp * n_shifts
        j = second_xs + second_xp * n_shifts
        return [i, j]
    
    def qToXIndex(q_index): #  [i,j] -->  [[p,s],[p,s]]   # TODO test function
        q_i, q_j = q_index
        first_xp = int(q_i/n_shifts)
        first_xs = q_i%n_shifts
        
        second_xp = int(q_j/n_shifts)
        second_xs = q_j%n_shifts

        return [[first_xp, first_xs], [second_xp, second_xs]]

    if cl <=1:

        # USING qiskit QP and np matrix 
        qp = QuadraticProgram()     
        # TODO Move section to makeQubo or add bitstring encoding

        all_x = []
        for p in range(n_physicians):        
            for s in range(n_shifts):
                qp.binary_var(name=f'x{p}{s}') 
                all_x.append((p,s))

        # QP constraints, later converted to penalties
        for s in range(n_shifts): # Exactly 1 p per s
            demand = demand_df['demand'].iloc[s]
            qp.linear_constraint(
                linear={f'x{p}{s}': 1 for p in range(n_physicians)},
                sense='==',
                rhs=demand, # right hand side
                name=f'fill_shift{s}')
        
            
        max_shifts_per_p = int(round(n_shifts/n_physicians+0.49999,1)) # TODO fix round
        for p in range(n_physicians): # fairness: s per p <= S/P
            qp.linear_constraint(
                linear={f'x{p}{s}': 1 for s in range(n_shifts)},
                sense='<=',
                rhs= max_shifts_per_p,
                name=f'fairness{p}')
        
        # Dummy quadratic constraint, did not work bc. not supported by QuadraticProgramToQubo()
        '''for s in range(n_shifts): # not 2 docs on 1 shift (redundant for now but might use later)
            for p in range(n_physicians):
                qp.quadratic_constraint(
                    quadratic={(f'x{p}{s}',f'x{p2}{s}'): 1 for p2 in range(p,n_physicians)}, # change range to all except p?
                    sense='==',
                    rhs= 0,
                    name=f'no_doubles{p}{s}')'''
        
        # Attempt to penalize directly, did not work, probably bc. library misuse
        ''' first_x in all_x:  # TODO Make automatic from objective, no if:s? 
            for second_x in all_x:
                p1, s1 = first_x # physician, shift
                p2, s2  = second_x

                if s1 == s2 and p1 != p2: # 2 p on same shift
                    q = 2* lamda
                    qp.minimize(quadratic={(f'x{p1}{s1}', f'x{p2}{s2}'): q})

                elif s1 == s2 and p1 == p2: # linear term
                    q = 1 * lamda
                    qp.minimize(linear={f'x{p1}{s1}': q})'''

        qubo = QuadraticProgramToQubo().convert(qp) # convert QP constraints to qubo penalties
        return qubo
            
        '''for i in range(n_vars):
            for j in range(n_vars):
                if Q[i, j] != 0:
                    x_first, x_second = qToXIndex([i,j])
                    if i==j:
                        qubo.minimize(linear={'x{}{}'.format(x_first[0], x_first[1]): Q[i, j]})
                    else:
                        qubo.minimize(quadratic={(f'x{x_first[0]}', f'x{j}'): Q[i, j]})'''




        # USING dictionary:
        '''Q_oneEachDay = {}
        for i in range(nVars):
            for j in range(i,nVars):
                if i==j:
                    Q_oneEachDay[(i,j)] = -1  # Minimize diagonal variables (why -1?)
                elif (j-i)%nShifts ==0:
                    Q_oneEachDay[(i,j)] = 2   # Penalize 2 docs on same shift'''
        

    else: 
        print('constructObjectives is not coded yet for cl', str(cl))

    if prints:
        print('Physicians:', n_physicians)
        print('Shifts:', n_shifts)
        print(qubo)

    return qubo


# construct QUBO from objective functions
def makeQubo(objectives,  lamda_fair = 1, lamda_pref=1, prints=True)->QuadraticProgram:
    encoding = []
    qubo = [] # type QuadraticProgram
    return qubo, encoding