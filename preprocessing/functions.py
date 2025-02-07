# from data Import shift time info, preferences, constraints etc.
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from qiskit_optimization import QuadraticProgram
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
        demand_col = [1]*len(empty_calendar)             # demand = 1 physician per day
        df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col})

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

    if cl <=1:
        n_vars = n_physicians * n_shifts # n.o. decision variables 

        # USING qiskit QP    
        qubo = QuadraticProgram()     # TODO Move section to makeQubo or add bitstring encoding
        linear_q={}             # TODO Demand based on demand_df
        quad_q={}
        '''for i in range(n_physicians): # TODO remove loops?
            for j in range(n_shifts):
                qubo.binary_var(name='x{}{}'.format(i,j))
                if i==j:
                    linear_q['x{}{}'.format(i,j)] = 1 # TODO diagonal weights value
                elif (j-i)%n_shifts ==0:
                    quad_q['x{}{}'.format(i,j)] = 2
        print(linear_q)
        print(quad_q)
        qubo.minimize(linear=linear_q)
        qubo.minimize(quadratic=quad_q)'''
        

        # USING help (translating i,j to Q indices)

        Q = np.zeros((n_vars,n_vars))
        lamda = 1
        x_index = 0
        for j in range(n_shifts):
            for i in range(n_physicians):
                qubo.binary_var(name=f'x{i}{j}') 
                print('added:'+f'x{i}{j}')
                #qubo.binary_var(name='x{}'.format(x_index))  # NOTE x-indices only 1 digit
                #print('added:'+f'x{x_index}')
                x_index +=1
                q_index_str = str(i + j * n_physicians)+str(i + j * n_physicians)
                q_index = [i + j * n_physicians,i + j * n_physicians]
                Q[q_index] = lamda  # Linear term

                for k in range(i + 1, n_physicians):
                    kq_index = [i + j * n_physicians, k + j * n_physicians]
                    Q[kq_index] = 2 * lamda  # Quadratic term
            
        for i in range(n_vars):
            for j in range(n_vars):
                if Q[i, j] != 0:
                    if i==j:
                        qubo.minimize(linear={f'x{i}': Q[i, j]})
                    else:
                        qubo.minimize(quadratic={(f'x{i}', f'x{j}'): Q[i, j]})




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