import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler #StatevectorSampler, Estimator


# construct empty calendar with work days, holidays etc
def emptyCalendar(end_date, start_date, cl, prints=True, include_weekdays=True):
    all_dates = pd.date_range(start=start_date, end=end_date)

    years = list(range(int(start_date[:4]),int(end_date[:4])+1))
    #swedish_holidays = holidays.Sweden(years=years)
    swedish_holidays = all_dates.isin(holidays.Sweden(years=years))
    weekdays = all_dates.strftime('%A').values
    saturdays = weekdays =='Saturday'
    holidays_and_weekends = swedish_holidays+saturdays>= 1 # TODO replace bad solution with other OR function

    calendar_df = pd.DataFrame({'date': all_dates, 'is_holiday': holidays_and_weekends}) 
    if include_weekdays:
        calendar_df['weekday']= weekdays
    calendar_df.to_csv(f'data/intermediate/empty_calendar_cl{cl}.csv', index=False)
    if prints:
        print(calendar_df)
    

# Automatically generate data in "demand_clX.csv", following repeating demand rules based on weekdays/holidays
# cl stands for complexity level
def generateDemandData(empty_calendar, cl, weekday_workers=2, holiday_workers=1, prints=True):
    if cl == 1: 
        demand_col = [holiday_workers + int(empty_calendar.loc[i,'is_holiday']==False)*(weekday_workers-holiday_workers) for i in range(len(empty_calendar))] #[1]*len(empty_calendar)             
        df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col})

    else:
        print('generateDemandData is not coded yet for cl', str(cl))

    df.to_csv(f'data/intermediate/demand_cl{cl}.csv', index=False)


# preference column with dates in phys.csv + calendar info --> undesired combinations p & s
# (TODO add importance level or divide into definetely unavailable & prefer not)
# (TODO add reocurring preferences like thursday afternoons etc.?)
# TODO decide rules for preferences. Examples:
    # all p must have x prefered dates --> Maximize equal n.o. satisfied preferences
    # all p have varying number of prefered dates --> maximize equal share of satisfied preference, over time
    # other..?
def generatePreferences(empty_calendar_df, cl):
    date_to_s = {empty_calendar_df.loc[i,'date']:i for i in range(empty_calendar_df.shape[0])} # TODO more efficient coding
    physician_df = pd.read_csv(f'data/input/physician_cl{cl}.csv')
    preferences_shifts_col = []
    for p in range(physician_df.shape[0]):  # TODO paralellism?
        shifts = []
        dates = physician_df.loc[p,'preferences'].split(' ') #TODO better way to interpret csv list-like object?
        shifts.append(list(date_to_s[date] for date in dates))
        preferences_shifts_col.append(str(shifts).rstrip(']').lstrip('['))
    physician_df['preferences shifts'] = preferences_shifts_col
    physician_df.to_csv(f'data/intermediate/physician_cl{cl}.csv', index=None)

def xToQIndex(x_index, n_shifts): # [[p,s],[p,s]] --> [i,j]    # TODO test function
    # Takes ps-indices of 2 x-variables that are combined in Q, 
    # Returns ij-index of q-element 
    first_x, second_x = x_index
    first_xp, first_xs, second_xp, second_xs = first_x, second_x

    i = first_xs + first_xp * n_shifts
    j = second_xs + second_xp * n_shifts
    return [i, j]

def qToXIndex(q_index, n_shifts): #  [i,j] -->  [[p,s],[p,s]]   # TODO test function
    q_i, q_j = q_index
    first_xp = int(q_i/n_shifts)
    first_xs = q_i%n_shifts
    
    second_xp = int(q_j/n_shifts)
    second_xs = q_j%n_shifts
    return [[first_xp, first_xs], [second_xp, second_xs]]

# construct objective functions for fairness, preference
# and constraints such as 1 person per shift & max shifts per week
def constructObjectives(cl, n_physicians, n_shifts, max_shifts_per_p, preferences, prints=True):
    physician_df = pd.read_csv(f'data/intermediate/physician_cl{cl}.csv') 
    demand_df = pd.read_csv(f'data/intermediate/demand_cl{cl}.csv')

    if cl <=1:

        # USING qiskit QP
        qp = QuadraticProgram()     
        # TODO Change from QP to other module that supports quadratic constraints --> qubo

        all_x = []
        for p in range(n_physicians):        
            for s in range(n_shifts):
                qp.binary_var(name=f'x{p}{s}') 
                all_x.append((p,s))

        # QP constraints, later converted to penalties

        for s in range(n_shifts): # Exactly demanded p per s
            demand = demand_df['demand'].iloc[s]
            qp.linear_constraint(
                linear={f'x{p}{s}': 1 for p in range(n_physicians)},
                sense='==',
                rhs=demand, # right hand side
                name=f'fill_shift{s}')
        
        for p in range(n_physicians): # fairness: s per p <= S/P
            qp.linear_constraint(
                linear={f'x{p}{s}': 1 for s in range(n_shifts)},
                sense='<=',
                rhs= max_shifts_per_p,
                name=f'fairness{p}')
            
        if preferences:
            for p in range(n_physicians): # preferences: 
                shifts = [int(s) for s in physician_df.loc[p,'preferences shifts'].strip('"').split(',')]
                qp.linear_constraint(
                    linear={f'x{p}{s}': 1 for s in shifts},
                    sense='<=',
                    rhs= int(len(shifts)/2), # TODO decide max limit for preference dissatisfaction  
                    name=f'preference{p}')
                # TODO decide rules for preferences. Examples:
                    # all p must have x prefered dates --> Maximize equal n.o. satisfied preferences
                    # all p have varying number of prefered dates --> maximize equal share of satisfied preference, over time
                    # other..?

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
        
    else: 
        print('constructObjectives is not coded yet for cl', str(cl))
    

    return qubo


# construct QUBO from objective functions
def makeQubo(objectives,  lamda_fair = 1, lamda_pref=1, prints=True):
    encoding = []
    qubo = [] 
    return qubo, encoding