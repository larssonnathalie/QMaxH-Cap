import pandas as pd
import numpy as np
import sympy as sp
import holidays
from datetime import datetime
#from qiskit_algorithms.optimizers import COBYLA

# construct empty calendar with work days, holidays etc
def emptyCalendar(end_date, start_date, prints=True, include_weekdays=True):
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
    calendar_df.to_csv(f'data/intermediate/empty_calendar.csv', index=False)
    if prints:
        print(calendar_df)

# Automatically generate physicians_data with random preferences on relevant dates
def generatePhysicianData(empty_calendar, n_physicians, seed=True):

    all_dates = empty_calendar['date'] # TODO preferences on separate shifts instead of dates
    n_dates = len(all_dates)
    possible_extents = [25,50,50,75,75,100,100,100,100,100,100]  
    possible_titles = ['UL', 'ÖL', 'AT', 'ST', 'Chef'] # more copies --> more likely
    possible_competences = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  

    name_col = []
    extent_col=[]
    prefer_col = []
    prefer_not_col = []
    unavail_col = []
    title_col = [] 
    competence_col = []

    if seed:
        np.random.seed(12)
    
    for p in range(n_physicians):
        remaining_dates_p = list(all_dates)
        name_col.append(f'physician{p}')
        extent_col.append(np.random.choice(possible_extents))
        title_col.append(np.random.choice(possible_titles))
        competence_col.append(np.random.choice(possible_competences))

        # RANDOM PREFERENCES
        prefer_not_p = np.random.choice(remaining_dates_p, size=np.random.randint(0, n_dates // 2 + 1), replace=False) # NOTE maybe change upper size limit
        prefer_not_col.append(list(prefer_not_p))
        for s in prefer_not_p:
            remaining_dates_p.remove(s)

        prefer_p = np.random.choice(remaining_dates_p, size=np.random.randint(0, len(remaining_dates_p)), replace=False)
        prefer_col.append(list(prefer_p))
        for s in prefer_p:
            remaining_dates_p.remove(s)

        unavail_p = np.random.choice(remaining_dates_p, size=np.random.randint(0, len(remaining_dates_p)), replace=False)
        unavail_col.append(list(unavail_p))

    physician_data_df = pd.DataFrame({'name':name_col, 'extent': extent_col, 'prefer':prefer_col, 'prefer not':prefer_not_col, 'unavailable':unavail_col, 'title':title_col, 'competence':competence_col})

    physician_data_df.to_csv('data/intermediate/physician_data.csv', index=None)


# Automatically generate "shift_data.csv"
# following repeating demand rules based on weekdays/holidays
def generateShiftData(empty_calendar, cl, weekday_workers=2, holiday_workers=1, prints=True):
    if cl<4:
        demand_col = [holiday_workers + int(empty_calendar.loc[i,'is_holiday']==False)*(weekday_workers-holiday_workers) for i in range(len(empty_calendar))]          
        shift_data_df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col})
    elif cl>=4:
        # SHIFT TYPES
        date_col = []
        shift_type_col=[]
        demand_col = []
        demands = {('dag', False):2, ('kväll',False):2, ('natt',False):2,('dag',True):1, ('kväll',True):1, ('natt',True):1} #TODO decide values
        for i in range(len(empty_calendar)):
            date_i = empty_calendar.loc[i,'date']
            is_holiday_i = empty_calendar.loc[i,'is_holiday']

            for shift_type in ['dag', 'kväll','natt']:            # NOTE assuming 3 shifts per day
                date_col.append(date_i)
                shift_type_col.append(shift_type)
                demand = demands[(shift_type, is_holiday_i)]
                demand_col.append(demand)

        shift_data_df = pd.DataFrame({'date':date_col, 'shift type':shift_type_col, 'demand': demand_col})
    shift_data_df.to_csv(f'data/intermediate/shift_data.csv', index=False)

# Convert preferences from dates to shift numbers
def convertPreferences(empty_calendar_df):
    shifts_df = pd.read_csv(f'data/intermediate/shift_data.csv')
    n_shifts=len(shifts_df)
    date_to_s={}
    for s in range(n_shifts):
        date = shifts_df.loc[s,'date']
        if date in date_to_s.keys():
            date_to_s[date].append(s)
        else:
            date_to_s[date] = [s]
    included_dates = [str(empty_calendar_df['date'].iloc[i]) for i in range(len(empty_calendar_df))]
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv') 
    n_physicians = physician_df.shape[0]

    prefer_shifts_col = ['']*n_physicians
    prefer_not_shifts_col = ['']*n_physicians
    unavailable_shifts_col = ['']*n_physicians

    for p in range(n_physicians): 
        prefer_dates =  physician_df.loc[p,'prefer']
        prefer_shifts = []
        if prefer_dates !='[]':
            prefer_dates = prefer_dates.strip(']').strip('[').split(',')
            for date in prefer_dates:
                date = str(date.strip('"').strip("'"))
                if date in included_dates:
                    for s in date_to_s[date]:
                        prefer_shifts.append(s)
        prefer_shifts_col[p]=prefer_shifts 

        prefer_not_dates = physician_df.loc[p,'prefer not']
        prefer_not_shifts =[]
        if prefer_not_dates!='[]':
            prefer_not_dates = prefer_not_dates.strip(']').strip('[').split(',') 
            for date in prefer_not_dates:
                date = date.strip('"').strip("'")
                if date in included_dates:
                    for s in date_to_s[date]:
                        prefer_not_shifts.append(s)
        prefer_not_shifts_col[p]= prefer_not_shifts 
        
        unavailable_dates = physician_df.loc[p,'unavailable']
        unavailable_shifts =[]
        if unavailable_dates!='[]':  
            unavailable_dates = unavailable_dates.strip(']').strip('[').split(',') 
            for date in unavailable_dates:
                date = date.strip('"').strip("'")
                if date in included_dates:
                    for s in date_to_s[date]:
                        unavailable_shifts.append(s)   
        unavailable_shifts_col[p]= unavailable_shifts 

    physician_df['prefer'] = prefer_shifts_col
    physician_df['prefer not'] = prefer_not_shifts_col
    physician_df['unavailable'] = unavailable_shifts_col
    print(physician_df)
    physician_df.to_csv(f'data/intermediate/physician_data.csv', index=None)


def makeObjectiveFunctions(n_demand, cl, lambdas, prints=False):
    # Both objective & constraints formulated as Hamiltonians to be combined to QUBO form
    # Using sympy to simplify the H expressions

    shifts_df = pd.read_csv(f'data/intermediate/shift_data.csv')
    n_shifts = len(shifts_df)
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    n_physicians = len(physician_df)
    
    # define decision variables (a list of lists)
    x_symbols = []
    for p in range(n_physicians):
        x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
        x_symbols.append(x_symbols_p)
    
    H_fair = 0
    H_meet_demand = 0
    H_pref = 0
    H_unavail = 0

    # Objective: minimize UNFAIRNESS
    # Hfair = ∑ᵢ₌₁ᴾ (∑ⱼ₌₁ˢ xᵢⱼ − S/P)²                 S = n_demand, P = n_physicians
    max_shifts_per_p = int((n_demand/n_physicians)+0.999 ) # fair distribution of shifts
    for p in range(n_physicians):
        H_fair_s_sum_p = sum(x_symbols[p][s] for s in range(n_shifts))   
        H_fair_p = (H_fair_s_sum_p - max_shifts_per_p)**2   
        H_fair += H_fair_p

    if cl>=2:
        # Objective: minimize PREFERENCE dissatisfaction
        for p in range(n_physicians): 
            prefer_p = physician_df['prefer'].iloc[p]
            if prefer_p != '[]':
                prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
                H_pref_p = sum(x_symbols[p][int(s)] for s in prefer_shifts_p) # Reward prefered shifts (negative penalties)
                H_pref -= H_pref_p 

            prefer_not_p = physician_df['prefer not'].iloc[p]
            if prefer_not_p != '[]':
                prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  
                H_pref_not_p = sum(x_symbols[p][int(s)] for s in prefer_not_shifts_p) # Penalize unprefered shifts
                H_pref += H_pref_not_p

            # UNAVAILABLE constraint
            unavail_shifts_p = physician_df['unavailable'].iloc[p]
            if unavail_shifts_p != '[]':
                unavail_shifts_p = unavail_shifts_p.strip('[').strip(']').split(',')  
                H_unavail_p = sum(x_symbols[p][int(s)] for s in unavail_shifts_p)
                H_unavail += H_unavail_p

    if cl>=3:
        # COMPETENCE constraint
        pass

    # Constraint: Meet DEMAND
    # ∑s=1 (demanded – (∑p=1  x_ps))^2
    for s in range(n_shifts): 
        demand_s = shifts_df['demand'].iloc[s]
        workers_s = sum(x_symbols[p][s] for p in range(n_physicians))   
        H_meet_demand_s = (workers_s-sp.Integer(demand_s))**2 
        H_meet_demand += H_meet_demand_s

    # Combine all to one single H
    # H = λ₁H_fair + λ₂H_pref + λ₃H_meetDemand + ...
    if prints:
        print('Hdemand', sp.nsimplify(sp.expand(H_meet_demand*lambdas['demand'])))
        print('Hfair', sp.nsimplify(sp.expand(H_fair*lambdas['fair'])))

    all_hamiltonians = sp.nsimplify(sp.expand(H_meet_demand*lambdas['demand'] + H_fair*lambdas['fair'] + H_pref*lambdas['pref'] + H_unavail*lambdas['unavail']))
    return all_hamiltonians, x_symbols

def objectivesToQubo(all_hamiltonians, x_symbols, cl, mirror=True):
    shifts_df = pd.read_csv(f'data/intermediate/shift_data.csv')
    n_shifts = len(shifts_df)
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    n_physicians = len(physician_df)
    
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
                    print('\n\nTHIS SHOULD NOT OCCUR, SOMETHING WRONG IN MAKEQUBO()\n\n')   #Q[idx1, idx1] += coeff  # Self-interaction terms

    # Save Q to csv
    Q_df = pd.DataFrame(Q, index=None)
    Q_df.to_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv', index=False, header=False)

    return Q