import pandas as pd
import numpy as np
import sympy as sp
import holidays
from datetime import datetime
from qaoa.converters import *
#from qiskit_algorithms.optimizers import COBYLA

# EMPTY CALENDAR with weekdays & holidays
def emptyCalendar(end_date, start_date, cl, time_period):
    all_dates = pd.date_range(start=start_date, end=end_date)
    n_days = len(all_dates)
    years = list(range(int(start_date[:4]),int(end_date[:4])+1))
    swedish_holidays = all_dates.isin(holidays.Sweden(years=years)) 
    weekdays = all_dates.strftime('%A').values
    saturdays = weekdays =='Saturday'
    holidays_and_weekends = swedish_holidays+saturdays>= 1 # TODO replace with better OR function
    total_holidays = np.sum(holidays_and_weekends)
    #if len(all_dates)%7!=0 and cl<3:
        #print('\nWarning: Dates should be a [int] number of weeks for cl<3\n') # (start on tuesday end on monday works too)
    
    get_T = {'shift':n_days+(n_days*2*int(cl>=3)), 'day':n_days, 'week':(n_days+6)//7}
    T = get_T[time_period]
    print('\nNEW OPTIMIZATION  EACH '+time_period)
    print(str(T)+' '+time_period+':s')
    print('\ntotal holidays:',total_holidays)

    calendar_df= pd.DataFrame({'date': all_dates, 'is_holiday': holidays_and_weekends, 'weekday':weekdays}) 
    
    if cl>=3:
        # 3-SHIFT
        date_col, shift_type_col, holiday_col, weekday_col = [], [], [], []
        for i, date in enumerate(calendar_df['date']):
            holi, weekday = calendar_df[['is_holiday','weekday']].iloc[i]
            for type in ['dag', 'kväll', 'natt']:
                date_col += [date] 
                shift_type_col += [type]
                holiday_col += [holi]
                weekday_col += [weekday]
        
        calendar_df = pd.DataFrame({'date': date_col, 'shift type':shift_type_col, 'weekday':weekday_col,'is_holiday': holiday_col})
    calendar_df.to_csv(f'data/intermediate/empty_calendar.csv', index=False)

    return T, total_holidays, n_days

# PHYSICIAN DATA with random preferences on relevant dates
def generatePhysicianData(empty_calendar, n_physicians, cl, seed=True, only_fulltime=False):
    
    #TODO look over all probabilities in random choices and set realistic values

    all_dates = empty_calendar['date'] # TODO preferences on separate shifts instead of dates
    n_dates = len(list(set(list(all_dates))))
    possible_extents = [25,50,50,75,75,100,100,100,100,100,100]  # more copies -> more likely
    if only_fulltime:
        possible_extents = [100]
    possible_titles = ['ÖL', 'ST', 'AT','Chef','UL'] * (n_physicians//5+1)

    name_col = []
    extent_col=[]
    prefer_col = [[] for _ in range(n_physicians)]
    prefer_not_col = [[] for _ in range(n_physicians)]
    unavail_col = [[] for _ in range(n_physicians)]
    title_col = [] 
    worked_shifts_col = [0 for _ in range(n_physicians)] 
    work_rate_col = [0 for _ in range(n_physicians)] 
    satisfaction_col = [0 for _ in range(n_physicians)] 
    worked_last_col = [0 for _ in range(n_physicians)] 

    if seed:
        np.random.seed(56)
    
    for p in range(n_physicians):
        remaining_dates_p = list(set(list(all_dates)))
        name_col.append(f'physician{p}')
        extent_col.append(np.random.choice(possible_extents))
        title_col.append(possible_titles[p]) 

        if cl >=2:
            # RANDOM PREFERENCES
            size = np.random.randint(0, n_dates//2)
            prefer_not_p = np.random.choice(remaining_dates_p, size=size, replace=False) # NOTE temporary upper size limit 
            prefer_not_col[p] =(list(prefer_not_p))
            for s in prefer_not_p:
                remaining_dates_p.remove(s)

            if len(remaining_dates_p)>0:
                size = np.random.randint(0, len(remaining_dates_p)//2)
                prefer_p = np.random.choice(remaining_dates_p, size=size, replace=False)
                prefer_col[p]=list(prefer_p)
                for s in prefer_p:
                    remaining_dates_p.remove(s)

            if len(remaining_dates_p)>0:
                size= np.random.randint(0, len(remaining_dates_p)//2)
                unavail_p = np.random.choice(remaining_dates_p, size=size, replace=False)
                unavail_col[p] = list(unavail_p)

    physician_data_df = pd.DataFrame({'name':name_col, 'title':title_col, 'extent': extent_col, 'shifts worked':worked_shifts_col, 'work rate':work_rate_col, 'prefer':prefer_col, 'prefer not':prefer_not_col, 'unavailable':unavail_col, 'satisfaction':satisfaction_col, 'worked last':worked_last_col})
    physician_data_df.to_csv('data/intermediate/physician_data.csv', index=None)

# SHIFT DEMAND & ATTRACTIVENESS
# following repeating demand rules based on weekdays/holidays
def generateShiftData(empty_calendar, T, cl, demands, time_period):

    physician_df= pd.read_csv(f'data/intermediate/physician_data.csv')
    n_physicians = len(physician_df)
    n_shifts = len(empty_calendar) 
    
    if cl < 3:
        # DEMAND
        demand_col = [demands['holiday'] + int(empty_calendar.loc[i,'is_holiday']==False)*(demands['weekday']-demands['holiday']) for i in range(n_shifts)]          
        shift_data_df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col})

    if cl >=3: 
        # 3 SHIFTS PER DAY
        demand_col =[]
        for s in range(len(empty_calendar)):
            shift_type, is_hol = empty_calendar[['shift type','is_holiday']].iloc[s]
            demand_col.append(demands[(shift_type, is_hol)])
        shift_data_df = pd.DataFrame({'date':empty_calendar['date'], 'shift type':empty_calendar['shift type'], 'demand':demand_col})

    if cl >=2: 
        # SHIFT ATTRACTIVENESS
        prefer = {p:physician_df.loc[p,'prefer'].strip('[]').split(',') for p in range(n_physicians)} 
        prefer_not = {p:physician_df.loc[p,'prefer not'].strip('[]').split(',') for p in range(n_physicians)}
        unavailable = {p:physician_df.loc[p,'unavailable'].strip('[]').split(',') for p in range(n_physicians)}

        prefer_counts, prefer_not_counts, unavailable_counts = [],[],[] 
        for p in range(n_physicians):
            prefer_counts += [s.strip(" '") for s in prefer[p]]
            prefer_not_counts += [s.strip(" '") for s in prefer_not[p]]
            unavailable_counts += [s.strip(" '") for s in unavailable[p]]

        attractiveness_col = np.zeros(n_shifts)
        for s, date in enumerate(empty_calendar['date']):
            n_prefer_s = prefer_counts.count(date)
            n_prefer_not_s = prefer_not_counts.count(date)
            n_unavailable_s = unavailable_counts.count(date)
            attractiveness_col[s] = (n_prefer_s - n_prefer_not_s)/(demand_col[s]+n_unavailable_s+0.1) # +0.1 to avoid /0

        shift_data_df['attractiveness'] = attractiveness_col
    
    # ALL T:s
    shift_data_df.to_csv('data/intermediate/shift_data_all_t.csv', index=False)

    # ONE FILE PER t
    shifts_per_t = getShiftsPerT(time_period, cl)

    for t in range(T):
        start_idx,stop_idx = t*shifts_per_t, min((t+1)*shifts_per_t, len(shift_data_df))
        shift_data_df_w = shift_data_df.iloc[start_idx:stop_idx]
        shift_data_df_w.to_csv(f'data/intermediate/shift_data_t{t}.csv', index=False)

# PREFERENCES from dates to shift-numbers
def convertPreferences(empty_calendar_df, t, only_prefer=False):
    shifts_df = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
    n_shifts=len(shifts_df)
    date_to_s={}
    for s in range(n_shifts):
        date = shifts_df.loc[s,'date']
        if date in date_to_s.keys():
            date_to_s[date].append(s)
        else:
            date_to_s[date] = [s]
    included_dates = list(shifts_df['date'])
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv') 
    n_physicians = physician_df.shape[0]

    prefer_shifts_col = ['']*n_physicians
    prefer_not_shifts_col = ['']*n_physicians
    unavailable_shifts_col = ['']*n_physicians

    for p in range(n_physicians): 
        prefer_dates = physician_df.loc[p,'prefer']
        prefer_shifts = []
        if prefer_dates !='[]':
            prefer_dates = prefer_dates.strip('[]').split(',')

            for date in prefer_dates:
                date = date.strip('"').strip(' ').strip("'")

                if date in included_dates:
                    for s in date_to_s[date]:
                        prefer_shifts.append(s)
        prefer_shifts_col[p]=prefer_shifts 

        prefer_not_dates = physician_df.loc[p,'prefer not']
        prefer_not_shifts =[]
        if prefer_not_dates!='[]' and not only_prefer:
            prefer_not_dates = prefer_not_dates.strip(']').strip('[').split(',') 
            for date in prefer_not_dates:
                date = date.strip('"').strip(' ').strip("'")
                if date in included_dates:
                    for s in date_to_s[date]:
                        prefer_not_shifts.append(s)
        prefer_not_shifts_col[p]= prefer_not_shifts 
        
        unavailable_dates = physician_df.loc[p,'unavailable']
        unavailable_shifts =[]
        if unavailable_dates!='[]' and not only_prefer:  
            unavailable_dates = unavailable_dates.strip(']').strip('[').split(',') 
            for date in unavailable_dates:
                date = date.strip('"').strip(' ').strip("'")
                if date in included_dates:
                    for s in date_to_s[date]:
                        unavailable_shifts.append(s)   
        unavailable_shifts_col[p]= unavailable_shifts 

    physician_df[f'prefer t{t}'] = prefer_shifts_col
    physician_df[f'prefer not t{t}'] = prefer_not_shifts_col
    physician_df[f'unavailable t{t}'] = unavailable_shifts_col
    physician_df.to_csv(f'data/intermediate/physician_data.csv', index=None)

def makeObjectiveFunctions(n_demand, t, T, cl, lambdas, time_period, prints=False):
    # Both objectives & constraints formulated as Hamiltonians to be combined to QUBO form
    # Using sympy to simplify the H expressions

    shifts_df = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
    n_shifts = len(shifts_df)
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    n_physicians = len(physician_df)
    
    # define decision variables (a list of lists)
    x_symbols = []
    for p in range(n_physicians):
        x_symbols_p = [sp.symbols(f'x{p}_{s}') for s in range(n_shifts)]
        x_symbols.append(x_symbols_p)
    
    H_fair = 0
    H_extent = 0
    H_meet_demand = 0
    H_pref = 0
    H_unavail = 0
    H_rest = 0
    H_titles = 0

    # minimize UNFAIRNESS
    if cl == 1:
        if T !=1:
            print('nERROR in makeObjectives..(): cl 1 not implemented for more than one t')  
            return 
        
        # Hfair = ∑ᵢ₌₁ᴾ (∑ⱼ₌₁ˢ xᵢⱼ − S/P)²                 S = n_demand, P = n_physicians
        max_shifts_per_p = int((n_demand/n_physicians)+0.999 ) # fair distribution of shifts
        for p in range(n_physicians):
            H_fair_s_sum_p = sum(x_symbols[p][s] for s in range(n_shifts))   
            H_fair_p = (H_fair_s_sum_p - max_shifts_per_p)**2   
            H_fair += H_fair_p

    else: # (cl > 1)
        if T ==1:
            # Minimize PREFERENCE dissatisfaction
            for p in range(n_physicians): 
                prefer_p = physician_df[f'prefer t{t}'].iloc[p]
                if prefer_p != '[]':
                    prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
                    H_pref_p = sum(x_symbols[p][int(s)] for s in prefer_shifts_p) # Reward prefered shifts (negative penalties)
                    H_pref -= H_pref_p 

                prefer_not_p = physician_df[f'prefer not t{t}'].iloc[p]
                if prefer_not_p != '[]':
                    prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  
                    H_pref_not_p = sum(x_symbols[p][int(s)] for s in prefer_not_shifts_p) # Penalize unprefered shifts
                    H_pref += H_pref_not_p

                # UNAVAILABLE constraint
                unavail_shifts_p = physician_df[f'unavailable t{t}'].iloc[p]
                if unavail_shifts_p != '[]':
                    unavail_shifts_p = unavail_shifts_p.strip('[').strip(']').split(',')  
                    H_unavail_p = sum(x_symbols[p][int(s)] for s in unavail_shifts_p)
                    H_unavail += H_unavail_p
                    max_shifts_per_p = int((n_demand/n_physicians)+0.999 ) # fair distribution of shifts

        else: 
            # long term SATISFACTION fairness
            equality_priority = t/T        # Last t:s: prioritize fairness
            optimize_priority = 1-equality_priority # First weeks: optimize overall satisfaction
            min_prio = 0.05 * optimize_priority      # so the most satisfied p still has some priority. (First week this applies to all p)

            prefer = {p:physician_df.loc[p,f'prefer t{t}'].strip('[').strip(']').split(',') for p in range(n_physicians)}
            prefer_not = {p:physician_df.loc[p,f'prefer not t{t}'].strip('[').strip(']').split(',') for p in range(n_physicians)}
            unavailable = {p:physician_df.loc[p,f'unavailable t{t}'].strip('[').strip(']').split(',') for p in range(n_physicians)}

            if t == 0:
                satisfaction = np.ones(n_physicians)
            else:
                satisfaction = np.array([float(sat) for sat in physician_df['satisfaction']])
                min_sat = np.min(satisfaction)
                satisfaction = satisfaction - min_sat + 1   # shift whole column to set least satisfied = 1
            satisfaction_rate = satisfaction/np.max(satisfaction) 
            #print(satisfaction_rate,'\trates')
            priority = np.where((1 - satisfaction_rate)>min_prio, (1 - satisfaction_rate), min_prio)   # less satisfied are more important & apply min_prio
            #print(priority, '\tprios')

            for p in range(n_physicians):
                priority_p = priority[p]

                for s in prefer[p]:
                    if s != '':
                        H_fair -= priority_p * x_symbols[p][int(s)]**2  # reward prefered shifts
                        #print(p, 'prefered', s, 'has priority', priority_p)

                for s in prefer_not[p]:
                    if s != '':
                        H_fair += priority_p * x_symbols[p][int(s)]**2  # penalize unprefered shifts
                        #print(p, 'prefered not', s, 'has priority', priority_p)

                for s in unavailable[p]:
                    if s != '':
                        H_unavail += x_symbols[p][int(s)]**2 # penalize assigning unavailable
        
        
        # EXTENT
        if time_period == 'shift' or time_period =='day':
            days_passed = getDaysPassed(t, time_period)
            extent_priority =min(days_passed/7, 1) # Extent is less important first days, so not all are assigned the first shifts
            for p in range(n_physicians):
                work_rate_p = physician_df['work rate'].iloc[p]
                    #print(p, 'has work rate', work_rate_p)
                priority_p = abs(extent_priority * (1 - float(work_rate_p) ))
                    #print('priority\t', priority_p)

                if work_rate_p < 1:
                    if prints:
                        print(p,'´s work rate is',work_rate_p)
                    for s in range(n_shifts):
                        H_extent -= priority_p * x_symbols[p][s]**2  # Reward shift assignment to p:s who have low work rate
                
                elif work_rate_p >=1:
                    for s in range(n_shifts):
                        H_extent += priority_p * x_symbols[p][s]**2 # penalize shift assignment to p:s who have high work rate


        elif time_period == 'week':
            for p in range(n_physicians):
                extent_p = int(physician_df['extent'].iloc[p])
                n_shifts_target = targetShiftsPerWeek(extent_p, cl)
                assigned_shifts = sum(x_symbols[p][s] for s in range(n_shifts))
                H_extent += (assigned_shifts-n_shifts_target)**2
    
        if cl>=3:
            # REST between shifts
            for p in range(n_physicians):
                worked_last = int(physician_df['worked last'].iloc[p]) 
                if worked_last == 1:
                    H_rest += x_symbols[p][0]**2 # penalize first shift if p worked last shift in previous t
                
                if n_shifts>1:
                    for s in range(1,n_shifts):
                        H_rest += x_symbols[p][s]*x_symbols[p][s-1] # penalize working two following shift

            if cl >=4:
                # TODO TITLES constraint
                    # TODO assign boss to necessary tasks
                pass


    # DEMAND
    # ∑s=1 (demanded – (∑p=1  x_ps))^2
    for s in range(n_shifts): 
        demand_s = shifts_df['demand'].iloc[s]
        print('demand',demand_s)
        workers_s = sum(x_symbols[p][s] for p in range(n_physicians))   
        H_meet_demand_s = (workers_s-sp.Integer(demand_s))**2 
        H_meet_demand += H_meet_demand_s
    
    H_fair = sp.expand(H_fair)
    H_extent = sp.expand(H_extent)
    H_meet_demand = sp.expand(H_meet_demand)
    H_pref = sp.expand(H_pref)
    H_unavail = sp.expand(H_unavail)
    H_rest = sp.expand(H_rest)
    if prints:
        print('H demand:', sp.simplify(H_meet_demand*lambdas['demand']))
        #print('\nH fair:', sp.simplify(H_fair*lambdas['fair']))
        print('\nH extent:', sp.simplify(H_extent*lambdas['extent']))

    # Combine all to one single H
    # H = λ₁H_fair + λ₂H_pref + λ₃H_meetDemand + ...
    all_hamiltonians = sp.nsimplify(sp.expand(H_meet_demand*lambdas['demand'] + H_fair*lambdas['fair'] + H_pref*lambdas['pref'] + H_unavail*lambdas['unavail'] + H_extent*lambdas['extent'] + H_rest*lambdas['rest']))
    return all_hamiltonians, x_symbols

def objectivesToQubo(all_hamiltonians, n_shifts, x_symbols, cl, mirror=True, prints=False):
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    n_physicians = len(physician_df)
    
    x_list = {(f'{p}_{s}'):x_symbols[p][s] for p in range(n_physicians) for s in range(n_shifts)} #TEST

    n_vars = n_physicians*n_shifts
    Q = np.zeros((n_vars,n_vars))

    for term in all_hamiltonians.as_ordered_terms():
        coeff, variables = term.as_coeff_mul()
        coeff = int(coeff)
        if len(variables) == 1: # Linear terms
            ps_strings = str(variables[0]).split('_')
            p = ps_strings[0].strip('x')
            s = ps_strings[1].strip('**2')
            if p != '' and s!='':
                s, p = int(s), int(p)
                
                idx = xToQIndex((p,s),(p,s),n_shifts)
                if prints:
                    print(p,s)
                    print(idx)
                    print(f'p{p}s{s} has coeff:', coeff)
                Q[idx[0], idx[1]] += coeff  
                

        elif len(variables) == 2: # Quadratic terms
            ps_strings1 = str(variables[0]).split('_')
            #print('2 vars', type(coeff))
            p1 = ps_strings1[0].strip('x')
            s1 = ps_strings1[1].strip('**2') 

            ps_strings2 = str(variables[1]).split('_')
            p2 = ps_strings2[0].strip('x')
            s2 = ps_strings2[1].strip('**2') 
            if p1 != '' and s1 != '' and p2 != '' and s2!='':
                p1, s1, p2, s2 = int(p1), int(s1), int(p2), int(s2)
                idx1,idx2 = xToQIndex((p1,s1),(p2,s2),n_shifts)

                if idx1>idx2: # upper triangular
                    idx1, idx2 = idx2, idx1
                Q[idx1, idx2] += coeff  # Off-diagonal terms
                if prints:
                    print(f'p{p1}s{s1} * p{p2}s{s2} has coeff:', coeff)
                if idx1 != idx2:
                    if mirror:
                        Q[idx2, idx1] += coeff  # Symmetric QUBO matrix
                else:
                    print('\nTHIS SHOULD NOT OCCUR')

                
            

    # Save Q to csv
    Q_df = pd.DataFrame(Q, index=None)
    Q_df.to_csv(f'data/intermediate/Qubo_matrix_cl{cl}.csv', index=False, header=False)

    return Q