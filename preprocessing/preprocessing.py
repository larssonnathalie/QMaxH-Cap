import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
#from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
#from qiskit.primitives import Sampler, StatevectorSampler, Estimator   REPLACED WITH SAMPLERV2 FROM QISKIT_IBM_RUNTIME


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
    # maximize over-all satisfaction, ex. someone has many easily solved preferences and someone else has one that is difficult to solve, go for many easy ones?
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

