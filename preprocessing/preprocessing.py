import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from qiskit_algorithms.optimizers import COBYLA

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
    calendar_df.to_csv(f'data/intermediate/empty_calendar.csv', index=False)
    if prints:
        print(calendar_df)
    

# Automatically generate data in "demand_clX.csv", following repeating demand rules based on weekdays/holidays
# cl stands for complexity level
def generateDemandData(empty_calendar, cl, weekday_workers=2, holiday_workers=1, prints=True):

    demand_col = [holiday_workers + int(empty_calendar.loc[i,'is_holiday']==False)*(weekday_workers-holiday_workers) for i in range(len(empty_calendar))] #[1]*len(empty_calendar)             
    df = pd.DataFrame({'date':empty_calendar['date'], 'demand': demand_col})

    df.to_csv(f'data/intermediate/demand.csv', index=False)


# preference column with dates in phys.csv + calendar info --> undesired combinations p & s
# (TODO add importance level or divide into definetely unavailable & prefer not)
# (TODO add reocurring preferences like thursday afternoons etc.?)
# TODO decide rules for preferences. Examples:
    # all p must have x prefered dates --> Maximize equal n.o. satisfied preferences
    # all p have varying number of prefered dates --> maximize equal share of satisfied preference, over time
    # maximize over-all satisfaction, ex. someone has many easily solved preferences and someone else has one that is difficult to solve, go for many easy ones?
    # other..?
def convertPreferences(empty_calendar_df):
    # Converts preferences from dates to shift numbers, if the dates are in the schedule
    date_to_s = {empty_calendar_df.loc[i,'date']:i for i in range(empty_calendar_df.shape[0])} 
    included_dates = list(empty_calendar_df['date'])
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv') 
    n_physicians = physician_df.shape[0]

    prefer_shifts_col = ['']*n_physicians
    prefer_not_shifts_col = ['']*n_physicians
    unavailable_shifts_col = ['']*n_physicians

    for p in range(n_physicians): 
        prefer_dates = physician_df.loc[p,'Prefer']
        prefer_shifts = []
        if type(prefer_dates) != float and type(prefer_dates) != np.float64: # if not empty  # TODO better if statement (x3)
            prefer_dates = prefer_dates.split(',')
            prefer_shifts.append([date_to_s[date] for date in prefer_dates if date in included_dates])
            prefer_shifts_col[p]=prefer_shifts[0] #TODO remove brackets in csv

        prefer_not_dates = physician_df.loc[p,'Prefer Not']
        prefer_not_shifts =[]
        if type(prefer_not_dates) != float and type(prefer_not_dates) != np.float64: # if not empty
            prefer_not_dates = prefer_not_dates.split(',') 
            prefer_not_shifts.append([date_to_s[date] for date in prefer_not_dates if date in included_dates])
            prefer_not_shifts_col[p]= prefer_not_shifts[0] # TODO remove brackets in csv
        
        unavailable_dates = physician_df.loc[p,'Unavailable']
        unavailable_shifts =[]
        if type(unavailable_dates) != float and type(unavailable_dates) != np.float64: # if not empty       
            unavailable_dates = unavailable_dates.split(',') 
            unavailable_shifts.append([date_to_s[date] for date in unavailable_dates if date in included_dates])
            unavailable_shifts_col[p]= unavailable_shifts[0] # TODO remove brackets in csv

    physician_df['Prefer'] = prefer_shifts_col
    physician_df['Prefer Not'] = prefer_not_shifts_col
    physician_df['Unavailable'] = unavailable_shifts_col
    physician_df.to_csv(f'data/intermediate/physician_data.csv', index=None)

