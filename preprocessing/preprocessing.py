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

# Automatically generate physicians_data with random preferences on relevant dates
def generatePhysicianData(empty_calendar, n_physicians, seed=True):
    all_shifts = empty_calendar['date']
    n_shifts = len(all_shifts)
    possible_extents = [25,50,50,75,75,100,100,100,100,100,100]  # NOTE just an example, more copies --> more likely
    possible_titles = ['UL', 'ÖL', 'AT', 'ST', 'Chef'] # NOTE should maybe be at least one of some, make choice without replacement
    
    name_col = []
    extent_col = []
    prefer_col = []
    prefer_not_col = []
    unavail_col = []
    title_col = [] 

    if seed:
        np.random.seed(12)

    for p in range(n_physicians):
        remaining_shifts_p = list(all_shifts)
        name_col.append(f'Physician{p}')
        extent_col.append(np.random.choice(possible_extents))
        title_col.append(np.random.choice(possible_titles))

        prefer_not_p = np.random.choice(remaining_shifts_p, size=np.random.randint(0, n_shifts // 2 + 1), replace=False) # NOTE maybe change upper size limit 
        prefer_not_col.append(list(prefer_not_p))
        for s in prefer_not_p:
            remaining_shifts_p.remove(s)

        prefer_p = np.random.choice(remaining_shifts_p, size=np.random.randint(0, len(remaining_shifts_p)), replace=False)
        prefer_col.append(list(prefer_p))

        for s in prefer_p:
            remaining_shifts_p.remove(s)

        unavail_p = np.random.choice(remaining_shifts_p, size=np.random.randint(0, len(remaining_shifts_p)), replace=False)
        unavail_col.append(list(unavail_p))
    
    physician_data_df = pd.DataFrame({'Name':name_col, 'Extent': extent_col, 'Prefer':prefer_col, 'Prefer Not':prefer_not_col, 'Unavailable':unavail_col, 'Title':title_col})
    physician_data_df.to_csv('data/intermediate/physician_data.csv', index=None)


    

# Automatically generate data in "demand.csv", following repeating demand rules based on weekdays/holidays
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
            prefer_dates = prefer_dates.strip(']').strip('[').split(',')
            for date in prefer_dates:
                date = date.strip('"').strip("'")
                if date in included_dates:
                    prefer_shifts.append(date_to_s[date])
            prefer_shifts_col[p]=prefer_shifts #TODO remove brackets in csv? x3


        prefer_not_dates = physician_df.loc[p,'Prefer Not']
        prefer_not_shifts =[]
        if type(prefer_not_dates) != float and type(prefer_not_dates) != np.float64: # if not empty
            prefer_not_dates = prefer_not_dates.strip(']').strip('[').split(',') 
            for date in prefer_not_dates:
                date = date.strip('"').strip("'")
                if date in included_dates:
                    prefer_not_shifts.append(date_to_s[date])
            prefer_not_shifts_col[p]= prefer_not_shifts 
        
        unavailable_dates = physician_df.loc[p,'Unavailable']
        unavailable_shifts =[]
        if type(unavailable_dates) != float and type(unavailable_dates) != np.float64: # if not empty       
            unavailable_dates = unavailable_dates.strip(']').strip('[').split(',') 
            for date in unavailable_dates:
                date = date.strip('"').strip("'")
                if date in included_dates:
                    unavailable_shifts.append(date_to_s[date])            
            unavailable_shifts_col[p]= unavailable_shifts 

    physician_df['Prefer'] = prefer_shifts_col
    physician_df['Prefer Not'] = prefer_not_shifts_col
    physician_df['Unavailable'] = unavailable_shifts_col
    physician_df.to_csv(f'data/intermediate/physician_data.csv', index=None)

