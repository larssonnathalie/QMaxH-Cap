# from data Import shift time info, preferences, constraints etc.
import pandas as pd
import holidays
from datetime import datetime
from qiskit_optimization import QuadraticProgram


# construct empty calendar with work days, holidays etc
def emptyCalendar(end_date, prints=True):
    current_year = datetime.now().year

    all_dates = pd.date_range(start=f'{current_year}-01-01', end=end_date)
    swedish_holidays = holidays.Sweden(years=current_year) # TODO start and end at specific dates
    calendar_df = pd.DataFrame({'date': all_dates, 'is_holiday': all_dates.isin(swedish_holidays)})
    return calendar_df



# construct objective functions for fairness, preference
# and constraints such as 1 person per shift & max shifts per week
def constructObjectives(empty_calendar_df, lamda_fair = 1, lamda_pref=1, preferences=True, prints=True):
    return []


# construct QUBO from objective functions


def makeQubo(objectives, prints=True)->QuadraticProgram:
    encoding = []
    qubo = [] # type QuadraticProgram
    return qubo, encoding