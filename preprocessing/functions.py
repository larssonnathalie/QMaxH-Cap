# from data Import shift time info, preferences, constraints etc.
import pandas as pd
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
    nPhysicians, nShifts = physician_df.shape[0], demand_df.shape[0]    # NOTE assuming 1 shift per row

    if cl <=1:
        nVars = nPhysicians * nShifts # n.o. decision variables 

        # USING qiskit QP    
        qubo = QuadraticProgram()     # TODO Move section to makeQubo or add bitstring encoding
        linear_weights={}             # TODO Demand based on demand_df
        two_on_same={}
        for i in range(nPhysicians): # TODO remove loops?
            for j in range(nShifts):
                qubo.binary_var(name='x{}{}'.format(i,j))
                if i==j:
                    linear_weights['x{}{}'.format(i,j)] = -1 # TODO diagonal weights value
                elif (j-i)%nShifts ==0:
                    two_on_same['x{}{}'.format(i,j)] = 2
        qubo.minimize(linear=linear_weights)
        qubo.minimize(quadratic=two_on_same)


        # USING dictionary:
        Q_oneEachDay = {}
        for i in range(nVars):
            for j in range(i,nVars):
                if i==j:
                    Q_oneEachDay[(i,j)] = -1  # Minimize diagonal variables (why -1?)
                elif (j-i)%nShifts ==0:
                    Q_oneEachDay[(i,j)] = 2   # Penalize 2 docs on same shift
        

    else: 
        print('constructObjectives is not coded yet for cl', str(cl))

    if prints:
        print('Physicians:', nPhysicians)
        print('Shifts:', nShifts)
        print(qubo)

    return qubo


# construct QUBO from objective functions
def makeQubo(objectives,  lamda_fair = 1, lamda_pref=1, prints=True)->QuadraticProgram:
    encoding = []
    qubo = [] # type QuadraticProgram
    return qubo, encoding