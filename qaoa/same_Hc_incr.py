
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.qaoa import *

# SAVE QUBO FOR INCR RUNS, TO ENSURE SAME HC

# Parameters
start_date = '2025-06-22' 
end_date = '2025-06-28'
n_shifts = 7
cl = 3               # complexity level: 
n_physicians =  119  #,5,6,7,10,14,(17), 21,28,35,42,49,70,119
preference_seed = 10

time_period = 'all'
t = 0 # Only 1 optimization

lambdas = {'demand':3, 'fair':10, 'pref':5, 'unavail':15, 'extent':8, 'rest':0, 'titles':5, 'memory':3}  # NOTE Must be integers

T, total_holidays, n_days = emptyCalendar(end_date, start_date, cl, time_period=time_period)
all_dates_df = pd.read_csv(f'data/intermediate/empty_calendar.csv',index_col=None)

generatePhysicianData(all_dates_df, n_physicians, cl, seed=preference_seed)  
physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')

# DEMANDS
if shiftsPerWeek(cl)==7:    
    # Set from amount of workers and their extent
    target_n_shifts_total_per_week = sum(targetShiftsPerWeek(physician_df['extent'].iloc[p], cl) for p in range(n_physicians)) 
    target_n_shifts_total = target_n_shifts_total_per_week * (len(all_dates_df) / shiftsPerWeek(cl))

    demand_hd = max(target_n_shifts_total_per_week//12, 1)
    demand_wd = max((target_n_shifts_total_per_week - 2*demand_hd)//5, 1)
    demands = {'weekday': demand_wd, 'holiday': demand_hd}  
    print('demands:', demands)

# SHIFTS
generateShiftData(all_dates_df, T, cl, demands, time_period=time_period)
all_shifts_df = pd.read_csv(f'data/intermediate/shift_data_all_t.csv',index_col=None)

convertPreferences(all_shifts_df, t)   # Dates to shift-numbers
shifts_df = all_shifts_df

Qubo_full = generateFullQubo(demands, cl, lambdas, all_shifts_df, makeObjectiveFunctions, objectivesToQubo)
Qubo_full_df = pd.DataFrame(Qubo_full)
Qubo_full_df.to_csv(f'data/intermediate/Qubo_incr_{n_physicians}phys.csv', index=None, header=False)
