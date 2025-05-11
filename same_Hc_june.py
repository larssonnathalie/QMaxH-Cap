
from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.qaoa import *

# SAVE QUBO FOR JUNE RUNS, TO ENSURE SAME HC

# Parameters
start_date = '2025-06-01' 
end_date = '2025-06-28'
n_shifts = 28
n_physicians = 15
cl = 3               # complexity level: 

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
physician_universal_june = pd.read_csv('data/intermediate/physician_data.csv', index_col = None)
#physician_universal_june.to_csv('data/intermediate/physician_universal_june.csv', index=None)

Qubo_full = generateFullQubo(demands, cl, lambdas, all_shifts_df, makeObjectiveFunctions, objectivesToQubo)
Qubo_full_df = pd.DataFrame(Qubo_full)
Qubo_full_df.to_csv('data/intermediate/Qubo_full_june.csv', index=None, header=False)

Qubo_aer_june = pd.read_csv('data/intermediate/Qubo_matrix_cl3.csv', header=None).to_numpy()
Qubo_universal_june = pd.read_csv('data/intermediate/Qubo_full_june.csv',header=None).to_numpy()

print(Qubo_aer_june.shape)
if Qubo_aer_june.shape != Qubo_universal_june.shape:
    print('Not same shapes:', Qubo_aer_june.shape ,Qubo_universal_june.shape)

else:
    for i in range(len(Qubo_aer_june)):
        for j in range(len((Qubo_aer_june))):
            if Qubo_aer_june[i][j] == Qubo_universal_june[i][j]:
                pass#print(Qubo_aer_june[i][j], Qubo_universal_june[i][j])
            else:
                print('\n DIFFERENCE:', Qubo_aer_june[i][j], Qubo_universal_june[i][j])
                break
        
        break