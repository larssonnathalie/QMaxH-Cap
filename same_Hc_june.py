
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



#Qubo_full = generateFullQubo(demands, cl, lambdas, all_shifts_df, makeObjectiveFunctions, objectivesToQubo)
#Qubo_full_df = pd.DataFrame(Qubo_full)
#Qubo_full_df.to_csv('data/intermediate/Qubo_full_june.csv', index=None, header=False)
#TESTING________

from qaoa.testQandH import assignVariables
#Objectives, x_symbols_full_T = makeObjectivesSeparated(demands, 0, 1, cl, lambdas, time_period='all')
#gurobi_bs = "101100101100000111101111100101111001101111011001001111100111101111111101110000101011110110101111110011100110110000110010111111011110011101110111100001111001011111001111110111010101101111111001011001110101110010111110001111101110100011110011111111111000011111001111000101100011111111111001001110110110011111100110110111111011101100011110100011001110111110110011011101111010101111001110101111100111111011111000110010111110"
#ibm_bs = "000101011111100011011011011001101000100010110000001110011010010111011001101111101010011110100110110111100001011000110010111101001110001111110110000101110101110011010111100110011100101110000011001000010010101011000111000111000111110001111001011111111001010110001111001011100010111001111000001100111011001110110111110011011000111001001100100110001011100100100011000011000011011101000110100011100110110011101011110011110111"
#aer_bs = "000101001111100110101111100001111010100010000000001011010111100101011101111100010011110010001111001011100111111000110100011001011110101001100110100001111101010001100111100110111000100011100001011001010111001110111101000111100111010011100001011100111001011010001111001110101011110000111001001110101101011111100110111011011001001000001001101010011110111110100011011001010010000101001100001010101111000011101001011111011110"
with open(f'data/results/increasing_qubits/distributions/random_7phys_49vars_time-1746460257.json', "r") as f:
    randoms = json.load(f)
    f.close()


def bitstringToVector(bs):
    vector = np.zeros(len(bs))
    for count,bit in enumerate(bs):
        vector[count] = int(bit)
    return vector

Q = pd.read_csv('data/intermediate/Qubo_incr_7phys.csv',header=None).to_numpy() # UNIVERSAL Q
b = - sum(Q[i,:] + Q[:,i] for i in range(Q.shape[0]))
Hc = QToHcTEST(Q, b)

xQx_sort = {}
Hc_sort = {}
n_strings = 1000
for i, bitstring in enumerate(list(randoms.keys())[:n_strings]):
    x = bitstringToVector(bitstring)
    xQx = x @ Q @ x
    xQx_sort[i] = int(xQx)
    Hc_i = int(np.real(costOfBitstring(bitstring, Hc)))
    Hc_sort[i] = Hc_i

sorted_xQx = dict(sorted(xQx_sort.items(), key=lambda item: item[1]))
sorted_Hc = dict(sorted(Hc_sort.items(), key=lambda item: item[1]))
for i in range(n_strings):  
    print(list(sorted_xQx.values())[i], list(sorted_Hc.values())[i])

#names = ['gurobi', 'ibm', 'aer']
#objectives_test, x_symbols_full_T = makeObjectiveFunctions(demands, 0, 1, cl, lambdas, time_period='all')


"""substitution_g = assignVariables(gurobi_bs, x_symbols_full_T)
objectives_sum_g = objectives_test.subs(substitution_g)
print('g obj', objectives_sum_g)

substitution_ibm = assignVariables(ibm_bs, x_symbols_full_T)
objectives_sum_ibm = objectives_test.subs(substitution_ibm)
print('ibm obj',objectives_sum_ibm)

substitution_aer = assignVariables(aer_bs, x_symbols_full_T)
objectives_sum_aer = objectives_test.subs(substitution_aer)
print('aer obj', objectives_sum_aer)

print('min:', np.argmin([objectives_sum_g, objectives_sum_ibm, objectives_sum_aer]))"""

#Qubo_test = objectivesToQubo(objectives_test, n_shifts, x_symbols_full_T, cl, mirror=False) #TESTING TRUE
"""
Qubo_test =pd.read_csv('data/intermediate/Qubo_full_june.csv',header=None).to_numpy() # UNIVERSAL Q
b = - sum(Qubo_test[i,:] + Qubo_test[:,i] for i in range(Qubo_test.shape[0]))

xtqx = {}
print('xT Q x')
x_g = bitstringToVector(gurobi_bs)
#print('g',x_g @ Qubo_test @ x_g)
xtqx['g'] = int(x_g @ Qubo_test @ x_g)

x_ibm = bitstringToVector(ibm_bs)
#print('ibm',x_ibm @ Qubo_test @ x_ibm)
xtqx['ibm'] =int( x_ibm @ Qubo_test @ x_ibm)

x_aer = bitstringToVector(aer_bs)
#print('aer',x_aer @ Qubo_test @ x_aer)
xtqx['aer'] = int(x_aer @ Qubo_test @ x_aer)
sorted_xt = dict(sorted(xtqx.items(), key=lambda item: item[1]))
print(sorted_xt)

print('Hc')
Hc_test = QToHcTEST(Qubo_test, b)
Hc_dict = {}
#print('gurobi Hc', costOfBitstring(gurobi_bs, Hc_test))
Hc_dict['g'] = int(np.real(costOfBitstring(gurobi_bs, Hc_test)))

#print('ibm Hc', costOfBitstring(ibm_bs, Hc_test))
Hc_dict['ibm'] = int(np.real(costOfBitstring(ibm_bs, Hc_test)))

#print('aer Hc', costOfBitstring(aer_bs, Hc_test))
Hc_dict['aer'] = int(np.real(costOfBitstring(aer_bs, Hc_test)))
sorted_Hc = dict(sorted(Hc_dict.items(), key=lambda item: item[1]))
print(sorted_Hc)"""

"""
bitstrings = [gurobi_bs, ibm_bs, aer_bs]
i = 0

for bs in bitstrings:
    print()
    print(names[i])
    substitution = assignVariables(bs, x_symbols_full_T)
    print('subs')
    demand = Objectives['demand'].subs(substitution)
    print('dem')

    ext = Objectives['extent'].subs(substitution)
    print('ext')

    titles = Objectives['titles'].subs(substitution)
    print('tit')

    unavail = Objectives['unavail'].subs(substitution)
    print('un')

    pref = Objectives['pref'].subs(substitution)
    print('pref')

    fair = Objectives['fair'].subs(substitution)

    print('demand', demand)
    print('ext', ext)
    print('titles', titles)
    print('unavail', unavail)
    print('pref', pref)
    print('fair', fair)
    i+=1


#Results:
#GUROBI:
#demand 84
#ext 3568.00000000000
#titles 29.8660714285718
#unavail 0
#pref -355
#fair 0

#IBM
#demand 111
#ext 4480.00000000000
#titles 7.81250000000023
#unavail 60
#pref -190
#fair 0

#AER
#demand 87
#ext 2720.00000000000
#titles 6.56250000000022
#unavail 15
#pref -270
#fair 0
"""
'''Q_full_T = objectivesToQubo(all_hamiltonians_full_T, len(all_shifts_df),x_symbols_full_T, cl, mirror=False )

#TEST_______________
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
'''