from postprocessing.postprocessing import *
import matplotlib.pyplot as plt

def combineDataJune(backend, n_physicians, timestamp):
    print(f'\Sorting data for JUNE run on {backend}, {n_physicians} physicians at time {timestamp}')

    
    phys_str = f'{n_physicians}phys_'if timestamp >= 1746621883 else ''
    physician_df = pd.read_csv(f'data/results/physician/{backend}_{phys_str}time{int(timestamp)}.csv')
    schedule_df = pd.read_csv(f'data/results/schedules/{backend}_{n_physicians}phys_time{int(timestamp)}.csv')

    json_file_path = f'data/results/runs/{backend}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(json_file_path, "r") as f:
        run_data = json.load(f)
    
    all_data = {'run':run_data, 'physician':physician_df, 'schedule':schedule_df}
    return all_data



#Compare june: (show: time, schedules, constraints, Hc:s        + Quantum: depth, 2-gates)
    # june aer 15phys
    # june ibm 15phys
    # june gurobi 15phys
    # (june z3 15phys)


backend = 'ibm'
n_physicians = 15 # SHOULD BE 15 


methods = ['aer', 'z3']   #, 'ibm', 'gurobi', 'z3'] #maybe not z3
all_data = {}

timestamps = {'aer':1746443312,'ibm':00000,'gurobi':0000,'z3':1746621883}
times_plot, Hcs_plot, manys_plot, fews_plot, titles_plot, sat_avgs_plot, sat_vars_plot, extents_plot, unavails_plot, physicians_compare, schedules_plot  = [], [],[], [],[], [],[], [],[], [],[]
for method in methods:
    if method =='z3':
        n_physicians = 5
    all_data[method] = combineDataJune(backend, n_physicians, timestamps[method])

def printDataJune(method:str):
    print(method,'demands:',all_data[method]['run']['demands']) # print parameters so its same
    print(method,'lambdas:',all_data[method]['run']['lambdas']) 
    print(method,'pref seed:',all_data[method]['run']['pref seed']) 
    physicians_compare.append(all_data[method]['physician'])

def plotDataJune(method:str):
    
    # TODO PLOT RUNS DATA         # 'run': {'full time':end_time-start_time, 'Hc full':qaoa_Hc_cost, 'bitstring':qaoa_bitstring, 'demands':demands, 'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas, 'constraints':constraint_scores}
    times_plot.append(all_data[method]['run']['full time'])
    Hcs_plot.append(all_data[method]['run']['Hc full'])
    

    # PLOT CONSTRAINT SCORES   # 'constraints': {'demand': {'correct rate': 1.0, 'too many': 0, 'too few': 0}, 'titles': {'ST error': 0, 'UL error': 0, 'ÖL error': 0}, 'preference': {'satisfaction': [0.0, 0.0, 0.0, 0.0], 'prefer satisfied': [nan, nan, nan, nan], 'prefer not satisfied': [nan, nan, nan, nan]}, 'extent': {'error': [[-53.33333333333334, -6.666666666666677, -6.666666666666677, -6.666666666666677]]}, 'unavail': {'unavail': 0.0}}}
    constraints = all_data[method]['run']['constraints']
    manys_plot.append(constraints['demand']['too many'])
    fews_plot.append(constraints['demand']['too few'])
    titles_plot.append(np.mean(constraints['titles']['ST error'],constraints['titles']['UL error'], constraints['titles']['ÖL error'] ))
    sat_avgs_plot.append(np.mean(constraints['preference']['satisfaction']))
    sat_vars_plot.append(np.std(constraints['preference']['satisfaction'])**2)
    extents_plot.append(np.mean(abs(constraints['extent']['error'])))
    unavails_plot.append(constraints['unavail']['unavail'])

    # PLOT SCHEDULE
        





        
