from postprocessing.postprocessing import *
import matplotlib.pyplot as plt
    
# TODO
    # Plotta distributions
    # Plotta avg Hc
    # Plotta tid
    #  (Samma physician_df, gör convertPrefs, mata in i cleanPlot)
    #


def combineDataIncr(backend, n_physicians, timestamp):
    print(f'\Sorting data for INCREASING run on {backend}, {n_physicians} physicians at time {timestamp}')

    phys_str = f'{n_physicians}phys_'if timestamp >= 1746621883 else ''
    physician_df = pd.read_csv(f'data/results/increasing_qubits/physician/{backend}_{phys_str}time{int(timestamp)}.csv')
    schedule_df = pd.read_csv(f'data/results/increasing_qubits/schedules/{backend}_{n_physicians}phys_time{int(timestamp)}.csv')
    
    distribution_file_path = f'data/results/increasing_qubits/distributions/{backend}_{n_physicians}phys_{n_physicians*7}vars_time-{int(timestamp)}.json'
    with open(distribution_file_path, "r") as f:
        distribution_data = json.load(f)
    runs_file_path = f'data/results/increasing_qubits/runs/{backend}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(runs_file_path, "r") as f:
        run_data = json.load(f)
    
    all_data = {'run':run_data, 'physician':physician_df, 'schedule':schedule_df, 'distribution':distribution_data}
    return all_data

# Compare incr:  (show: time, avg Hc,                               + Quantum: depth, 2-gates)
    # aer 3 phys
    # ibm 3 phys
    # random 3 phys 
    # (gurobi 3 phys)

    # ibm 4 phys
    # random 4 phys
    # (gurobi 4 phys)

    # ibm 5 phys
    # random 5 phys
    # (gurobi 5 phys)

    # ibm 7 phys
    # random 7 phys
    # (gurobi 7 phys)
    
    # ibm 10 phys
    # random 10 phys
    # (gurobi 10 phys)

    # ibm 14 phys
    # random 14 phys
    # (gurobi 14 phys)

    # ibm 17 phys
    # random 17 phys
    # (gurobi 17 phys)

    # ibm 21 phys
    # random 21 phys
    # (gurobi 21 phys)


backend = 'ibm'
n_physicians = 15 
methods = ['ibm', 'gurobi']   #, 'ibm', 'gurobi', 'z3'] #maybe not z3
all_data = {}


    
#TODO INCREASE PHYS
timestamps = {'ibm':0,'gurobi':0, 'random':0}
for method in methods:
    all_data[method] = combineDataIncr(backend, n_physicians, timestamps[method])

def printDataIncr(method:str):
    pass
def plotDataIncr(method:str):
    pass
                    
    # 'schedule'
    # 'physician'
    
    
    # TODO PLOT distribution          # 'distribution':



    # TODO PLOT RUN DATA  # {'full time':end_time-start_time, 'best params':qaoa.params_best[0].tolist(), 'best params cost':qaoa.params_best[1].tolist(), 'demands':demands, 'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas, 'avg Hc':avg_Hc, 'avg Hc random':avg_Hc_random}


    #'titles':{'ST error': 0, 'UL error': 0, 'ÖL error': 0},

    #  'preference': {'satisfaction': [0.0, 0.0, 0.0, 0.0], 'prefer satisfied': [nan, nan, nan, nan], 'prefer not satisfied': [nan, nan, nan, nan]},
    # 'extent': {'error': [[-53.33333333333334, -6.666666666666677, -6.666666666666677, -6.666666666666677]]}
    #  'unavail': {'unavail': 0.0}}}'''

    # TODO: PLOT SCHEDULE
    