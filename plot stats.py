from postprocessing.postprocessing import *


def combineDataJune(backend, n_physicians, timestamp):
    print(f'\Sorting data for JUNE run on {backend}, {n_physicians} physicians at time {timestamp}')

    physician_df = pd.read_csv(f'data/results/physician/{backend}_time{int(timestamp)}.csv')
    schedule_df = pd.read_csv(f'data/results/schedules/{backend}_{n_physicians}phys_time{int(timestamp)}.csv')

    json_file_path = f'data/results/runs/{backend}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(json_file_path, "r") as f:
        run_data = json.load(f)
    
    all_data = {'run':run_data, 'physician':physician_df, 'schedule':schedule_df}
    return all_data


def combineDataIncr(backend, n_physicians, timestamp):
    print(f'\Sorting data for INCREASING run on {backend}, {n_physicians} physicians at time {timestamp}')

    physician_df = pd.read_csv(f'data/results/increasing_qubits/physician/{backend}_time{int(timestamp)}.csv')
    schedule_df = pd.read_csv(f'data/results/increasing_qubits/schedules/{backend}_{n_physicians}phys_time{int(timestamp)}.csv')
    
    distribution_file_path = f'data/results/increasing_qubits/distributions/{backend}_{n_physicians}phys_{n_physicians*7}vars_time-{int(timestamp)}.json'
    with open(distribution_file_path, "r") as f:
        distribution_data = json.load(f)
    runs_file_path = f'data/results/increasing_qubits/runs/{backend}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(runs_file_path, "r") as f:
        run_data = json.load(f)
    
    all_data = {'run':run_data, 'physician':physician_df, 'schedule':schedule_df, 'distribution':distribution_data}
    return all_data



#Compare june: (show: time, schedules, constraints, Hc:s        + Quantum: depth, 2-gates)
    # june aer 15phys
    # june ibm 15phys
    # june gurobi 15phys
    # (june z3 15phys)

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
n_physicians = 4 # SHOULD BE 15 
start_time = ''
runtype='june' # incr or june
if runtype == 'june':
    pass#n_physicians = 15

backends = ['aer']   #, 'ibm', 'gurobi', 'z3'] #maybe not z3
all_data = {}

if runtype == 'june':
    timestamps = {'aer':'1746442127','ibm':'xxx','gurobi':'xxxx','z3':'xxxx'}
    for backend in backends:
        all_data[backend] = combineDataJune(backend, n_physicians, timestamps[backend])
    
    # TODO PLOT RUNS DATA
    run_data_aer = all_data['aer']['run']  # {'full time':end_time-start_time, 'Hc full':qaoa_Hc_cost, 'bitstring':qaoa_bitstring, 'demands':demands, 'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas, 'constraints':constraint_scores}
    time_aer = run_data_aer['full time'] 
    Hc_aer = run_data_aer['Hc full']
    demands_aer = run_data_aer['demands']
    print(demands_aer) #TODO print all demands and lambdas so its same
    print('doubles', run_data_aer['double gates'])
    print('depth', run_data_aer['depth'])

    # TODO PLOT CONSTRAINT SCORES
    constraints_aer = run_data_aer['constraints'] # 'constraints': {'demand': {'correct rate': 1.0, 'too many': 0, 'too few': 0}, 'titles': {'ST error': 0, 'UL error': 0, 'ÖL error': 0}, 'preference': {'satisfaction': [0.0, 0.0, 0.0, 0.0], 'prefer satisfied': [nan, nan, nan, nan], 'prefer not satisfied': [nan, nan, nan, nan]}, 'extent': {'error': [[-53.33333333333334, -6.666666666666677, -6.666666666666677, -6.666666666666677]]}, 'unavail': {'unavail': 0.0}}}

    demand_aer = constraints_aer['demand']
    correct_rate_aer = demand_aer['correct rate']
    too_many_aer = demand_aer['too many']
    too_few_aer = demand_aer['too few']

    titles_aer = constraints_aer['titles'] #{'ST error': 0, 'UL error': 0, 'ÖL error': 0},

    preference_aer = constraints_aer['preference'] #  'preference': {'satisfaction': [0.0, 0.0, 0.0, 0.0], 'prefer satisfied': [nan, nan, nan, nan], 'prefer not satisfied': [nan, nan, nan, nan]},
    satisfaction_scores = preference_aer['satisfaction']
    prefer_rates = preference_aer['prefer satisfied']
    prefer_not_rates = preference_aer['prefer not satisfied']

    extents = constraints_aer['extent'] # 'extent': {'error': [[-53.33333333333334, -6.666666666666677, -6.666666666666677, -6.666666666666677]]}
    unavail = constraints_aer['unavail'] #  'unavail': {'unavail': 0.0}}}'''

    # TODO: PLOT SCHEDULE
    bitstring_aer = run_data_aer['bitstring']

elif runtype == 'incr':
    timestamps = {'aer':'xxxx','ibm':'xxx','gurobi':'xxxx','z3':'xxxx'}
    for backend in backends:
        all_data[backend] = combineDataIncr(backend, n_physicians, timestamps['backend'])
