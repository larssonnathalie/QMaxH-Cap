from postprocessing.postprocessing import *
from preprocessing.preprocessing import convertPreferences
import matplotlib.pyplot as plt
from qaoa.qaoa import QToHc, costOfBitstring

# TODO:
    # schedules
        # aer
        # ibm
        # gurobi
        # z3
    # samma phys

def combineDataJune(backend, n_physicians, timestamp):
    print(f'\nSorting data for JUNE run on {backend}, {n_physicians} physicians at time {timestamp}')

    phys_str = f'{n_physicians}phys_'if timestamp >= 1746621883 else ''
    #physician_df = pd.read_csv(f'data/results/physician/{backend}_{phys_str}time{int(timestamp)}.csv')
    physician_path = f'data/results/physician/{backend}_{phys_str}time{int(timestamp)}.csv'

    schedule_df = pd.read_csv(f'data/results/schedules/{backend}_{n_physicians}phys_time{int(timestamp)}.csv')

    json_file_path = f'data/results/runs/{backend}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(json_file_path, "r") as f:
        run_data = json.load(f)
    if 'full time' not in run_data:
        run_data['full time'] = run_data['total time']
    if 'constraints' not in run_data:
        run_data['constraints'] = run_data['constraint scores']

    all_data = {'run':run_data, 'physician path':physician_path, 'schedule':schedule_df}
    return all_data


def printDataJune(method:str):
    print(method,'demands:',all_data[method]['run']['demands']) # print parameters so its same
    print(method,'lambdas:',all_data[method]['run']['lambdas']) 
    print(method,'pref seed:',all_data[method]['run']['pref seed']) 
    physicians_compare.append(all_data[method]['physician path'])

def plotDataJune(method:str, schedule=True):
    colors = {'ibm':'skyblue', 'gurobi':'tab:orange', 'aer':'green', 'random':'gray'}

    # TODO PLOT RUNS DATA         # 'run': {'full time':end_time-start_time, 'Hc full':qaoa_Hc_cost, 'bitstring':qaoa_bitstring, 'demands':demands, 'layers':n_layers,'search iterations (if aer)':search_iterations, 'pref seed':preference_seed,'n candidates':n_candidates,'lambdas':lambdas, 'constraints':constraint_scores}
    times_plot.append(all_data[method]['run']['full time'])
    Hcs_plot.append(all_data[method]['run']['Hc full'])

    # CONTROL universal Hc
    if method != 'z3': # bc. using a dummy with n_phys = 5
        qubo_uni = pd.read_csv(f'data/intermediate/Qubo_full_june.csv',header=None).to_numpy()
        b = - sum(qubo_uni[i,:] + qubo_uni[:,i] for i in range(qubo_uni.shape[0]))
        Hc_uni = QToHc(qubo_uni,b)
        cost_uni = np.real(costOfBitstring(all_data[method]['run']['bitstring'], Hc_uni))
        if cost_uni == all_data[method]['run']['Hc full']:
            print(cost_uni, '=', all_data[method]['run']['Hc full'])
        else:
            print(f'\nERROR: in {method}', cost_uni, '≠', all_data[method]['run']['Hc full'])

    # PLOT CONSTRAINT SCORES   # 'constraints': {'demand': {'correct rate': 1.0, 'too many': 0, 'too few': 0}, 'titles': {'ST error': 0, 'UL error': 0, 'ÖL error': 0}, 'preference': {'satisfaction': [0.0, 0.0, 0.0, 0.0], 'prefer satisfied': [nan, nan, nan, nan], 'prefer not satisfied': [nan, nan, nan, nan]}, 'extent': {'error': [[-53.33333333333334, -6.666666666666677, -6.666666666666677, -6.666666666666677]]}, 'unavail': {'unavail': 0.0}}}
    constraints = all_data[method]['run']['constraints']
    manys_plot.append(constraints['demand']['too many'])
    fews_plot.append(constraints['demand']['too few'])
    titles_plot.append(np.mean([constraints['titles']['ST error'],constraints['titles']['UL error'], constraints['titles']['ÖL error']]))
    sat_avgs_plot.append(np.mean(constraints['preference']['satisfaction']))
    sat_vars_plot.append(np.std(constraints['preference']['satisfaction'])**2)
    extents_plot.append(np.mean(np.abs(np.array(constraints['extent']['error']))))
    unavails_plot.append(constraints['unavail']['unavail'])

    if schedule:
        # PLOT SCHEDULE
        evaluator_m = Evaluator(all_data[method]['schedule'], cl, time_period, lambdas, physician_path='data/intermediate/physician_universal_june.csv') # TODO: should be same prefs & exts as: f'data/results/physician/{method}_15phys_time{timestamps[method]}.csv'
        evaluator_m.makeResultMatrix()
        evaluator_m.evaluateConstraints(1)
        fig = evaluator_m.cleanPlot(width=10,title=f'June schedule using {str(method).capitalize()}', tile_col = colors[method])
        fig.savefig(f'data/results/final_plots/june/schedules/{method}_final_schedule.png')


def plotStats(plot_data, methods, title='', ylabel=''):
    colors = ['skyblue', 'tab:orange', 'green']
    plt.figure()
    bars = plt.bar(methods, plot_data, width=0.3, color=colors[:len(methods)], alpha = 0.9) # TODO fixa placering, första har mitten på 0, har width bredd
    #plt.xlim((-0.1,0.2)) # probably change
    for bar in bars:
        height = bar.get_height()
        bar_width = bar.get_width()
        plt.text(bar.get_x() + bar_width/2, min(max(height-0.5,0.1*height), 0.9*height), str(round(height,1)),  ha='center', va='bottom')
    plt.xticks(np.arange(len(methods)))
    if title=='Hc costs': # Draw x = 0
        plt.plot(np.linspace(-bar_width,len(methods)-1+bar_width,len(methods)),[0]*(len(methods)), color = 'black', linewidth=0.2)
        plt.xlim((-bar_width, len(methods)-1+bar_width))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'data/results/final_plots/june/{title}.png')
    plt.show()

n_physicians = 15 # SHOULD BE 15 
methods = ['aer', 'gurobi', 'ibm'] #'aer', 'ibm', 'gurobi']   #, 'ibm', 'gurobi', 'z3'] #maybe not z3
all_data = {}
time_period = 'all'
cl = 3
lambdas = {'demand':3, 'fair':10, 'pref':5, 'unavail':15, 'extent':8, 'rest':0, 'titles':5, 'memory':3} 
physician_universal_june = pd.read_csv('data/intermediate/physician_universal_june.csv', index_col=None)

#timestamps = {'aer':1746722550,'ibm':1746706255,'gurobi':1747062967,'z3':1747063119}
timestamps = {'aer':1746722550,'ibm':1746706255,'gurobi':1746970615}#,'z3':1747063119}
times_plot, Hcs_plot, manys_plot, fews_plot, titles_plot, sat_avgs_plot, sat_vars_plot, extents_plot, unavails_plot, physicians_compare  = [], [],[], [],[], [],[], [],[], []
for method in methods:
    #if method =='z3':
     #   n_physicians = 5
    all_data[method] = combineDataJune(method, n_physicians, timestamps[method])

for method in methods:
    printDataJune(method)
    plotDataJune(method, schedule=True) # Plot schedule and append stats to lists




plotStats(times_plot, methods, title='Full computation time', ylabel='Time [s]')
plotStats(Hcs_plot, methods, title='Hc costs')
plotStats(manys_plot, methods, title='Too many workers')
plotStats(fews_plot, methods, title='Too few workers')
plotStats(titles_plot, methods, title='Wrong number of assigned titles')
plotStats(sat_avgs_plot, methods, title='Satisfaction', ylabel='Avg. satisfaction score')
plotStats(sat_vars_plot, methods, title='Satisfaction fairness', ylabel='Variance in satisfaction scores')
plotStats(extents_plot, methods, title='Employment extent error', ylabel='Avg. distance from target n.o. shifts [%]')
plotStats(unavails_plot, methods, title='Shifts assigned to unavailable physicians')

if 'aer' in methods and 'ibm' in methods:
    # Plot 2-gates and circuit depth
    quantum_methods = ['aer', 'ibm']
    aer_n_doubles, aer_depth = all_data['aer']['run']['double gates'], all_data['aer']['run']['depth']
    ibm_n_doubles, ibm_depth = all_data['ibm']['run']['double gates'], all_data['ibm']['run']['depth']
    #plotStats([aer_n_doubles, ibm_n_doubles],quantum_methods, title='Number of 2-qubit gates')
    #plotStats([aer_depth, ibm_depth], quantum_methods, title='Depth of transpiled circuit')'''

cl, time_period, lambdas = 3, 'all', {'demand':3, 'fair':10, 'pref':5, 'unavail':15, 'extent':8, 'rest':0, 'titles':5, 'memory':3} 



"""def plotShortSchedule(method, schedule):

    # PLOT SCHEDULE
    evaluator_m = Evaluator(schedule, cl, time_period, lambdas, physician_path='data/intermediate/physician_universal_june.csv') # TODO: should be same prefs & exts as: f'data/results/physician/{method}_15phys_time{timestamps[method]}.csv'
    evaluator_m.makeResultMatrix()
    evaluator_m.evaluateConstraints(1)
    fig = evaluator_m.cleanPlot(width=10,title=f'June schedule using {method}', tile_col = None)
    fig.savefig(f'data/results/final_plots/june/schedules/{method}_short_schedule.png')

z3_schedule = pd.read_csv('data/results/schedules/z3_15phys_time1747063119.csv',index_col=None)
plotShortSchedule('z3', z3_schedule)

g_schedule = pd.read_csv('data/results/schedules/gurobi_15phys_time1747062967.csv',index_col=None)
plotShortSchedule('gurobi', g_schedule)"""