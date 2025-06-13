from postprocessing.postprocessing import *
import matplotlib.pyplot as plt
from qaoa.qaoa import QToHc, costOfBitstring
import os

# TODO
    # plot 2-qubit gates, depth
    # (Kör om ibm 17 phys)
    # (Samma physician_df, gör convertPrefs,
    # (add missing full times from ibm website)?


def combineDataIncr(method, n_physicians, timestamp):
    print(f'Sorting data for: {method}, {n_physicians} physicians at time {timestamp}')
    all_data = {}

    #phys_str = f'{n_physicians}phys_'if timestamp >= 1746621883 else ''
    #physician_df = pd.read_csv(f'data/results/increasing_qubits/physician/{method}_{phys_str}time{int(timestamp)}.csv')
    if method not in ['gurobi', 'z3']:
        distribution_file_path = f'data/results/increasing_qubits/distributions/{method}_{n_physicians}phys_{n_physicians*7}vars_time-{int(timestamp)}.json'
        with open(distribution_file_path, "r") as f:
            distribution_data = json.load(f)
            f.close()
        all_data['distribution'] = distribution_data 

    runs_file_path = f'data/results/increasing_qubits/runs/{method}_{n_physicians}phys_time{int(timestamp)}.json'
    with open(runs_file_path, "r") as f:
        run_data = json.load(f)
        f.close()
    all_data['run'] = run_data
   
    return all_data

def getTimestamps():
    timestamps = {}
    all_distributions = os.listdir(f'data/results/increasing_qubits/distributions')
    all_distributions.sort()
    for file in all_distributions:
        method, phys, vars, time = file.split('_')
        phys = int(phys.rstrip('phys'))
        timestamp = int(time.lstrip('time-').rstrip('.json'))
        if (method, phys) not in timestamps:
            timestamps[(method, phys)] = timestamp 
        else:
            if timestamp > timestamps[(method, phys)]: # Replacing with the newest one
                timestamps[(method, phys)] = timestamp 


    all_runs = os.listdir(f'data/results/increasing_qubits/runs') # Look in 2 folders bc. Gurobi has no distribution and random has no runs
    for file in all_runs:
        method, phys, time = file.split('_')
        phys = int(phys.rstrip('phys'))
        timestamp = int(time.lstrip('time-').rstrip('.json'))
        if (method, phys) not in timestamps:
            timestamps[(method, phys)] = timestamp 
        else:
            if timestamp > timestamps[(method, phys)]: # Replacing with the newest one
                timestamps[(method, phys)] = timestamp 
    all_runs.sort()

    return timestamps


def getPlotlistsIncr(runs, all_data, extra_gurobi):
    plot_times, plot_Hcs, plot_distr = {'ibm':[], 'gurobi':[], 'aer':[]}, {'ibm':[], 'gurobi':[], 'random':[], 'aer':[], 'z3':[]}, {'ibm':[], 'random':[], 'aer':[]}
    for n_phys in [3,4,5,6,7,10,14]: # + maybe 17
        qubo_n = pd.read_csv(f'data/intermediate/Qubo_incr_{n_phys}phys.csv',header=None).to_numpy()
        b = - sum(qubo_n[i,:] + qubo_n[:,i] for i in range(qubo_n.shape[0]))
        Hc_n = QToHc(qubo_n,b)

        for run in runs:
            if run[1] == n_phys:
                method=run[0]
                run_data  =  all_data[(method, n_phys)]['run']
                if method != 'random' and 'full time' in run_data: # some times are missing
                    plot_times[method].append(run_data['full time'])
                elif 'total time' in run_data:
                    plot_times[method].append(run_data['total time'])

                if method not in ['gurobi', 'z3']:
                    plot_distr[method].append(all_data[(method, n_phys)]['distribution'])
                
                if 'Hc full' in run_data:
                    plot_Hcs[method].append(run_data['Hc full'] )
                elif 'avg Hc' in run_data:
                    plot_Hcs[method].append(run_data['avg Hc'] )


                else:
                    # COMPUTE AVG Hc
                    print(f'Get avg for {n_phys}phys, {method}')
                    all_costs = []
                    for i, bitstring_i in enumerate(list(all_data[(method, n_phys)]['distribution'].keys())):
                        if i%1000 == 0:
                            print(i)
                        count_i = all_data[(method, n_phys)]['distribution'][bitstring_i]
                        cost_i = np.real(costOfBitstring(bitstring_i, Hc_n))
                        all_costs += [cost_i]*count_i
                    avg_Hc = np.mean(all_costs)
                    print('avg', avg_Hc)
                    plot_Hcs[method].append(avg_Hc)

                    # Add to run file
                    run_data['Hc full'] = avg_Hc
                    with open(f'data/results/increasing_qubits/runs/{method}_{n_phys}phys_time{timestamps[(method, n_phys)]}.json', "w") as f:
                        json.dump(run_data, f)
                        f.close()

                # COMPARE UNIVERSAL HC
                if method == 'gurobi' or method == 'z3':
                    if 'universal Hc' not in run_data:
                        universal_Hc = np.real(costOfBitstring(run_data['bitstring'], Hc_n))
                        run_data['universal Hc'] = universal_Hc
                        with open(f'data/results/increasing_qubits/runs/{method}_{n_phys}phys_time{timestamps[(method, n_phys)]}.json', "w") as f:
                            json.dump(run_data, f)
                            f.close()
    if extra_gurobi:
        for n_phys in [17,21,28,35,42,49,70,119]:
            plot_times['gurobi'].append((all_data[('gurobi', n_phys)]['run']['total time']))
            plot_Hcs['gurobi'].append((all_data[('gurobi', n_phys)]['run']['Hc full']))

    return plot_times, plot_distr, plot_Hcs

def plotsIncr(plot_times, plot_Hcs, extra_gurobi=False):
    #colors = {'ibm':'#54A4CC', 'gurobi':'tab:orange', 'aer':'green', 'random':'gray'}
    colors = {'gurobi':'#FF8E2E', 'z3':'#FFDD33', 'ibm':'#0099DD', 'aer':'skyblue', 'random':'gray'}

    phys_values = list(plot_Hcs.keys())
    phys_values.sort()
    if extra_gurobi:
        xticks=[3,4,5,6,7,10,14,17,21,28,35,42,49,70,119]
    else:
        xticks=[3,4,5,6,7,10,14]
    xlabels = [i*7 for i in xticks]
    alp=0.6
    siz = 10
    lw = 3.3

    # TIMES
    plt.figure()
    plt.title(f'Computation times')
    plt.plot(xticks[:len(plot_times['gurobi'])], plot_times['gurobi'], linewidth=lw, label='Gurobi', color = colors['gurobi'], alpha=alp)
    plt.plot(xticks[:len(plot_times['ibm'])], plot_times['ibm'], label = 'IBM',linewidth=lw, color = colors['ibm'], alpha=alp)
    plt.plot(xticks[:len(plot_times['aer'])], plot_times['aer'], label = 'Aer',linewidth=lw, color = colors['aer'], alpha=alp)
    # dots
    plt.scatter(xticks[:len(plot_times['gurobi'])], plot_times['gurobi'], s=siz, color = colors['gurobi'])
    plt.scatter(xticks[:len(plot_times['ibm'])], plot_times['ibm'], s=siz, color = colors['ibm'])
    plt.scatter(xticks[:len(plot_times['aer'])], plot_times['aer'], s=siz, color = colors['aer'])

    plt.legend()
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlabel('Decision variables')
    plt.ylabel('Time [s]')
    extra_str = '_extra' if extra_gurobi else ''
    plt.savefig(f'data/results/increasing_qubits/final_plots/computation_times{extra_str}.png')
    plt.show()

    # Avg Hc
    plt.figure()
    plt.title(f'Average Hc costs')
    plt.plot(xticks[:len(plot_Hcs['random'])], plot_Hcs['random'], linewidth=lw, label='Random', color = colors['random'], alpha=alp)
    plt.plot(xticks[:len(plot_Hcs['gurobi'])], plot_Hcs['gurobi'], linewidth=lw, label='Gurobi', color = colors['gurobi'], alpha=alp)
    plt.plot(xticks[:len(plot_Hcs['ibm'])], plot_Hcs['ibm'], linewidth=lw, label='IBM', color = colors['ibm'], alpha=alp)
    plt.plot(xticks[:len(plot_Hcs['aer'])], plot_times['aer'], label = 'Aer',linewidth=lw, color = colors['aer'], alpha=alp)
    plt.ylabel('Hc cost')

    # dots
    plt.scatter(xticks[:len(plot_Hcs['random'])], plot_Hcs['random'], s=siz, color = colors['random'])
    plt.scatter(xticks[:len(plot_Hcs['gurobi'])], plot_Hcs['gurobi'], s=siz, color = colors['gurobi'])
    plt.scatter(xticks[:len(plot_Hcs['ibm'])], plot_Hcs['ibm'], s=siz, color = colors['ibm'])
    plt.scatter(xticks[:len(plot_Hcs['aer'])], plot_Hcs['aer'], s=siz, color = colors['aer'])


    plt.legend()
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlabel('Decision variables')

    plt.savefig(f'data/results/increasing_qubits/final_plots/Hc_costs{extra_str}.png')
    plt.show()

def plotDistributions(n_phys):
    methods = [ 'random','ibm', 'gurobi']
    #colors = {'ibm':'#64B4DC', 'gurobi':'tab:orange', 'aer':'green', 'random':'gray'}
    colors = {'gurobi':'#FF8E2E', 'z3':'#FFDD33', 'ibm':'#0099DD', 'aer':'skyblue', 'random':'gray'}

    for j, method in enumerate(methods):
        run_data = all_data[(method, n_phys)]['run']
        
        if 'cost distribution' in run_data:
            all_costs = run_data['cost distribution']
        
        elif method != 'gurobi':
            # COMPUTE COSTS
            qubo_n = pd.read_csv(f'data/intermediate/Qubo_incr_{n_phys}phys.csv',header=None).to_numpy()
            b = - sum(qubo_n[i,:] + qubo_n[:,i] for i in range(qubo_n.shape[0]))
            Hc_n = QToHc(qubo_n,b)
            print('Computing cost for solution distribution of', method, n_phys,'phys')
            all_costs = []
            for i, bitstring_i in enumerate(list(all_data[(method, n_phys)]['distribution'].keys())):
                if i%1000 == 0:
                    print(i)
                count_i = all_data[(method, n_phys)]['distribution'][bitstring_i]
                cost_i = np.real(costOfBitstring(bitstring_i, Hc_n))
                all_costs += [cost_i]*count_i

            # Add to file
            run_data['cost distribution'] = all_costs
            with open(f'data/results/increasing_qubits/runs/{method}_{n_phys}phys_time{timestamps[(method, n_phys)]}.json', "w") as f:
                json.dump(run_data, f)
                f.close()

        if method != 'gurobi':
            plt.hist(all_costs, bins=50, label=method.capitalize(), color=colors[method], alpha=0.3)

    for method in methods:
        run_data = all_data[(method, n_phys)]['run']
        try:
            plt.axvline(x=run_data['Hc full'], color=colors[method], linestyle=':', linewidth=2, label=f'{method.capitalize()} avg.')
        except:
            plt.axvline(x=run_data['avg Hc'], color=colors[method], linestyle=':', linewidth=2, label=f'{method.capitalize()} avg.')

    
    #plt.text(-50, 27000, 'avg',  ha='center', va='bottom', rotation=90, color='gray')
    plt.legend()
    plt.xlabel('Cost (Hc)')
    plt.ylabel('Probability [%]')
    plt.title(f'Hc cost distributions at {n_phys*7} variables')
    plt.yticks(ticks=np.linspace(0,100000,11), labels=['','10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    plt.xlim((-500, 500))
    plt.ylim((0,30000))
    plt.savefig(f'data/results/increasing_qubits/final_plots/cost_distribution_{n_phys*7}vars.png')
    plt.show()


# Compare incr:  (show: time, avg Hc,     + Quantum: depth, 2-gates)
    #     ibm, gurobi, random
    #  3
    #  4
    #  5
    #  6
    #  7
    #  10
    #  14
    # (17)

extra_gurobi = False
timestamps = getTimestamps()
print('ibm5:',timestamps[('ibm', 5)])
runs = list(timestamps.keys())
runs.sort()
methods = set([run[0] for run in runs])
print('timestamps', timestamps)
print('runs', runs)
print('methods', methods)

all_data = {}
for run in runs:
    method, n_phys = run[0], run[1]
    if n_phys==17 and method in ['ibm', 'random']: # error in 17 run for ibm
        continue
    if not extra_gurobi and n_phys >17:
        continue
    all_data[run] = combineDataIncr(method, n_phys, timestamps[(method,n_phys)])
plot_times, plot_distr,  plot_Hcs = getPlotlistsIncr(runs, all_data, extra_gurobi=extra_gurobi)
plotsIncr(plot_times, plot_Hcs, extra_gurobi=extra_gurobi)
plotDistributions(n_phys=4)