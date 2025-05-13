import matplotlib.pyplot as plt
import pandas as pd
g_scaling = pd.read_csv('data/results/Classical scaling/gurobi_scaling_results_updated.csv', index_col=None)
z_scaling = pd.read_csv('data/results/Classical scaling/z3_scaling_results (1).csv', index_col=None)

colors = {'gurobi':'#FF8E2E', 'z3':'#FFDD33'}
alp=0.8
siz = 15
'''
# TIMES & MEMORY
plt.figure()
x_ticks = g_scaling['physicians']
x_labels = x_ticks*14 # NOTE assuming 14 days
plt.plot(g_scaling['physicians'], g_scaling['peak_memory_MB'], linewidth=3, label='Gurobi', color = colors['gurobi'], alpha=alp)
plt.plot(z_scaling['physicians'], z_scaling['peak_memory_MB'], linewidth=3, label='Z3', color = colors['z3'], alpha=alp)

# dots
plt.scatter(g_scaling['physicians'], g_scaling['peak_memory_MB'], s=siz, color = colors['gurobi'])
plt.scatter(z_scaling['physicians'], z_scaling['peak_memory_MB'], s=siz, color = colors['z3'])
plt.xlabel('Number of Variables')
plt.ylabel('Peak Memory Usage [MB]')
plt.xticks(ticks=x_ticks, labels = x_labels)
plt.title('Peak memory usage vs. Problem Size')
plt.legend()
plt.savefig('data/results/final_plots/classical/classical_scaling_memory_vars.png')
plt.show()'''

from postprocessing.postprocessing import Evaluator
cl = 3
time_period = 'all'
lambdas = {'demand':3, 'fair':10, 'pref':5, 'unavail':15, 'extent':8, 'rest':0, 'titles':5, 'memory':3} 

g_schedule = pd.read_csv('data/results/Classical scaling/gurobi_10phys_time1747077477.csv', index_col=None)
z_schedule = pd.read_csv('data/results/Classical scaling/z3_10phys_time1747077372.csv', index_col=None)


# SCHEDULES
evaluator_m = Evaluator(g_schedule, cl, time_period, lambdas, physician_path='data/intermediate/10physician_data.csv') # TODO: should be same prefs & exts as: f'data/results/physician/{method}_15phys_time{timestamps[method]}.csv'
evaluator_m.makeResultMatrix()
evaluator_m.evaluateConstraints(1)
fig = evaluator_m.cleanPlot(width=10)
fig.savefig(f'data/results/final_plots/classical/gurobi_10phys_schedule.png')

# SCHEDULES
evaluator_m = Evaluator(z_schedule, cl, time_period, lambdas, physician_path='data/intermediate/10physician_data.csv') # TODO: should be same prefs & exts as: f'data/results/physician/{method}_15phys_time{timestamps[method]}.csv'
evaluator_m.makeResultMatrix()
evaluator_m.evaluateConstraints(1)
fig = evaluator_m.cleanPlot(width=10)
fig.savefig(f'data/results/final_plots/classical/z3_10phys_schedule.png')