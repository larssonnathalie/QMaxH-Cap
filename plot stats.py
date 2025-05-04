from postprocessing.postprocessing import *


# Plots:
    # june aer XP
    # june ibm XP
    # june gurobi XP
    # june z3 XP

#XP = physicians to be decided

backend = 'aer'
n_physicians = 10
start_time = ''

plotStats(f'data/results/runs/june_{backend}_full_{n_physicians}phys_time{int(start_time)}.json')
