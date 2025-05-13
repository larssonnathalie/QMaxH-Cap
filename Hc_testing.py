from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.qaoa import *
import os
import json


all_incr_runs = os.listdir(f'data/results/increasing_qubits/runs')
"""
#n_phys in [3,4,5,6,7,10,14]:
Qubo_n = pd.read_csv(f'data/intermediate/Qubo_incr_{n_phys}phys.csv',header=None).to_numpy() # UNIVERSAL Q
b = - sum(Qubo_june[i,:] + Qubo_june[:,i] for i in range(Qubo_june.shape[0]))
Hc = QToHc(Qubo_june, b)"""

for run in all_incr_runs:
    if run == '.DS_Store':
        continue
    if run[:6] == 'gurobi':
        print(run)
    else:
        continue
    n_phys = int(run[7:9].rstrip('p'))
    print(n_phys)
    path = 'data/results/increasing_qubits/runs/'+str(run)
    with open(path, "r") as f:
        run_data = json.load(f)
        f.close()
    
    Qubo_n = pd.read_csv(f'data/intermediate/Qubo_incr_{n_phys}phys.csv',header=None).to_numpy() # UNIVERSAL Q
    b = b = - sum(Qubo_n[i,:] + Qubo_n[:,i] for i in range(Qubo_n.shape[0]))
    Hc_n = QToHc(Qubo_n, b)
    Hc_full = np.real(costOfBitstring(run_data['bitstring'], Hc_n))
    run_data['Hc full'] = Hc_full
    print(run[:6], Hc_full)

    with open(path, "w") as f:      
        json.dump(run_data,f)
        f.close()