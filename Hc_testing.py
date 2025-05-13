from preprocessing.preprocessing import *
from postprocessing.postprocessing import *
from qaoa.qaoa import *
import os
import json

Qubo_june = pd.read_csv('data/intermediate/Qubo_full_june.csv',header=None).to_numpy() # UNIVERSAL Q
b = - sum(Qubo_june[i,:] + Qubo_june[:,i] for i in range(Qubo_june.shape[0]))
Hc = QToHc(Qubo_june, b)
all_june_runs = os.listdir(f'data/results/runs')

for run in all_june_runs:
    if run == '.DS_Store':
        continue
    path = 'data/results/runs/'+str(run)
    with open(path, "r") as f:
        run_data = json.load(f)
        f.close()
    
    Hc_full = np.real(costOfBitstring(run_data['bitstring'], Hc))
    run_data['Hc full'] = Hc_full
    print(run[:5], Hc_full)
    with open(path, "w") as f:      
        json.dump(run_data,f)
        f.close()