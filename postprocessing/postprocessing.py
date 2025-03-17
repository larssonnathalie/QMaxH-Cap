import pandas as pd
from plotnine import * # pip install plotnine
import numpy as np
import matplotlib.pyplot as plt

def bitstringIndexToPS(idx, n_vars, n_shifts):

    idx=int((n_vars-1)-idx) # NOTE assuming strings are reversed 
    p = int(idx/n_shifts)
    s = idx%n_shifts
    return p,s

def psVarNamesToI(xp_s, n_shifts): # Might not be needed
        p,s = xp_s.lstrip('x').split('_')
        i = int(p) * n_shifts + int(s)
        return f'x{i}'

def bitstringToSchedule(bitstring:str, empty_calendar_df, cl, n_shifts, prints=True) -> pd.DataFrame:
    staff_col = [[] for _ in range(empty_calendar_df.shape[0])]

    n_vars = len(bitstring)
    for i in range(n_vars): 
        bit = bitstring[i]
        if bit == '1':
            p,s = bitstringIndexToPS(i, n_vars=n_vars, n_shifts=n_shifts)
            staff_col[s].append(str(p))

    result_schedule_df = empty_calendar_df.copy()
    result_schedule_df['staff'] = staff_col
    #result_schedule_df.to_csv(f'data/results/result_schedule.csv', index=False)
    return result_schedule_df


def controlSchedule(result_schedule_df, demand_df, cl, prints=True):
    combined_df = demand_df.merge(result_schedule_df, on='date', how='outer')
    ok_col = []
    for i in range(combined_df.shape[0]):
        if combined_df.loc[i,'demand'] == len(combined_df.loc[i,'staff']):
            ok_col.append('ok')
        else:
           ok_col.append(' NOT ok!')
    combined_df['Shift covered']=ok_col
    if prints:
        print(combined_df)
    combined_df.to_csv(f'data/results/result_and_demand_cl{cl}.csv', index=False)
    return combined_df
    
def controlPlot(result_df): #TODO compare with ok column
    physician_df =pd.read_csv('data/intermediate/physician_data.csv',index_col=False) #TODO (change to /input/, compare specific dates?)
    n_physicians = len(physician_df)
    n_shifts = len(result_df)

    result_matrix = np.zeros((n_physicians,n_shifts))
    for s in range(n_shifts):
        workers_s = result_df['staff'].iloc[s]
        for p in workers_s:
            result_matrix[int(p)][s] = 1 

    prefer_matrix = np.zeros((n_physicians,n_shifts))
    for p in range(n_physicians):
        prefer_p = physician_df['Prefer'].iloc[p]
        if type(prefer_p) == str:
            prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
            for s in prefer_shifts_p:
                prefer_matrix[p][int(s)] = 1

        prefer_not_p = physician_df['Prefer Not'].iloc[p]
        if type(prefer_not_p) == str:
            prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
            for s in prefer_not_shifts_p:
                prefer_matrix[p][int(s)] = -1
    
    #TODO add rest of constraints
   

    fig, ax = plt.subplots()

    prefer_colors = np.where(prefer_matrix.flatten()==1,'green',prefer_matrix.flatten())
    prefer_colors = np.where(prefer_matrix.flatten()==-1,'red',prefer_colors)
    prefer_colors = np.where(prefer_matrix.flatten()==0,'white',prefer_colors)


    im = ax.imshow(result_matrix, cmap="Blues", interpolation="nearest")

    pc = ax.pcolor(np.arange(n_shifts), np.arange(n_physicians), np.zeros((n_physicians,n_shifts)), 
              cmap="grey", linewidths=7, edgecolors=prefer_colors, alpha=0.5) #
    
    #pc = ax.pcolor([0,1,2,3], [1,2,3], np.zeros((3,4)), 
               #linewidths=7, edgecolors='red', alpha=0.5)

    # Remove axis ticks for clarity
    ax.set_xticks(ticks=np.arange(n_shifts), labels=[date for date in result_df['date']])
    ax.set_yticks(ticks=np.arange(n_physicians), labels=[phys for phys in physician_df['Physician']])

    plt.colorbar(im, ax=ax, label="Fill Color Scale")
    plt.figure()
    plt.title('Result')
    plt.imshow(result_matrix, cmap='Blues')
    plt.figure()
    plt.title('Preferences')
    plt.imshow(-prefer_matrix,cmap='coolwarm')
    plt.show()
