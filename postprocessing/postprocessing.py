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

def bitstringToSchedule(bitstring:str, empty_calendar_df, n_shifts) -> pd.DataFrame:
    staff_col = [[] for _ in range(n_shifts)] # TODO week-wise interpretation, bc bitstrings assume 7(*3) shifts

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


def controlSchedule(result_schedule_df, shift_data_df, cl):
    combined_df = shift_data_df.merge(result_schedule_df, on='date', how='outer')
    ok_col = []
    for i in range(combined_df.shape[0]):
        if combined_df.loc[i,'demand'] == len(combined_df.loc[i,'staff']):
            ok_col.append('ok')
        else:
           ok_col.append('NOT ok!')
    combined_df['shift covered']=ok_col
    # print(combined_df)
    combined_df.to_csv(f'data/results/result_and_demand_cl{cl}.csv', index=False)

    return combined_df
    
def controlPlot(result_df, weeks): 
    physician_df =pd.read_csv('data/intermediate/physician_data.csv',index_col=False) #TODO (change to /input/, compare specific dates?)
    n_physicians = len(physician_df)
    n_shifts = len(result_df)

    result_matrix = np.zeros((n_physicians,n_shifts))
    for s in range(n_shifts):
        workers_s = result_df['staff'].iloc[s]
        for p in workers_s:
            result_matrix[int(p)][s] = 1 
    
    prefer_matrix = np.zeros((n_physicians,n_shifts))

    if type(weeks)==int:
        weeks =list(weeks)

    for week in weeks:
        for p in range(n_physicians):
            prefer_p = physician_df[f'prefer w{week}'].iloc[p]
            if prefer_p != '[]':
                prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
                for s in prefer_shifts_p:
                    prefer_matrix[p][int(s)] = 1

            prefer_not_p = physician_df[f'prefer not w{week}'].iloc[p]
            if prefer_not_p != '[]':
                prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  
                for s in prefer_not_shifts_p:
                    prefer_matrix[p][int(s)] = -1

            unavail_p = physician_df[f'unavailable w{week}'].iloc[p]
            if unavail_p != '[]':
                unavail_shifts_p = unavail_p.strip('[').strip(']').split(',')  
                for s in unavail_shifts_p:
                    prefer_matrix[p][int(s)] = -2

    ok_row = np.zeros((1,n_shifts))
    ok_row[0,:] = result_df['shift covered']=='ok'

    #TODO add rest of constraints
   
    #fig, ax = plt.subplots()
    x_size = 5
    y_size = n_physicians/n_shifts * x_size
    plt.figure(figsize=(x_size,y_size))
    prefer_colors = np.where(prefer_matrix.flatten()==1,'lightgreen',prefer_matrix.flatten()) # prefer
    prefer_colors = np.where(prefer_matrix.flatten()==-1,'pink',prefer_colors) # prefer not
    prefer_colors = np.where(prefer_matrix.flatten()==0,'none',prefer_colors) # neutral
    prefer_colors = np.where(prefer_matrix.flatten()==-2,'red',prefer_colors) # unavailable
    
    plt.pcolor(np.arange(n_shifts+1)-0.5, np.arange(n_physicians+1)-0.5,result_matrix, 
              cmap="Greens")
    x, y = np.meshgrid(np.arange(n_shifts), np.arange(n_physicians))  
    plt.scatter(x.ravel(), y.ravel(), s=(50*(x_size/n_shifts))**2, c='none',marker='s', linewidths=9,edgecolors=prefer_colors)
    plt.pcolor(np.arange(n_shifts+1)-0.5,[n_physicians-0.5,n_physicians-0.4],ok_row, cmap='RdYlGn', vmin=0,vmax=1) 

    plt.xticks(ticks=np.arange(n_shifts), labels=[date for date in result_df['date']])
    yticks = [i for i in np.arange(n_physicians)]+[n_physicians-0.4]
    plt.yticks(ticks=yticks, labels=[phys[-1] for phys in physician_df['name']]+['OK n.o.\nworkers'])
    plt.show()
