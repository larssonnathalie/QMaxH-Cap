import pandas as pd
from plotnine import * # pip install plotnine
import numpy as np
import matplotlib.pyplot as plt
from qaoa.converters import *

def bitstringIndexToPS(idx, n_vars, n_shifts):
    idx=int((n_vars-1)-idx) # NOTE assuming strings are reversed 
    p = int(idx/n_shifts)
    s = idx%n_shifts
    return p,s

def psVarNamesToI(xp_s, n_shifts): # Might not be needed
        p,s = xp_s.lstrip('x').split('_')
        i = int(p) * n_shifts + int(s)
        return f'x{i}'

def bitstringToSchedule(bitstring:str, empty_calendar_df) -> pd.DataFrame:
    n_shifts=len(empty_calendar_df)
    staff_col = [[] for _ in range(n_shifts)]
    
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

    if cl >=3:
        combined_df = shift_data_df.merge(result_schedule_df, on=['date', 'shift type'], how='outer')
    ok_col = []

    for i in range(combined_df.shape[0]):
        if combined_df['demand'].iloc[i] == len(combined_df['staff'].iloc[i]):
            ok_col.append('ok')
        else:
           ok_col.append('NOT ok!')
    combined_df['shift covered']=ok_col
    combined_df.to_csv(f'data/results/result_and_demand_cl{cl}.csv', index=False)
    return combined_df


# Remember previous PREFERENCE SATISFACTION and EXTENT to ensure fairness and correct # hours
satisfaction_plot = []
def recordHistory(result_schedule_df_t, t, cl, time_period):
    physician_df = pd.read_csv(f'data/intermediate/physician_data.csv')
    shift_df_t = pd.read_csv(f'data/intermediate/shift_data_t{t}.csv')
    n_physicians = len(physician_df)
    n_shifts = len(result_schedule_df_t)
    
    prefer = {p:physician_df[f'prefer t{t}'].iloc[p].strip('[').strip(']').split(',') for p in range(n_physicians)}
    prefer_not = {p:physician_df[f'prefer not t{t}'].iloc[p].strip('[').strip(']').split(',') for p in range(n_physicians)}
    unavailable = {p:physician_df[f'unavailable t{t}'].iloc[p].strip('[').strip(']').split(',') for p in range(n_physicians)}

    assigned_shifts = {p:[] for p in range(n_physicians)}
    for s in range(n_shifts):
        staff_s = result_schedule_df_t['staff'].iloc[s]
        for p in staff_s:
            assigned_shifts[int(p)].append(s)

    # SATISFACTION scores
    self_weight = 1  # how much p's own preference is weighted against everyone's preferences
    satisfaction_col_t = []
    for p in range(n_physicians):
        if t ==0:
            satisfaction_p = 0
        else:
            satisfaction_p = physician_df['satisfaction'].iloc[p]

        for s in prefer[p]:
            if s!='':
                s=int(s)
                if s in assigned_shifts[p]:
                    satisfaction_p += shift_df_t['attractiveness'].iloc[s] + self_weight
                else:
                    satisfaction_p -= shift_df_t['attractiveness'].iloc[s] + self_weight

        for s in prefer_not[p]:
            if s!='':
                s=int(s)
                if s in assigned_shifts[p]:
                    satisfaction_p -= shift_df_t['attractiveness'].iloc[s] + self_weight
                else:
                    satisfaction_p += shift_df_t['attractiveness'].iloc[s] + self_weight

        for s in unavailable[p]:
            if s!='':
                s=int(s)
                if s in assigned_shifts[p]:
                    print('\n Shift assigned to UNAVAILABLE physician')
                    satisfaction_p -= shift_df_t['attractiveness'].iloc[s] + self_weight*5 # TODO maybe change or make impossible
        
        satisfaction_col_t.append(satisfaction_p)
    #print(satisfaction_col_t, 'satisfaction')
    satisfaction_plot.append(satisfaction_col_t) # NOTE global list
    physician_df['satisfaction'] = satisfaction_col_t

    # EXTENT
    shifts_worked_col, work_rate_col = [],[]
    total_shifts = (t+1)*getShiftsPerT(time_period, cl)
    percentage = {p:physician_df['extent'].iloc[p] for p in range(n_physicians)}
    for p in range(n_physicians):
        shifts_worked_p = physician_df['shifts worked'].iloc[p] + len(assigned_shifts[p])
        shifts_worked_col.append(shifts_worked_p)
        target_percent_of_shifts = percentOfShifts(percentage[p], cl)
        work_rate_col.append((shifts_worked_p/total_shifts)/target_percent_of_shifts) # how many % too much or too little they worked

    physician_df['shifts worked'] = shifts_worked_col
    physician_df['work rate'] = work_rate_col

    physician_df.to_csv('data/intermediate/physician_data.csv', index=None)


def controlPlot(result_df, Ts, cl,time_period, width=10): 
    physician_df =pd.read_csv('data/intermediate/physician_data.csv', index_col=False) #TODO (change to /input/, compare specific dates?)
    n_physicians = len(physician_df)
    n_shifts = len(result_df)

    shifts_per_t = getShiftsPerT(time_period, cl)

    result_matrix = np.zeros((n_physicians,n_shifts))
    for s in range(n_shifts):
        workers_s = result_df['staff'].iloc[s]
        if type(workers_s) == str: # For using results saved in files
            workers_s = workers_s.strip("[] ").split(',')
        for p in workers_s:
            p=p.strip(" '")
            result_matrix[int(p)][s] = 1 
    
    ok_row = np.zeros((1,n_shifts))
    ok_row[0,:] = result_df['shift covered']=='ok'
    
    if cl>=2:
        # PREFERENCE SATISFACTION
        prefer_matrix = np.zeros((n_physicians,n_shifts))

        if type(Ts)==int:
            Ts = list(Ts)

        for t in Ts:
            for p in range(n_physicians):
                prefer_p = physician_df[f'prefer t{t}'].iloc[p]
                if prefer_p != '[]':
                    prefer_shifts_p = prefer_p.strip('[').strip(']').split(',')  #TODO fix csv list handling
                    for s in prefer_shifts_p:
                        s_tot = int(s)+t*shifts_per_t
                        prefer_matrix[p][s_tot] = 1

                prefer_not_p = physician_df[f'prefer not t{t}'].iloc[p]
                if prefer_not_p != '[]':
                    prefer_not_shifts_p = prefer_not_p.strip('[').strip(']').split(',')  
                    for s in prefer_not_shifts_p:
                        s_tot = int(s)+t*shifts_per_t
                        prefer_matrix[p][s_tot] = -1

                unavail_p = physician_df[f'unavailable t{t}'].iloc[p]
                if unavail_p != '[]':
                    unavail_shifts_p = unavail_p.strip('[').strip(']').split(',')  
                    for s in unavail_shifts_p:
                        s_tot = int(s)+t*shifts_per_t
                        prefer_matrix[p][s_tot] = -2
                
        # EXTENT
        extent_error = np.zeros((n_physicians,1))
        target_shifts_percent = {25:0.595, 50:0.119, 75:0.179, 100:0.238}
        n_assigned_shifts = np.sum(result_matrix,axis=1)
        
        for p in range(n_physicians):
            target_shifts = target_shifts_percent[physician_df['extent'].iloc[p]]*n_shifts
            error = ((n_assigned_shifts[p]/target_shifts)-1)*100 # % too much or little work hours
            extent_error[p,0] =error

        #TODO add rest constraint

        prefer_colors = np.where(prefer_matrix.flatten()==1,'lightgreen',prefer_matrix.flatten()) # prefer
        prefer_colors = np.where(prefer_matrix.flatten()==-1,'pink',prefer_colors) # prefer not
        prefer_colors = np.where(prefer_matrix.flatten()==0,'none',prefer_colors) # neutral
        prefer_colors = np.where(prefer_matrix.flatten()==-2,'red',prefer_colors) # unavailable

        # PREFERENCE satisfaction
        only_preference = np.where(prefer_matrix==-2, 0,prefer_matrix) # remove unavailable
        only_prefer = np.where(only_preference==-1, 0,only_preference) # remove prefer not
        only_prefer_not = -np.where(only_preference==1, 0,only_preference) # remove prefer

        prefer_satisfy_rate = np.zeros((n_physicians,1))
        prefer_not_satisfy_rate = np.zeros((n_physicians,1))

        for p in range(n_physicians):
            prefer_met = np.sum(only_prefer[p,:]*result_matrix[p,:]==1)
            if np.sum(only_prefer[p]) !=0:
                prefer_satisfy_rate[p] = prefer_met/np.sum(only_prefer[p]) # % of "prefer" that was satisfied
            else:
                prefer_satisfy_rate[p] = 0#np.nan 

            prefer_not_but_worked = np.sum(only_prefer_not[p,:]*(result_matrix[p,:])==1)
            prefer_not_was_free = np.sum(only_prefer_not[p]) - prefer_not_but_worked
            if np.sum(only_prefer_not[p]) !=0:
                prefer_not_satisfy_rate[p] = prefer_not_was_free/np.sum(only_prefer_not[p]) # % of "prefer not" that was satisfied
            else:
                prefer_not_satisfy_rate[p] = 0#np.nan
        
        # Save preference rates
        preference_stats_df = physician_df.copy()
        preference_stats_df['# prefered not'] = np.sum(only_prefer_not, axis=1)
        preference_stats_df['% prefered not'] = prefer_not_satisfy_rate
        preference_stats_df['# prefered'] = np.sum(only_prefer, axis=1)
        preference_stats_df['% prefered'] = prefer_satisfy_rate
        preference_stats_df = preference_stats_df[['name', 'title', '# prefered', '% prefered', '# prefered not', '% prefered not']]
        #print(preference_stats_df)
        preference_stats_df.to_csv('data/results/preference_stats_df.csv')

    x_size = width
    y_size = n_physicians/n_shifts * x_size + 1
    fig, ax = plt.subplots(figsize=(x_size,y_size))
    result = ax.pcolor(np.arange(n_shifts+1)-0.5, np.arange(n_physicians+1)-0.5,result_matrix, cmap="Greens")
    x, y = np.meshgrid(np.arange(n_shifts), np.arange(n_physicians)) 

    if cl>=2:  # Preference squares
        pref = ax.scatter(x.ravel(), y.ravel(), s=(50*(x_size/n_shifts))**2, c='none',marker='s', linewidths=9, edgecolors=prefer_colors)
        pref_met = ax.pcolor([n_shifts-0.5,n_shifts], np.arange(n_physicians+1)-0.5, prefer_satisfy_rate, cmap='RdYlGn', shading='auto', vmin=0, vmax=1) 
        pref_not_met = ax.pcolor([n_shifts,n_shifts+0.5], np.arange(n_physicians+1)-0.5, prefer_not_satisfy_rate, cmap='RdYlGn', shading='auto', vmin=0,vmax=1) 
        ext = ax.pcolor([n_shifts+0.5,n_shifts+1], np.arange(n_physicians+1)-0.5, extent_error, cmap='RdYlGn', shading='auto', vmin=-100, vmax=100) 
        #sat = ax.pcolor([n_shifts+0.5,n_shifts+1], np.arange(n_physicians+1)-0.5, physician_df['satisfaction'], cmap='RdYlGn') #vmin=?,vmax=?) 

        #prefer_satisfy_rate = np.where(prefer_satisfy_rate ==np.NaN, 0, prefer_satisfy_rate) # tried rate = NaN if no preferences, instead of 0%, got error
        #prefer_not_satisfy_rate = np.where(prefer_not_satisfy_rate==np.NaN, 0, prefer_not_satisfy_rate)
        for p in range(n_physicians):
            #print(prefer_satisfy_rate[p][0]*100)
            #print(prefer_not_satisfy_rate[p][0]*100)
            pref_text = ax.text(n_shifts-0.25,p, str(int(prefer_satisfy_rate[p][0]*100)), ha="center", va="center", color="black", fontsize=8,zorder=10)
            pref_not_text = ax.text( n_shifts+0.25,p, str(int(prefer_not_satisfy_rate[p][0]*100)), ha="center", va="center", color="black", fontsize=8, zorder=10)
            #sat_text = ax.text( n_shifts+0.75,p, str(physician_df['satisfaction'].iloc[p]), ha="center", va="center", color="black", fontsize=8, zorder=10)
            ext_text = ax.text(n_shifts+0.75,p,str(int(extent_error[p][0])), ha="center", va="center", color="black", fontsize=8, zorder=10)

    # Shift covered row
    shift = ax.pcolor(np.arange(n_shifts+1)-0.5,[n_physicians-0.5,n_physicians-0.4], ok_row, cmap='RdYlGn', vmin=0,vmax=1) 
    
    xticks = [i for i in np.arange(n_shifts)]+[n_shifts-0.25]+[n_shifts+0.25]
    ax.set_xticks(ticks=xticks, labels=[date[5:] for date in result_df['date']]+['pref.\n%', 'pref.\nnot %'],fontsize=8) # NOTE removed year from ticks
    yticks = [i for i in np.arange(n_physicians)]+[n_physicians-0.4]
    ax.set_yticks(ticks=yticks, labels=[phys[-1] for phys in physician_df['name']]+['OK n.o.\nworkers'])
    ax.spines["right"].set_linewidth(0) # remove right side of frame
    ax.spines["top"].set_linewidth(0) 
    plt.subplots_adjust(left=0.05, right=0.95,bottom=0.3) # Adjust padding


    fig.savefig('data/results/schedule.png')
    plt.show()