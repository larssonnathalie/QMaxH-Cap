import pandas as pd

def bitstringIndexToPS(idx, n_shifts):
    idx=int(idx)
    p = int(idx/n_shifts)
    s = idx%n_shifts
    return p,s

def psVarNamesToI(xp_s, n_shifts): # Might not be needed
        p,s = xp_s.lstrip('x').split('_')
        i = int(p) * n_shifts + int(s)
        return f'x{i}'

def bitstringToSchedule(bitstring:str, empty_calendar_df, cl, n_shifts, prints=True) -> pd.DataFrame:
    if cl == 1:
        staff_col = [[] for _ in range(empty_calendar_df.shape[0])]
 
        for i, bit in enumerate(bitstring):
            if bit == '1':
                p,s = bitstringIndexToPS(i, n_shifts)
                staff_col[s].append(str(p))
    
        result_schedule_df = empty_calendar_df.copy()
        result_schedule_df['staff'] = staff_col
        result_schedule_df.to_csv(f'data/output/result_schedule_cl{cl}.csv', index=False)
        return result_schedule_df

    else:
        print('BitstringToSchedule not done for cl'+str(cl))

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
    combined_df.to_csv(f'data/output/result_and_demand_cl{cl}.csv', index=False)
    
