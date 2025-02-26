import pandas as pd

def variableNameToPS(var_name, n_shifts):
    var_name = var_name.lstrip('x')
    idx = int(var_name)
    p = int(idx/n_shifts)
    s = idx%n_shifts
    return p,s

def psVarNamesToI(xp_s, n_shifts): # Might not be needed
        p,s = xp_s.lstrip('x').split('_')
        i = int(p) * n_shifts + int(s)
        return f'x{i}'

def bitstringToSchedule(bitstring, empty_calendar_df, cl, n_shifts, prints=True) -> pd.DataFrame:
    if cl ==  1:
        staff_col = [[] for _ in range(empty_calendar_df.shape[0])]
 
        for var_name in list(bitstring.keys()): # TODO more efficient bitstring encodings, not dict?
            if bitstring[var_name] == 1:
                #p, s = x[1], int(x[3]) # TODO add digits, if more than 10 physicians. ex split on "_"
                p,s = variableNameToPS(var_name, n_shifts)
                staff_col[s].append('p'+str(p))
    
        result_schedule_df = empty_calendar_df.copy()
        result_schedule_df['staff'] = staff_col
        result_schedule_df.to_csv(f'data/output/result_schedule_cl{cl}.csv', index=False)
        return result_schedule_df

    else:
        print('BitstringToSchedule not done for cl'+str(cl))



def controlSchedule(result_schedule_df, demand_df, cl, prints=False):
    combined_df = demand_df.merge(result_schedule_df, on='date', how='outer')
    for i in range(combined_df.shape[0]):
        if combined_df.loc[i,'demand'] == len(combined_df.loc[i,'staff']):
            if prints:
                print(str(combined_df.loc[i, 'date'])+' is OK') # TODO add as column in df instead
        else:
            print(str(combined_df.loc[i, f'date'])+' NOT ok!')
    print(combined_df)
    combined_df.to_csv(f'data/output/result_and_demand_cl{cl}.csv', index=False)
    
