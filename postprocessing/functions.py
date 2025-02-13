import pandas as pd

def bitstringToSchedule(bitstring, empty_calendar_df, cl, encoding, prints=True) -> pd.DataFrame:
    if cl ==  1:
        staff_col = [[] for _ in range(empty_calendar_df.shape[0])]
 
        for x in list(bitstring.keys()): # TODO more efficient bitstring encodings, not dict?
            if bitstring[x] == 1:
                p, s = x[1], int(x[2]) # TODO add digits, if more than 10 physicians
                staff_col[s].append('p'+p)
    
        result_schedule_df = empty_calendar_df.copy()
        result_schedule_df['staff'] = staff_col
        result_schedule_df.to_csv(f'data/output/result_schedule_cl{cl}.csv', index=False)
        return result_schedule_df

    else:
        print('BitstringToSchedule not done for cl'+str(cl))



def controlSchedule(result_schedule_df, demand_df, cl, prints=True):
    combined_df = demand_df.merge(result_schedule_df, on='date', how='outer')
    for i in range(combined_df.shape[0]):
        if combined_df.loc[i,'demand'] == len(combined_df.loc[i,'staff']):
            if prints:
                print(str(combined_df.loc[i, 'date'])+' is OK') # TODO add as column in df instead
        else:
            print(str(combined_df.loc[i, f'date'])+' NOT ok!')
    print(combined_df)
    combined_df.to_csv(f'data/output/result_and_demand_cl{cl}.csv', index=False)
    
