import pandas as pd

def bitstringToSchedule(bitstring, empty_calendar_df, cl, encoding, prints=True) -> pd.DataFrame:
    if cl ==  1:
        staff_col = [[] for _ in range(empty_calendar_df.shape[0])]
        print(empty_calendar_df.shape[0])
        print(staff_col)
        print(len(staff_col))
        for x in list(bitstring.keys()): # TODO more efficient bitstring encodings, not dict?
            if bitstring[x] == 1:
                p, s = x[1], int(x[2]) # TODO add digits, if more than 10 physicians
                print(s)
                staff_col[s].append('p'+p)
        result_schedule_df = empty_calendar_df.copy()
        result_schedule_df['staff'] = staff_col
        result_schedule_df.to_csv('data/result_schedule_cl1.csv')
        return result_schedule_df

    else:
        print('BitstringToSchedule not done for cl'+str(cl))



def controlSchedule(result_schedule_df, demand_df):
    '''combined_df = demand_df.merge(result_schedule_df, left_on='date',right_on='date', how='outer')
    for i in range(combined_df.shape[0]):
        pass
    print(combined_df)'''
    pass
