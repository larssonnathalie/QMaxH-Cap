import pandas as pd

def load_data(shifts_file, physicians_file):
    """Loads and processes shift and physician data"""
    shifts_df = pd.read_csv(shifts_file)
    physicians_df = pd.read_csv(physicians_file)

    physicians = list(physicians_df["Physician"])
    shifts = list(shifts_df["Shift"])
    demand = dict(zip(shifts_df["Shift"], shifts_df["Demand"]))

    # Convert to preference dictionary
    preference = {}
    for _, row in physicians_df.iterrows():
        physician_name = row["Physician"]
        preference[physician_name] = {shift: row[shift] for shift in shifts}

    return physicians, shifts, demand, preference