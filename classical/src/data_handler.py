import pandas as pd

def load_data(shifts_file, physicians_file):
    """Loads and processes shift and physician data"""
    shifts_df = pd.read_csv(shifts_file)
    physicians_df = pd.read_csv(physicians_file)

    physicians = list(physicians_df["Physician"])
    shifts = list(shifts_df["Shift"])
    demand = dict(zip(shifts_df["Shift"], shifts_df["Demand"]))

    # Extract Shift columns (1-21)
    shift_columns = [col for col in physicians_df.columns if "Shift" in col]

    # Convert to preference dictionary
    preference = {}
    for _, row in physicians_df.iterrows():
        physician_name = row["Physician"]
        shift_prefs = {shift: row[shift] for shift in shift_columns}
        preference[physician_name] = shift_prefs

    return physicians, shifts, demand, preference