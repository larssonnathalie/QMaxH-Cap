import pandas as pd
import os

def load_data_from_intermediate():
    """Loads shift and physician data generated from QUBO-style pipeline"""

    # Get absolute path to the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    intermediate_dir = os.path.join(base_dir, "data", "intermediate")

    shifts_path = os.path.join(intermediate_dir, "shift_data.csv")
    physicians_path = os.path.join(intermediate_dir, "physician_data.csv")

    shifts_df = pd.read_csv(shifts_path)
    physicians_df = pd.read_csv(physicians_path)

    if "shift type" in shifts_df.columns:
        shifts = [f"{row['date']} ({row['shift type']})" for _, row in shifts_df.iterrows()]
    else:
        shifts = shifts_df["date"].tolist()

    demand = {}
    for i, row in shifts_df.iterrows():
        shift_label = row['date']
        if "shift type" in row:
            shift_label += f" ({row['shift type']})"
        demand[shift_label] = int(row['demand'])

    physicians = physicians_df["name"].tolist()

    preference = {p: {s: 1 for s in shifts} for p in physicians}
    for idx, row in physicians_df.iterrows():
        p = row["name"]

        def parse_shift_list(column):
            if isinstance(row[column], list):
                return row[column]
            if row[column] == '[]':
                return []
            return [int(s.strip()) for s in row[column].strip('[]').split(',') if s.strip().isdigit()]

        for s_idx in parse_shift_list("prefer not") + parse_shift_list("unavailable"):
            if s_idx < len(shifts):
                preference[p][shifts[s_idx]] = 0

    return physicians, shifts, demand, preference

