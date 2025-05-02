import pandas as pd
import os

def load_data_from_intermediate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data", "intermediate")
    shifts_path = os.path.join(data_dir, "shift_data_all_t.csv")
    physicians_path = os.path.join(data_dir, "physician_data.csv")

    shifts_df = pd.read_csv(shifts_path)
    physicians_df = pd.read_csv(physicians_path)

    # Build shift labels and base dates
    if "shift type" in shifts_df.columns:
        shift_dates = shifts_df["date"].astype(str).str.strip().values
        shift_types = shifts_df["shift type"].astype(str).values
        shifts = [f"{d} ({t})" for d, t in zip(shift_dates, shift_types)]
    else:
        shift_dates = shifts_df["date"].astype(str).str.strip().values
        shifts = list(shift_dates)

    # Build demand dictionary
    demand = {}
    for i, row in shifts_df.iterrows():
        label = str(row['date']).strip()
        if "shift type" in row:
            label += f" ({row['shift type']})"
        demand[label] = int(row['demand'])

    physicians = physicians_df["name"].values
    preference = {p: {s: 0 for s in shifts} for p in physicians}

    def parse_date_set(date_string):
        if pd.isna(date_string) or date_string == '[]':
            return set()
        return set(d.strip(" '\"") for d in date_string.strip('[]').split(',') if d.strip())

    # Process each physician
    for idx, p in enumerate(physicians):
        row = physicians_df.iloc[idx]
        prefer_dates = parse_date_set(row["prefer"])
        prefer_not_dates = parse_date_set(row["prefer not"])
        unavailable_dates = parse_date_set(row["unavailable"])
        pref_row = preference[p]

        for s_label, s_date in zip(shifts, shift_dates):
            if s_date in unavailable_dates:
                pref_row[s_label] = -2
            elif s_date in prefer_not_dates:
                pref_row[s_label] = -1
            elif s_date in prefer_dates:
                pref_row[s_label] = 1
            else:
                pref_row[s_label] = 0

    return list(physicians), shifts, demand, preference
