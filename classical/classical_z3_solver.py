import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from z3 import *

# Set random seed for reproducibility
np.random.seed(313)

# Load data
data = pd.read_csv("expanded_physician_schedule_data.csv")

# Extract unique physicians and shifts
P = data["Physician"].unique().tolist()
S = data["Shift"].unique().tolist()

# Dictionaries
D = data.groupby("Shift")["Required"].first().to_dict()
W_max = data.groupby("Physician")["Max_Shifts"].first().to_dict()
availability = {(row["Physician"], row["Shift"]): row["Available"] for _, row in data.iterrows()}
preferences = data.pivot(index="Physician", columns="Shift", values="Preference").fillna(5)
time_off = data.pivot(index="Physician", columns="Shift", values="Time_Off").fillna(0).astype(int)

# Solver
z3_solver = Optimize()

# Decision Variables
x = {(p, s): Bool(f"x_{p}_{s}") for p in P for s in S if availability.get((p, s), 0) == 1}

# Objective: Minimize dissatisfaction
dissatisfaction = Sum([preferences.loc[p, s] * If(x[(p, s)], 1, 0) for p in P for s in S if (p, s) in x])

# Hard Constraints
for s in S:
    z3_solver.add(Sum([If(x[(p, s)], 1, 0) for p in P if (p, s) in x]) == D.get(s, 1))

for p in P:
    z3_solver.add(Sum([If(x[(p, s)], 1, 0) for s in S if (p, s) in x]) <= W_max.get(p, 2))

for p in P:
    for i in range(len(S) - 1):
        s1, s2 = S[i], S[i + 1]
        if (p, s1) in x and (p, s2) in x:
            z3_solver.add(Or(Not(x[(p, s1)]), Not(x[(p, s2)])))

for p in P:
    for s in S:
        if (p, s) in x and time_off.loc[p, s] == 1:
            z3_solver.add(Not(x[(p, s)]))

# Solve the problem
z3_solver.minimize(dissatisfaction)

if z3_solver.check() == sat:
    model = z3_solver.model()
    schedule = pd.DataFrame(0, index=P, columns=S)  # Initialize schedule matrix

    print("Z3 Schedule:")
    for (p, s), assigned in x.items():
        if model.evaluate(x[(p, s)]):
            print(f"Physician {p} assigned to shift {s}")
            schedule.loc[p, s] = 1  # Mark as assigned

    # Convert DataFrames to NumPy arrays
    pref_array = preferences.values
    schedule_array = schedule.values

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Preferences
    ax1 = axes[0]
    im1 = ax1.imshow(pref_array, cmap="coolwarm", aspect="auto")
    ax1.set_xticks(range(len(S)))
    ax1.set_yticks(range(len(P)))
    ax1.set_xticklabels(S, rotation=45)
    ax1.set_yticklabels(P)
    ax1.set_title("Physician Shift Preferences")
    ax1.set_xlabel("Shifts")
    ax1.set_ylabel("Physicians")
    fig.colorbar(im1, ax=ax1, label="Preference Score")

    # Plot Final Schedule with Availability
    ax2 = axes[1]
    schedule_display = np.copy(schedule_array)

    for i, p in enumerate(P):
        for j, s in enumerate(S):
            if availability.get((p, s), 0) == 0:  # Physician not available for this shift
                schedule_display[i, j] = -1  # Mark as unavailable (for color coding)

    im2 = ax2.imshow(schedule_display, cmap="Greens", aspect="auto")

    # Add "N/A" text where physicians are unavailable
    for i, p in enumerate(P):
        for j, s in enumerate(S):
            if availability.get((p, s), 0) == 0:  # Not available
                ax2.text(j, i, "N/A", ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(S)))
    ax2.set_yticks(range(len(P)))
    ax2.set_xticklabels(S, rotation=45)
    ax2.set_yticklabels(P)
    ax2.set_title("Final Schedule (Assignments & Availability)")
    ax2.set_xlabel("Shifts")
    ax2.set_ylabel("Physicians")
    fig.colorbar(im2, ax=ax2, label="Assigned (1=Yes, 0=No)")

    plt.tight_layout()
    plt.show()

else:
    print("No solution found with Z3.")
