### QUBO Model for Physician Scheduling
import numpy as np
import dimod
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(313)

# Load data from CSV file
csv_filename = "physician_schedule_data.csv"
#csv_filename = "expanded_physician_schedule_data.csv"
data = pd.read_csv(csv_filename)

# Extract unique physicians and shifts
P = data["Physician"].unique().tolist()
S = data["Shift"].unique().tolist()

# Convert CSV data into relevant dictionaries
D = data.groupby("Shift")["Required"].first().to_dict()  # Required physicians per shift
W_max = data.groupby("Physician")["Max_Shifts"].first().to_dict()  # Max shifts per physician

# Availability Matrix
availability = {(row["Physician"], row["Shift"]): row["Available"] for _, row in data.iterrows()}

# Dissatisfaction scores (Preferences)
preferences = data.pivot(index="Physician", columns="Shift", values="Preference").fillna(5).to_dict()

# Time-off requests
R = data.pivot(index="Physician", columns="Shift", values="Time_Off").fillna(0).astype(int).to_dict()

# Define QUBO model
Q = {}

# Constraint Weights
lambda_1 = 4  # Physician shift dissatisfaction penalty
lambda_2 = 1  # Workload balance penalty
lambda_3 = 2  # Minimum rest time penalty
lambda_4 = 10  # Shift coverage constraint penalty

# Objective function: Minimize dissatisfaction + penalties
for p in P:
    for s in S:
        if availability.get((p, s), 0) == 1:  # Only consider if physician is available
            Q[(p, s), (p, s)] = (
                lambda_1 * preferences[s][p] +
                lambda_2 * R[s].get(p, 0)  # Apply time-off request penalty
            )
        else:
            Q[(p, s), (p, s)] = float("inf")  # Strongly discourage unavailable assignments

# Shift coverage constraint (Quadratic Formulation)
for s in S:
    for p in P:
        for p_prime in P:
            if p != p_prime:
                Q[(p, s), (p_prime, s)] = lambda_4  # Penalize assignment of different physicians on the same shift
    Q[(p, s), (p, s)] -= lambda_4 * D[s]  #

# Workload balance constraint (Quadratic Formulation)
for p in P:
    for s in S:
        for s_prime in S:
            if s != s_prime:
                Q[(p, s), (p, s_prime)] = lambda_2
    Q[(p, s), (p, s)] -= lambda_2 * 2 * W_max.get(p, 2)

# Minimum Rest Time Constraint
for p in P:
    for s in range(len(S) - 1):
        Q[(p, S[s]), (p, S[s+1])] = lambda_3  # Penalize consecutive shifts

# Convert to binary quadratic model
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# Format QUBO output for readability
qubo_data = []
for key, value in Q.items():
    qubo_data.append({"Variable 1": key[0], "Variable 2": key[1], "Weight": value})
qubo_df = pd.DataFrame(qubo_data)

# Display QUBO model in a readable format
print("QUBO Model Representation:")
print(qubo_df.to_string(index=False))

# Plot preferences matrix
preferences_df = pd.DataFrame(preferences).fillna(5)
print("\nPhysician Shift Preferences:")
print(preferences_df.to_string())

plt.figure(figsize=(8, 6))
plt.imshow(preferences_df.values, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Dissatisfaction Score")
plt.xticks(range(len(S)), S)
plt.yticks(range(len(P)), P)
plt.xlabel("Shifts")
plt.ylabel("Physicians")
plt.title("Physician Shift Preferences")
plt.show()

# Ensure consistent ordering of QUBO variables
qubo_keys = sorted(
    list(set([key[0] for key in Q.keys()] + [key[1] for key in Q.keys()])),
    key=lambda x: (P.index(x[0]), S.index(x[1]))
)
index_map = {key: idx for idx, key in enumerate(qubo_keys)}

# Construct QUBO matrix in the correct order
qubo_matrix = np.zeros((len(qubo_keys), len(qubo_keys)))
for (var1, var2), value in Q.items():
    idx1, idx2 = index_map[var1], index_map[var2]
    qubo_matrix[idx1][idx2] = value
    qubo_matrix[idx2][idx1] = value  # Ensure symmetry

plt.figure(figsize=(12, 10))
plt.imshow(qubo_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="QUBO Weight")

# Generate readable labels
qubo_labels = [f"P{P.index(v1)+1}-S{S.index(v2)+1}" for (v1, v2) in qubo_keys]

plt.xticks(range(len(qubo_keys)), qubo_labels, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(qubo_keys)), qubo_labels, fontsize=8)

plt.xlabel("QUBO Variables")
plt.ylabel("QUBO Variables")
plt.title("QUBO Matrix Representation (Ordered)")

plt.tight_layout()
plt.show()
