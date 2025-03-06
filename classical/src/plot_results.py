import matplotlib.pyplot as plt

def plot_schedule(best_schedule, physicians, shifts):
    """Plots the best schedule using a scatter plot"""
    fix, ax = plt.subplots(figsize=(12,6))

    # Convert shift names to indices for plotting
    shift_indices = {s: i for i, s in enumerate(shifts)}
    physician_indices = {p: i for i, p in enumerate(physicians)}

    # Plot scheduled shifts as scatter points
    for p, assigned_shifts in best_schedule.items():
        y = [physician_indices[p]] * len(assigned_shifts)
        x = [shift_indices[s] for s in assigned_shifts]
        ax.scatter(x,y, label=p, marker='s', s=100)

    # Format plot
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(shifts, rotation=90)
    ax.set_yticks(range(len(physicians)))
    ax.set_yticklabels(physicians)
    ax.set_xlabel("Shifts")
    ax.set_ylabel("Physicians")
    ax.set_title("Best Scheduled Physicians (Minimized Dissatisfaction)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()