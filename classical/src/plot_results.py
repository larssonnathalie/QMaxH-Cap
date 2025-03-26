import matplotlib.pyplot as plt

def plot_schedule(schedule, physicians, shifts, title="Schedule"):
    """Plots a given physician schedule."""
    fig, ax = plt.subplots(figsize=(12, 6))

    shift_indices = {s: i for i, s in enumerate(shifts)}
    physician_indices = {p: i for i, p in enumerate(physicians)}

    for p, assigned_shifts in schedule.items():
        y = [physician_indices[p]] * len(assigned_shifts)
        x = [shift_indices[s] for s in assigned_shifts]
        ax.scatter(x, y, label=p, marker='s', s=100)

    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(shifts, rotation=90)
    ax.set_yticks(range(len(physicians)))
    ax.set_yticklabels(physicians)
    ax.set_xlabel("Shifts")
    ax.set_ylabel("Physicians")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def compare_solutions(results_dict):
    """
    Plots a bar chart comparing objective values or scores from different solvers.
    Input: dict {solver_name: value}
    """
    labels = list(results_dict.keys())
    values = [results_dict[label] if isinstance(results_dict[label], (int, float)) else -1 for label in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.ylabel("Objective Value / Dissatisfaction")
    plt.title("Solver Comparison (Lower is Better)")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{val}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()