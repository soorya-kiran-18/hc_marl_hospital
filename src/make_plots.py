import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_results():
    results = {}
    for path in glob.glob("docs/*_results.csv"):
        name = os.path.basename(path).replace("_results.csv", "")
        results[name] = pd.read_csv(path)
    return results


def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, df in results.items():
        if not name.startswith("HC-MARL"):
            continue
        curve = df.groupby('episode')['reward'].mean()
        ax.plot(curve.index, curve.values, label=name)
    ax.set_xlabel("Training episode (simulated day)")
    ax.set_ylabel("Mean episode reward (across seeds)")
    ax.set_title("HC-MARL Training Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/training_convergence.png", dpi=150)
    plt.close(fig)


def plot_comparison_bars(summary_path="docs/comparison_summary.csv"):
    summary = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(summary['method'], summary['mean_reward'], yerr=summary['std_reward'])
    axes[0].set_title("Mean Reward (converged)")
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(summary['method'], summary['mean_safety_violation_rate'], color='indianred')
    axes[1].set_title("Safety Violation Rate")
    axes[1].tick_params(axis='x', rotation=30)

    axes[2].bar(summary['method'], summary['mean_equity_gap_steps'], color='seagreen')
    axes[2].set_title("Equity Gap (15-min steps)")
    axes[2].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig("docs/comparative_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    results = load_all_results()
    if not results:
        print("No result CSVs found in docs/. Run python3 -m src.train first.")
    else:
        plot_convergence(results)
        plot_comparison_bars()
        print("Saved docs/training_convergence.png and docs/comparative_analysis.png")
