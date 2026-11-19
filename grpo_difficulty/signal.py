"""The theory: how often does a GRPO group carry any learning signal?

GRPO samples G answers for one task & scores each one 1 (correct) or 0 (wrong).
Each answer's advantage is its reward minus the group's average reward.

If all G answers are correct, every advantage is 1 - 1 = 0.
If all G answers are wrong, every advantage is 0 - 0 = 0.
Either way the loss is zero & so is the gradient: the group teaches nothing.

For a task the policy solves with probability p:

    P(all correct) = p^G
    P(all wrong)   = (1-p)^G
    P(group carries signal) = 1 - p^G - (1-p)^G

Run ``python -m grpo_difficulty.signal`` to print a table & save the plot.
"""

import numpy as np


def signal_fraction(p, G):
    """Probability that a group of G samples is mixed (not all right, not all wrong).

    Works on a single pass rate or a numpy array of pass rates.
    """
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pass rates must be between 0 and 1")
    if G < 1:
        raise ValueError("group size must be at least 1")
    return 1.0 - p**G - (1.0 - p) ** G


def mean_squared_advantage(p, G):
    """Expected size of the advantages in a group: E[mean over group of (r - mean r)^2].

    This equals ``p(1-p)(G-1)/G``. It says *how strong* the signal is on average,
    while ``signal_fraction`` only says how often there is any. Both peak at p = 0.5
    & both are zero at p = 0 & p = 1.
    """
    p = np.asarray(p, dtype=float)
    return p * (1.0 - p) * (G - 1) / G


def simulated_signal_fraction(p, G, trials=100000, seed=0):
    """Check the formula by brute force: sample many groups & count the mixed ones."""
    rng = np.random.default_rng(seed)
    correct = rng.random((trials, G)) < p
    n_correct = correct.sum(axis=1)
    mixed = (n_correct > 0) & (n_correct < G)
    return float(mixed.mean())


def plot_signal_theory(path, group_sizes=(4, 8, 16)):
    """Plot the signal fraction against pass rate for several group sizes."""
    import matplotlib.pyplot as plt

    from . import style

    style.apply_style()
    # Group size is an ordered quantity, so the lines share one hue, light to dark.
    shades = ["#86b6ef", "#2a78d6", "#104281"]
    p = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # Mark the three difficulty bins used in the experiment.
    for lo, hi, name in [(0.0, 0.1, "too\nhard"), (0.2, 0.8, "sweet spot"), (0.9, 1.0, "too\neasy")]:
        ax.axvspan(lo, hi, color=style.GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.text((lo + hi) / 2, 1.07, name, ha="center", va="bottom", fontsize=8.5,
                color=style.INK_SECONDARY, linespacing=0.95)

    for G, shade in zip(group_sizes, shades):
        ax.plot(p, signal_fraction(p, G), color=shade, label=f"G = {G}", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f"{int(round(v * 100))}%" for v in np.arange(0, 1.01, 0.1)])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(round(v * 100))}%" for v in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Task pass rate p")
    ax.set_ylabel("Groups with any gradient")
    ax.set_title("Share of GRPO groups that carry signal: 1 − p^G − (1−p)^G", pad=30)
    ax.legend(loc="lower center", title="Group size", ncol=3)
    fig.savefig(path)
    plt.close(fig)


def main():
    import os

    group_sizes = (4, 8, 16)
    pass_rates = (0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99)
    print("Fraction of groups that carry any gradient")
    print(f"{'p':>6} " + " ".join(f"{'G=' + str(G):>8}" for G in group_sizes))
    for p in pass_rates:
        print(f"{p:>6.2f} " + " ".join(f"{signal_fraction(p, G):>8.3f}" for G in group_sizes))
    os.makedirs("figures", exist_ok=True)
    plot_signal_theory("figures/signal_theory.png", group_sizes)
    print("Saved figures/signal_theory.png")


if __name__ == "__main__":
    main()
