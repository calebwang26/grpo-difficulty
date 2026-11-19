"""Figures & summary tables, all built from ``outputs/results.json``.

Run ``python -m grpo_difficulty.plots`` after the experiment has been run.
"""

import json
import os

import numpy as np

from . import style
from .grpo import SIGNAL_BUCKETS
from .policy import BIN_NAMES
from .signal import signal_fraction

MAIN_ARMS = ("hard", "sweet", "easy", "mixed")
ALL_ARM_LABELS = {
    **style.BIN_LABELS,
    "no_hard": "Random mix minus too-hard",
    "no_easy": "Random mix minus too-easy",
}
SHORT_LABELS = {"hard": "Too hard", "sweet": "Sweet spot", "easy": "Too easy", "mixed": "Random mix"}


def load_results(path="outputs/results.json"):
    with open(path) as f:
        return json.load(f)


def eval_curve(runs, key):
    """Stack one logged number across seeds. Returns (steps, array[seeds, evals])."""
    steps = [e["step"] for e in runs[0]["evals"]]
    values = np.array([[e[key] for e in run["evals"]] for run in runs])
    return np.array(steps), values


def final_values(runs, key="test_pass"):
    return np.array([run["evals"][-1][key] for run in runs])


def moving_average(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# --------------------------------------------------------------------------- figures


def plot_heldout_accuracy(results, path):
    """Held-out pass rate against training step, one line per arm."""
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    start = results["setup"]["test_pass"]
    ax.axhline(start, color=style.INK_MUTED, linewidth=1.2, zorder=1)
    last_step = results["config"]["steps"]
    ax.annotate("Pretrained checkpoint (no GRPO)", xy=(last_step, start), xytext=(-4, 4),
                textcoords="offset points", ha="right", color=style.INK_SECONDARY, fontsize=9)

    for arm in MAIN_ARMS:
        steps, values = eval_curve(results["runs"][arm], "test_pass")
        mean, spread = values.mean(axis=0), values.std(axis=0)
        color = style.BIN_COLORS[arm]
        ax.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(steps, mean, color=color, label=style.BIN_LABELS[arm], zorder=3)
        ax.annotate(f"{mean[-1]:.1%}", xy=(last_step, mean[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=style.INK, fontsize=9)

    ax.set_xlim(0, last_step)
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Held-out test pass rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    n_seeds = len(results["runs"]["sweet"])
    ax.set_title("Held-out pass rate by the difficulty of the training tasks")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, title=None)
    ax.text(0.0, -0.36, f"Lines: mean of {n_seeds} seeds. Bands: ± 1 standard deviation.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=style.INK_MUTED)
    fig.savefig(path)
    plt.close(fig)


def plot_signal_fraction(results, path):
    """Left: share of groups with signal over training. Right: the same share against
    pass rate, laid over the analytic curve from signal.py."""
    import matplotlib.pyplot as plt

    style.apply_style()
    G = results["config"]["group_size"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.24})

    # Left panel: measured (solid, smoothed) against predicted (dotted).
    window = max(1, min(50, results["config"]["steps"] // 4))
    for arm in MAIN_ARMS:
        runs = results["runs"][arm]
        color = style.BIN_COLORS[arm]
        measured = np.mean([run["steps"]["nonzero_fraction"] for run in runs], axis=0)
        smooth = moving_average(measured, window)
        steps = np.arange(window, len(measured) + 1)
        left.plot(steps, smooth, color=color, label=style.BIN_LABELS[arm], zorder=3)
        eval_steps, predicted = eval_curve(runs, "predicted_signal")
        left.plot(eval_steps, predicted.mean(axis=0), color=color, linewidth=1.3,
                  linestyle=(0, (1, 2)), zorder=2)
    left.plot([], [], color=style.INK_SECONDARY, label=f"Measured ({window}-step average)")
    left.plot([], [], color=style.INK_SECONDARY, linewidth=1.3, linestyle=(0, (1, 2)),
              label="Predicted by 1 − p^G − (1−p)^G")
    left.set_xlim(0, results["config"]["steps"])
    left.set_ylim(0, 1)
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.set_xlabel("GRPO training step")
    left.set_ylabel("Groups with a nonzero advantage")
    left.set_title("Share of groups that carry any gradient")
    left.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2)

    # Right panel: every sampled group, sorted by its task's pass rate at that moment.
    p = np.linspace(0, 1, 401)
    right.plot(p, signal_fraction(p, G), color=style.INK_SECONDARY, linewidth=1.6, zorder=2,
               label=f"Theory, G = {G}")
    centres = (np.arange(SIGNAL_BUCKETS) + 0.5) / SIGNAL_BUCKETS
    for arm in BIN_NAMES:
        groups = np.sum([run["signal_by_pass_rate"]["groups"] for run in results["runs"][arm]], axis=0)
        mixed = np.sum([run["signal_by_pass_rate"]["mixed"] for run in results["runs"][arm]], axis=0)
        enough = groups >= 500
        right.scatter(centres[enough], mixed[enough] / groups[enough], s=46,
                      color=style.BIN_COLORS[arm], edgecolor=style.SURFACE, linewidth=1.5,
                      zorder=3, label=f"Measured, {SHORT_LABELS[arm].lower()} run")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1.03)
    right.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    right.set_xlabel("Task's pass rate when its group was sampled")
    right.set_ylabel("Groups with a nonzero advantage")
    right.set_title("Measured groups sit on the theory curve")
    right.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def plot_gain_by_difficulty(results, path):
    """Change in test pass rate, split by how hard each test task was at the start."""
    import matplotlib.pyplot as plt

    style.apply_style()
    groups = [("test_pass_hard", "Test tasks that\nstarted too hard"),
              ("test_pass_sweet", "Test tasks that\nstarted in the sweet spot"),
              ("test_pass_easy", "Test tasks that\nstarted too easy"),
              ("test_pass", "All test tasks")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.19
    for i, arm in enumerate(MAIN_ARMS):
        runs = results["runs"][arm]
        gains = []
        for key, _ in groups:
            _, values = eval_curve(runs, key)
            gains.append(values[:, -1].mean() - values[:, 0].mean())
        xs = np.arange(len(groups)) + (i - 1.5) * (width + 0.015)
        ax.bar(xs, gains, width=width, color=style.BIN_COLORS[arm],
               label=f"Trained on: {style.BIN_LABELS[arm].lower()}", zorder=3)
        for x, gain in zip(xs, gains):
            label = "0" if round(gain * 100) == 0 else f"{gain * 100:+.0f}"
            ax.annotate(label, xy=(x, gain), xytext=(0, 3 if gain >= 0 else -3),
                        textcoords="offset points", ha="center",
                        va="bottom" if gain >= 0 else "top", fontsize=8, color=style.INK_SECONDARY)
    ax.axhline(0, color=style.AXIS, linewidth=1.0, zorder=2)
    ax.axvline(2.5, color=style.GRID, linewidth=1.0, zorder=1)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([label for _, label in groups])
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v * 100:+.0f}")
    ax.set_ylabel("Change in pass rate (percentage points)")
    ax.set_title("Where each run gains & loses, by the test task's starting difficulty")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- tables


def summarize(results):
    """Key numbers as markdown tables. This is where the README's numbers come from."""
    setup, config = results["setup"], results["config"]
    start = setup["test_pass"]
    lines = []

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Pass rate after pretraining: pretrain {setup['pretrain_pass']:.1%}, "
                 f"pool {setup['pool_pass']:.1%}, val {setup['val_pass']:.1%}, test {start:.1%}")
    lines.append(f"- Bin sizes before the cut: {setup['bin_sizes_before_cut']}, "
                 f"cut to {setup['tasks_per_arm']} tasks per arm")
    lines.append(f"- Pool tasks at 0 of {config['n_measure']} correct: {setup['pool_tasks_at_zero_correct']}")
    lines.append(f"- Test tasks by starting difficulty: {setup['test_group_sizes']}")
    lines.append(f"- GRPO: {config['steps']} steps × {config['batch_tasks']} tasks × G = {config['group_size']}"
                 f" = {config['steps'] * config['batch_tasks'] * config['group_size']:,} samples per run,"
                 f" lr {config['lr']:g}")
    for k, row in setup["pool_pass_at_k"].items():
        lines.append(f"- Pool pass@{k}: unbiased {row['unbiased']:.3f} "
                     f"(95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f}), naive {row['naive']:.3f}, "
                     f"truth {row['truth']:.3f}")
    lines.append("")

    lines.append("## Learning-rate selection (mixed arm, validation split)")
    lines.append("")
    lines.append("| Learning rate | Validation pass rate at the end |")
    lines.append("| --- | --- |")
    for row in results["lr_selection"]["table"]:
        chosen = " (chosen)" if row["lr"] == results["lr_selection"]["chosen"] else ""
        lines.append(f"| {row['lr']:g}{chosen} | {row['val_pass_final_mean']:.1%} |")
    lines.append(f"\nValidation pass rate before GRPO: {results['lr_selection']['table'][0]['val_pass_start']:.1%}")
    lines.append("")

    lines.append("## Main result")
    lines.append("")
    lines.append("| Trained on | Start pass rate of its tasks | Groups with signal at start | "
                 "Held-out pass rate (mean ± sd) | Change | Greedy accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(f"| (no GRPO, pretrained checkpoint) | | | {start:.1%} | | {setup['test_greedy']:.1%} |")
    for arm in MAIN_ARMS + ("no_hard", "no_easy"):
        if arm not in results["runs"]:
            continue
        finals = final_values(results["runs"][arm])
        greedy = final_values(results["runs"][arm], "test_greedy")
        info = setup["arms"][arm]
        lines.append(f"| {ALL_ARM_LABELS[arm]} | {info['true_pass']:.1%} | "
                     f"{info['predicted_signal_at_start']:.0%} | {finals.mean():.1%} ± {finals.std():.1%} | "
                     f"{(finals.mean() - start) * 100:+.1f} pts | {greedy.mean():.1%} |")
    lines.append("")

    lines.append("## Held-out pass rate at different budgets (mean over seeds)")
    lines.append("")
    budgets = [s for s in (100, 300, 1000, 3000) if s <= config["steps"]]
    lines.append("| Trained on | " + " | ".join(f"step {b}" for b in budgets) + " |")
    lines.append("| --- |" + " --- |" * len(budgets))
    for arm in MAIN_ARMS:
        steps, values = eval_curve(results["runs"][arm], "test_pass")
        cells = [f"{values[:, list(steps).index(b)].mean():.1%}" for b in budgets]
        lines.append(f"| {ALL_ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Where the change comes from (pass rate, start → end)")
    lines.append("")
    lines.append("| Trained on | Its own training tasks | Test: started too hard | "
                 "Test: started sweet spot | Test: started too easy |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in MAIN_ARMS:
        cells = []
        for key in ("train_pass", "test_pass_hard", "test_pass_sweet", "test_pass_easy"):
            _, values = eval_curve(results["runs"][arm], key)
            cells.append(f"{values[:, 0].mean():.1%} → {values[:, -1].mean():.1%}")
        lines.append(f"| {ALL_ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Ablations (held-out pass rate at the end, mean over seeds)")
    lines.append("")
    lines.append("| Variant | Too hard | Sweet spot | Too easy | Random mix |")
    lines.append("| --- | --- | --- | --- | --- |")
    main_cells = [f"{final_values(results['runs'][arm]).mean():.1%}" for arm in MAIN_ARMS]
    lines.append(f"| Main (G = {config['group_size']}, lr {config['lr']:g}) | " + " | ".join(main_cells) + " |")
    for name, ablation in results["ablations"].items():
        cells = [f"{final_values(ablation['runs'][arm]).mean():.1%}" if arm in ablation["runs"] else "–"
                 for arm in MAIN_ARMS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Replicates (new world shuffle & new pretrained checkpoint each)")
    lines.append("")
    lines.append("| World seed | Start | Too hard | Sweet spot | Too easy | Random mix |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(f"| {config['world']['seed']} (main) | {start:.1%} | " + " | ".join(main_cells) + " |")
    for rep in results["replicates"]:
        cells = [f"{final_values(rep['runs'][arm]).mean():.1%}" for arm in MAIN_ARMS]
        lines.append(f"| {rep['world_seed']} | {rep['setup']['test_pass']:.1%} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(results_path="outputs/results.json"):
    results = load_results(results_path)
    os.makedirs("figures", exist_ok=True)
    plot_heldout_accuracy(results, "figures/heldout_accuracy.png")
    plot_signal_fraction(results, "figures/signal_fraction.png")
    plot_gain_by_difficulty(results, "figures/gain_by_difficulty.png")
    summary = summarize(results)
    with open("outputs/summary.md", "w") as f:
        f.write(summary)
    print(summary)
    print("Saved figures/heldout_accuracy.png, figures/signal_fraction.png, "
          "figures/gain_by_difficulty.png & outputs/summary.md")


if __name__ == "__main__":
    main()
