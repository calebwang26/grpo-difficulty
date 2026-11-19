"""pass@k: the chance that at least one of k sampled answers is correct.

Three tools live here:

* ``pass_at_k_unbiased``: the estimator from the Codex paper (Chen et al., 2021).
* ``pass_at_k_naive``: the obvious plug-in estimator, which is biased.
* ``bootstrap_ci``: a confidence interval for a benchmark's mean pass@k.

Run ``python -m grpo_difficulty.passk`` for a demo of the naive estimator's bias.
"""

import numpy as np


def pass_at_k_unbiased(n, c, k):
    """Unbiased pass@k from n samples of which c were correct.

    Formula: ``1 - C(n-c, k) / C(n, k)``.

    Why: ``C(n-c, k) / C(n, k)`` is the fraction of size-k subsets of our n samples
    that contain *no* correct answer. Averaging over every subset uses all the data
    & gives exactly the right answer on average.

    The binomial coefficients get huge, so we never compute them. Instead we use
    the identity ``C(n-c, k) / C(n, k) = prod_{i=n-c+1}^{n} (1 - k/i)``,
    which is a product of c numbers that are all between 0 & 1.
    """
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        # Fewer than k wrong answers exist, so every size-k subset has a correct one.
        return 1.0
    i = np.arange(n - c + 1, n + 1)
    return float(1.0 - np.prod(1.0 - k / i))


def pass_at_k_naive(n, c, k):
    """Naive pass@k: estimate p as c/n, then plug it into ``1 - (1-p)^k``.

    This looks right but is biased low. See the demo & EXPLAIN.md for why.
    """
    if n < 1 or k < 1:
        raise ValueError(f"need n >= 1 and k >= 1, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    return float(1.0 - (1.0 - c / n) ** k)


def true_pass_at_k(p, k):
    """The quantity both estimators aim at, when the true pass rate p is known."""
    return 1.0 - (1.0 - p) ** k


def mean_pass_at_k(n, correct_counts, k, estimator=pass_at_k_unbiased):
    """Benchmark score: average pass@k over tasks. Each task had n samples."""
    return float(np.mean([estimator(n, int(c), k) for c in correct_counts]))


def bootstrap_ci(n, correct_counts, k, n_boot=2000, level=0.95, seed=0,
                 estimator=pass_at_k_unbiased):
    """Bootstrap confidence interval for the benchmark's mean pass@k.

    A benchmark is a sample of tasks. To see how much the score would move with a
    different sample, we redraw the task list with replacement many times &
    recompute the score each time. The interval is the middle ``level`` of those
    scores.

    Returns ``(point_estimate, low, high)``.
    """
    counts = np.asarray(correct_counts, dtype=int)
    if counts.size == 0:
        raise ValueError("need at least one task")
    per_task = np.array([estimator(n, int(c), k) for c in counts])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_task), size=(n_boot, len(per_task)))
    boot_means = per_task[picks].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(per_task.mean()), float(low), float(high)


def bias_demo(p=0.1, k=10, sample_sizes=(10, 20, 50, 100, 200), trials=200000, seed=0):
    """Simulate both estimators many times & compare their averages to the truth.

    For each sample size n we run ``trials`` pretend evaluations: draw n answers
    that are each correct with probability p, then compute both estimators.
    An unbiased estimator's average lands on the truth for every n.

    Returns a list of dicts, one per sample size.
    """
    rng = np.random.default_rng(seed)
    truth = true_pass_at_k(p, k)
    rows = []
    for n in sample_sizes:
        counts = rng.binomial(n, p, size=trials)
        # Each distinct count gives one estimator value, so compute those once.
        unbiased_by_c = np.array([pass_at_k_unbiased(n, c, k) for c in range(n + 1)])
        naive_by_c = np.array([pass_at_k_naive(n, c, k) for c in range(n + 1)])
        rows.append(
            {
                "n": n,
                "truth": truth,
                "unbiased_mean": float(unbiased_by_c[counts].mean()),
                "naive_mean": float(naive_by_c[counts].mean()),
            }
        )
    return rows


def plot_bias_demo(rows, p, k, path):
    """Plot the average of each estimator against sample size."""
    import matplotlib.pyplot as plt

    from . import style

    style.apply_style()
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(rows[0]["truth"], color=style.INK_MUTED, linewidth=1.2, zorder=1)
    ax.plot(ns, [r["unbiased_mean"] for r in rows], color=style.BLUE, marker="o",
            markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.5,
            label="Unbiased estimator", zorder=3)
    ax.plot(ns, [r["naive_mean"] for r in rows], color=style.ORANGE, marker="o",
            markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.5,
            label="Naive estimator", zorder=3)
    ax.annotate(f"True pass@{k} = {rows[0]['truth']:.3f}", xy=(ns[1], rows[0]["truth"]),
                xytext=(0, -14), textcoords="offset points", ha="center",
                color=style.INK_SECONDARY, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Samples per task (n)")
    ax.set_ylabel(f"Average estimate of pass@{k}")
    ax.set_title(f"Naive pass@{k} reads low when samples are few (true pass rate {p:.0%})")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def main():
    p, k = 0.1, 10
    rows = bias_demo(p=p, k=k)
    print(f"Bias demo: true pass rate p={p}, k={k}, true pass@{k} = {rows[0]['truth']:.4f}")
    print(f"{'n':>5} {'unbiased':>10} {'naive':>10} {'naive bias':>12}")
    for r in rows:
        print(f"{r['n']:>5} {r['unbiased_mean']:>10.4f} {r['naive_mean']:>10.4f} "
              f"{r['naive_mean'] - r['truth']:>+12.4f}")
    import os

    os.makedirs("figures", exist_ok=True)
    plot_bias_demo(rows, p, k, "figures/passk_bias.png")
    print("Saved figures/passk_bias.png")


if __name__ == "__main__":
    main()
