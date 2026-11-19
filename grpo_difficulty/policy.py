"""The policy: a small MLP that outputs a softmax over answer tokens.

This file also holds everything that happens *before* RL training:

* ``pretrain``: supervised learning with cross-entropy, stopped early on purpose
  so the policy is only partly competent.
* ``pass_probability`` / ``sampled_pass_counts``: how often the policy solves a task.
* ``assign_bins`` / ``equal_size_bins``: sort tasks into too hard, sweet spot & too easy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bin edges from the project spec, as pass rates.
HARD_BELOW = 0.10
SWEET_FROM, SWEET_TO = 0.20, 0.80
EASY_ABOVE = 0.90
BIN_NAMES = ("hard", "sweet", "easy")


class Policy(nn.Module):
    """Two hidden layers with ReLU. Input: 4 task features. Output: one logit per answer."""

    def __init__(self, num_answers, hidden=128, num_features=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_answers),
        )

    def forward(self, features):
        return self.net(features)  # logits; softmax turns them into probabilities


def make_policy(num_answers, hidden=128, seed=0):
    """Create a policy with a fixed random initialisation."""
    torch.manual_seed(seed)
    return Policy(num_answers, hidden)


def pretrain(policy, features, answers, steps=2000, batch_size=64, lr=2e-3, seed=0):
    """Supervised pretraining: show the policy tasks with their correct answers.

    The loss is cross-entropy, ``-log(probability the policy gives the right answer)``.
    We stop after a limited number of steps so the policy ends up solving some tasks
    reliably, some sometimes & some never.

    Returns the list of loss values, one per step.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        batch = torch.randint(0, len(features), (batch_size,), generator=generator)
        loss = F.cross_entropy(policy(features[batch]), answers[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def pass_probability(policy, features, answers):
    """Exact pass rate per task: the probability the policy assigns to the right answer.

    Because the policy is a plain softmax we can read this number off directly.
    A real coding model can't do that, which is why real evaluations sample instead.
    """
    probs = F.softmax(policy(features), dim=1)
    return probs.gather(1, answers[:, None])[:, 0]


@torch.no_grad()
def greedy_accuracy(policy, features, answers):
    """Share of tasks where the policy's single most likely answer is correct."""
    return (policy(features).argmax(dim=1) == answers).float().mean().item()


@torch.no_grad()
def sampled_pass_counts(policy, features, answers, n=32, generator=None):
    """Measure pass rates the way a real evaluation does: sample n answers per task.

    Returns the number of correct answers out of n for every task.
    """
    probs = F.softmax(policy(features), dim=1)
    samples = torch.multinomial(probs, n, replacement=True, generator=generator)
    return (samples == answers[:, None]).sum(dim=1)


def assign_bins(pass_rates):
    """Sort tasks into bins by pass rate. Returns {bin name: tensor of task positions}.

    Tasks between the bins (10–20% & 80–90%) are left out on purpose, so the three
    bins are clearly separated.
    """
    pass_rates = torch.as_tensor(pass_rates, dtype=torch.float)
    masks = {
        "hard": pass_rates < HARD_BELOW,
        "sweet": (pass_rates >= SWEET_FROM) & (pass_rates <= SWEET_TO),
        "easy": pass_rates > EASY_ABOVE,
    }
    return {name: torch.nonzero(mask)[:, 0] for name, mask in masks.items()}


def equal_size_bins(bins, size=None, seed=0):
    """Cut every bin down to the same number of tasks, chosen at random.

    Without this, a bigger bin would also be a more varied training set, & we could
    not tell "better difficulty" apart from "more tasks".
    """
    smallest = min(len(idx) for idx in bins.values())
    if size is None or size > smallest:
        size = smallest
    if size == 0:
        raise ValueError(f"a bin is empty: { {k: len(v) for k, v in bins.items()} }")
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name in sorted(bins):
        idx = bins[name]
        keep = torch.randperm(len(idx), generator=generator)[:size]
        out[name] = idx[keep].sort().values
    return out


def plot_pass_rate_histogram(counts, n, path):
    """Bar chart of how many pool tasks landed on each measured pass rate."""
    import matplotlib.pyplot as plt

    from . import style

    style.apply_style()
    rates = torch.arange(n + 1) / n
    tallies = torch.bincount(counts, minlength=n + 1)
    colors = []
    for r in rates.tolist():
        if r < HARD_BELOW:
            colors.append(style.BIN_COLORS["hard"])
        elif SWEET_FROM <= r <= SWEET_TO:
            colors.append(style.BIN_COLORS["sweet"])
        elif r > EASY_ABOVE:
            colors.append(style.BIN_COLORS["easy"])
        else:
            colors.append(style.AXIS)  # the gaps between bins, not used for training

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(rates.tolist(), tallies.tolist(), width=0.72 / n, color=colors, zorder=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.BIN_COLORS[k]) for k in BIN_NAMES]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=style.AXIS))
    labels = [style.BIN_LABELS[k] for k in BIN_NAMES] + ["Between bins (not used)"]
    ax.legend(handles, labels, loc="upper center", ncol=2)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xticklabels([f"{i * 10}%" for i in range(11)])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, max(tallies.tolist()) * 1.35)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(f"Measured pass rate (correct answers out of {n} samples)")
    ax.set_ylabel("Number of pool tasks")
    ax.set_title("After partial pretraining, unseen tasks span every difficulty")
    fig.savefig(path)
    plt.close(fig)


def main():
    import os

    from .world import make_world

    world = make_world()
    policy = make_policy(world.config.num_answers, seed=0)
    pretrain(policy, *world.split("pretrain"), seed=0)
    for name in ("pretrain", "pool", "test"):
        X, y = world.split(name)
        print(f"{name:>9}: mean pass rate {pass_probability(policy, X, y).mean():.3f}, "
              f"greedy accuracy {greedy_accuracy(policy, X, y):.3f}")

    X, y = world.split("pool")
    n = 32
    counts = sampled_pass_counts(policy, X, y, n=n, generator=torch.Generator().manual_seed(0))
    exact = pass_probability(policy, X, y)
    bins = assign_bins(counts / n)
    print(f"\nPool tasks binned by measured pass rate (n = {n} samples each):")
    for name in BIN_NAMES:
        idx = bins[name]
        print(f"  {name:>5}: {len(idx):4d} tasks, true mean pass rate {exact[idx].mean():.3f}")
    os.makedirs("figures", exist_ok=True)
    plot_pass_rate_histogram(counts, n, "figures/pass_rate_hist.png")
    print("Saved figures/pass_rate_hist.png")


if __name__ == "__main__":
    main()
