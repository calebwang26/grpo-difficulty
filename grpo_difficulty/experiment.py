"""The experiment: does the difficulty of the training tasks change how much GRPO teaches?

Pipeline:

1. ``prepare``: build the world, pretrain one policy with supervised learning, measure
   every pool task's pass rate with n = 32 samples, sort tasks into bins & cut all
   bins to the same number of tasks.
2. ``select_lr``: choose the GRPO learning rate using only the "mixed" arm (random
   tasks, no difficulty filter) & only the validation split. The three bins & the
   test set play no part in this choice.
3. ``run_experiment``: from the *same* pretrained checkpoint, train one GRPO run per
   arm (several seeds each) with identical settings, so every run uses the same
   number of steps & samples. Evaluate all of them on the same held-out test set.
4. ``run_ablations``: repeat step 3 with other group sizes, learning rates & the
   ÷ std advantage, to see whether the ranking of the arms survives.
5. ``run_replicates``: repeat steps 1 & 3 on freshly shuffled worlds with freshly
   pretrained checkpoints, to see whether the result depends on one lucky checkpoint.

The four main arms are the three bins plus "mixed", a random draw from the whole
pool: the "buy tasks without looking at difficulty" baseline.

Two follow-up arms were added *after* seeing the main result, to ask which filter
is worth applying: "no_hard" (a random draw with the too-hard bin removed) &
"no_easy" (a random draw with the too-easy bin removed).

Run ``python -m grpo_difficulty.experiment`` to produce ``outputs/results.json``.
"""

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace

import torch

from .grpo import train_grpo
from .passk import bootstrap_ci, mean_pass_at_k, pass_at_k_naive
from .policy import (
    BIN_NAMES,
    assign_bins,
    equal_size_bins,
    greedy_accuracy,
    make_policy,
    pass_probability,
    pretrain,
    sampled_pass_counts,
)
from .signal import signal_fraction
from .world import WorldConfig, make_world

ARMS = ("hard", "sweet", "easy", "mixed")
FOLLOWUP_ARMS = ("no_hard", "no_easy")


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    # pretraining
    policy_seed: int = 0
    pretrain_steps: int = 2000
    pretrain_seed: int = 0
    # measuring pass rates & building bins
    n_measure: int = 32
    measure_seed: int = 0
    tasks_per_bin: int = 1500
    bin_seed: int = 0
    # GRPO (identical for every arm: this is the "equal compute" part)
    steps: int = 3000
    batch_tasks: int = 32
    group_size: int = 8
    lr: float = 1e-3          # replaced by the winner of select_lr in the full run
    normalize: bool = False
    seeds: tuple = (0, 1, 2, 3, 4)
    eval_every: int = 25
    # learning-rate selection (mixed arm, validation split)
    lr_candidates: tuple = (1e-4, 3e-4, 1e-3, 3e-3)
    lr_seeds: tuple = (100, 101, 102)


@dataclass
class Setup:
    """Everything that is fixed before any GRPO run starts."""

    config: ExperimentConfig
    world: object
    pretrained_state: dict        # weights of the pretrained checkpoint
    pool_counts: torch.Tensor     # correct answers out of n_measure, per pool task
    pool_exact: torch.Tensor      # true pass rate per pool task
    arm_tasks: dict               # arm name -> positions inside the pool
    bin_sizes_before_cut: dict
    test_groups: dict             # test tasks grouped by their starting difficulty


def prepare(config=ExperimentConfig()):
    """World, pretrained checkpoint, measured pass rates & equal-size bins."""
    world = make_world(config.world)
    policy = make_policy(config.world.num_answers, seed=config.policy_seed)
    pretrain(policy, *world.split("pretrain"), steps=config.pretrain_steps,
             seed=config.pretrain_seed)

    pool_X, pool_y = world.split("pool")
    generator = torch.Generator().manual_seed(config.measure_seed)
    counts = sampled_pass_counts(policy, pool_X, pool_y, n=config.n_measure, generator=generator)
    bins = assign_bins(counts / config.n_measure)
    arm_tasks = equal_size_bins(bins, size=config.tasks_per_bin, seed=config.bin_seed)
    size = len(arm_tasks["hard"])

    # The "mixed" arm: the same number of tasks, drawn from the whole pool at random.
    generator = torch.Generator().manual_seed(config.bin_seed + 1000)
    arm_tasks["mixed"] = torch.randperm(len(pool_X), generator=generator)[:size].sort().values

    # Follow-up arms: a random draw from the pool with one extreme bin removed.
    for arm, dropped in (("no_hard", "hard"), ("no_easy", "easy")):
        keep = torch.ones(len(pool_X), dtype=torch.bool)
        keep[bins[dropped]] = False
        candidates = torch.nonzero(keep)[:, 0]
        pick = torch.randperm(len(candidates), generator=generator)[:size]
        arm_tasks[arm] = candidates[pick].sort().values

    # Group the test tasks by how hard they are for the pretrained policy, so we can
    # later see *which* test tasks each run improved. Uses the true pass rate.
    test_X, test_y = world.split("test")
    test_groups = assign_bins(pass_probability(policy, test_X, test_y))

    return Setup(
        config=config,
        world=world,
        pretrained_state=copy.deepcopy(policy.state_dict()),
        pool_counts=counts,
        pool_exact=pass_probability(policy, pool_X, pool_y),
        arm_tasks=arm_tasks,
        bin_sizes_before_cut={name: len(idx) for name, idx in bins.items()},
        test_groups=test_groups,
    )


def fresh_policy(setup):
    """A new copy of the pretrained checkpoint. Every run starts from this."""
    policy = make_policy(setup.config.world.num_answers, seed=0)
    policy.load_state_dict(setup.pretrained_state)
    return policy


def run_arm(setup, arm, seed, config=None):
    """Train one GRPO run on one arm's tasks & log held-out results along the way."""
    config = config or setup.config
    pool_X, pool_y = setup.world.split("pool")
    test_X, test_y = setup.world.split("test")
    val_X, val_y = setup.world.split("val")
    train_X, train_y = pool_X[setup.arm_tasks[arm]], pool_y[setup.arm_tasks[arm]]

    def evaluate(policy):
        test_p = pass_probability(policy, test_X, test_y)
        train_p = pass_probability(policy, train_X, train_y)
        out = {
            "test_pass": test_p.mean().item(),
            "test_greedy": greedy_accuracy(policy, test_X, test_y),
            "val_pass": pass_probability(policy, val_X, val_y).mean().item(),
            "train_pass": train_p.mean().item(),
            # what signal.py predicts for the next step, given each task's pass rate now
            "predicted_signal": float(signal_fraction(train_p.numpy(), config.group_size).mean()),
        }
        for name in BIN_NAMES:
            out[f"test_pass_{name}"] = test_p[setup.test_groups[name]].mean().item()
        return out

    policy = fresh_policy(setup)
    return train_grpo(policy, train_X, train_y, steps=config.steps,
                      batch_tasks=config.batch_tasks, group_size=config.group_size,
                      lr=config.lr, seed=seed, normalize=config.normalize,
                      eval_fn=evaluate, eval_every=config.eval_every)


def describe_setup(setup):
    """A small summary of the fixed setup, for the results file & the README."""
    n = setup.config.n_measure
    policy = fresh_policy(setup)
    summary = {}
    for name in ("pretrain", "pool", "val", "test"):
        X, y = setup.world.split(name)
        summary[f"{name}_pass"] = pass_probability(policy, X, y).mean().item()
        summary[f"{name}_greedy"] = greedy_accuracy(policy, X, y)
    summary.update({
        "bin_sizes_before_cut": setup.bin_sizes_before_cut,
        "tasks_per_arm": len(setup.arm_tasks["hard"]),
        "pool_tasks_at_zero_correct": int((setup.pool_counts == 0).sum()),
        "test_group_sizes": {k: len(v) for k, v in setup.test_groups.items()},
        "pool_counts": setup.pool_counts.tolist(),
        "arms": {},
    })
    # Score the pretrained policy on the pool the way a benchmark would, with pass@k.
    # For this toy policy we also know the true value, so we can check the estimators.
    counts = setup.pool_counts.tolist()
    summary["pool_pass_at_k"] = {}
    for k in (1, 8):
        point, low, high = bootstrap_ci(n, counts, k)
        summary["pool_pass_at_k"][str(k)] = {
            "unbiased": point, "ci_low": low, "ci_high": high,
            "naive": mean_pass_at_k(n, counts, k, estimator=pass_at_k_naive),
            "truth": (1 - (1 - setup.pool_exact) ** k).mean().item(),
        }
    for arm, idx in setup.arm_tasks.items():
        exact = setup.pool_exact[idx]
        summary["arms"][arm] = {
            "measured_pass": (setup.pool_counts[idx].float() / n).mean().item(),
            "true_pass": exact.mean().item(),
            "true_pass_median": exact.median().item(),
            "predicted_signal_at_start": float(
                signal_fraction(exact.numpy(), setup.config.group_size).mean()),
        }
    return summary


def select_lr(setup, verbose=True):
    """Choose the learning rate on the mixed arm & the validation split only.

    Returns ``(best_lr, table)`` where the table has one row per candidate.
    """
    config = setup.config
    table = []
    for lr in config.lr_candidates:
        finals = []
        for seed in config.lr_seeds:
            log = run_arm(setup, "mixed", seed, replace(config, lr=lr))
            finals.append(log["evals"][-1]["val_pass"])
        row = {"lr": lr, "val_pass_start": log["evals"][0]["val_pass"],
               "val_pass_final_mean": sum(finals) / len(finals), "val_pass_final": finals}
        table.append(row)
        if verbose:
            print(f"  lr {lr:g}: validation pass rate {row['val_pass_start']:.3f} -> "
                  f"{row['val_pass_final_mean']:.3f}", flush=True)
    best = max(table, key=lambda row: row["val_pass_final_mean"])
    return best["lr"], table


def run_experiment(setup, config=None, arms=ARMS, seeds=None, verbose=True, compact=False):
    """Every arm × every seed, all from one checkpoint, all with the same settings.

    With ``compact=True`` only the test curve of each run is kept (for ablations).
    """
    config = config or setup.config
    seeds = seeds if seeds is not None else config.seeds
    runs = {}
    for arm in arms:
        runs[arm] = []
        for seed in seeds:
            start = time.time()
            log = run_arm(setup, arm, seed, config)
            if compact:
                log = {"evals": [{"step": e["step"], "test_pass": e["test_pass"]}
                                 for e in log["evals"]]}
            runs[arm].append({"seed": seed, **log})
            if verbose:
                first, last = log["evals"][0], log["evals"][-1]
                print(f"  {arm:>5} seed {seed}: test pass rate {first['test_pass']:.3f} -> "
                      f"{last['test_pass']:.3f}  ({time.time() - start:.1f}s)", flush=True)
    return runs


def ablation_variants(config):
    """Settings to vary one at a time. Samples per step stay at batch_tasks × group_size."""
    samples = config.batch_tasks * config.group_size
    variants = {}
    for G in (4, 16):
        variants[f"G={G}"] = replace(config, group_size=G, batch_tasks=samples // G)
    variants["÷ std"] = replace(config, normalize=True)
    for lr in config.lr_candidates:
        if lr != config.lr:
            variants[f"lr={lr:g}"] = replace(config, lr=lr)
    return variants


def run_ablations(setup, config, seeds=(0, 1, 2), verbose=True):
    """Re-run the four main arms under each variant. Returns {variant name: runs}."""
    out = {}
    for name, variant in ablation_variants(config).items():
        if verbose:
            print(f" Ablation {name}", flush=True)
        out[name] = {
            "config": {"group_size": variant.group_size, "batch_tasks": variant.batch_tasks,
                       "lr": variant.lr, "normalize": variant.normalize},
            "runs": run_experiment(setup, variant, seeds=seeds, verbose=verbose, compact=True),
        }
    return out


def run_replicates(config, world_seeds=(1, 2, 3, 4), seeds=(0, 1, 2), verbose=True):
    """Redo everything on reshuffled worlds with freshly pretrained checkpoints."""
    out = []
    for s in world_seeds:
        if verbose:
            print(f" Replicate with world & pretraining seed {s}", flush=True)
        rep_config = replace(config, world=replace(config.world, seed=s), policy_seed=s,
                             pretrain_seed=s, measure_seed=s, bin_seed=s)
        setup = prepare(rep_config)
        summary = describe_setup(setup)
        summary.pop("pool_counts")
        out.append({
            "world_seed": s,
            "setup": summary,
            "runs": run_experiment(setup, rep_config, seeds=seeds, verbose=verbose, compact=True),
        })
    return out


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f)


def run_pipeline(config=ExperimentConfig(), ablation_seeds=(0, 1, 2),
                 replicate_world_seeds=(1, 2, 3, 4), replicate_seeds=(0, 1, 2), verbose=True):
    """Run every stage in order & return one results dictionary."""
    say = print if verbose else (lambda *args, **kwargs: None)

    say("Step 1: world, pretraining, pass rates, bins")
    setup = prepare(config)
    summary = describe_setup(setup)
    say(f"  pass rate after pretraining: pretrain {summary['pretrain_pass']:.3f}, "
        f"pool {summary['pool_pass']:.3f}, test {summary['test_pass']:.3f}")
    say(f"  bin sizes: {summary['bin_sizes_before_cut']}, cut to {summary['tasks_per_arm']} each")

    say("Step 2: choose the learning rate (mixed arm, validation split)")
    best_lr, lr_table = select_lr(setup, verbose)
    config = replace(config, lr=best_lr)
    setup.config = config
    say(f"  chosen learning rate: {best_lr:g}")

    say("Step 3: main experiment")
    runs = run_experiment(setup, config, verbose=verbose)

    say("Step 4: follow-up arms (random mix with one extreme bin removed)")
    runs.update(run_experiment(setup, config, arms=FOLLOWUP_ARMS, verbose=verbose))

    say("Step 5: ablations")
    ablations = run_ablations(setup, config, seeds=ablation_seeds, verbose=verbose)

    say("Step 6: replicates on reshuffled worlds")
    replicates = run_replicates(config, replicate_world_seeds, replicate_seeds, verbose)

    return {
        "config": asdict(config),
        "setup": summary,
        "lr_selection": {"chosen": best_lr, "table": lr_table},
        "runs": runs,
        "ablations": ablations,
        "replicates": replicates,
    }


def main(path="outputs/results.json"):
    torch.set_num_threads(1)  # tiny networks run fastest, & most repeatably, on one thread
    start = time.time()
    results = run_pipeline()
    save_results(results, path)
    print(f"Saved {path}  (total {time.time() - start:.0f}s)")
    return results


if __name__ == "__main__":
    main()
