import json
import os

import pytest
import torch

from grpo_difficulty import plots
from grpo_difficulty.experiment import (
    ARMS,
    FOLLOWUP_ARMS,
    ExperimentConfig,
    ablation_variants,
    prepare,
    run_arm,
    run_pipeline,
    select_lr,
)
from grpo_difficulty.policy import BIN_NAMES, assign_bins
from grpo_difficulty.world import WorldConfig

# A miniature version of the real experiment, small enough to run in seconds.
TINY = ExperimentConfig(
    world=WorldConfig(grid=12, n_pretrain=300, n_pool=1200, n_val=200, n_test=300),
    pretrain_steps=400,
    tasks_per_bin=40,
    steps=40,
    eval_every=20,
    seeds=(0, 1),
    lr_candidates=(1e-3, 3e-3),
    lr_seeds=(100,),
)


@pytest.fixture(scope="module")
def setup():
    return prepare(TINY)


def test_all_arms_have_the_same_number_of_tasks(setup):
    sizes = {arm: len(idx) for arm, idx in setup.arm_tasks.items()}
    assert set(sizes) == set(ARMS) | set(FOLLOWUP_ARMS)
    assert len(set(sizes.values())) == 1


def test_bin_arms_hold_tasks_from_the_right_bin_only(setup):
    measured = setup.pool_counts / TINY.n_measure
    bins = assign_bins(measured)
    for name in BIN_NAMES:
        assert set(setup.arm_tasks[name].tolist()) <= set(bins[name].tolist())
    hard, sweet, easy = (set(setup.arm_tasks[n].tolist()) for n in BIN_NAMES)
    assert not hard & sweet and not sweet & easy and not hard & easy
    assert not set(setup.arm_tasks["no_hard"].tolist()) & set(bins["hard"].tolist())
    assert not set(setup.arm_tasks["no_easy"].tolist()) & set(bins["easy"].tolist())


def test_every_run_starts_from_the_same_checkpoint_and_uses_equal_compute(setup):
    logs = {arm: run_arm(setup, arm, seed=0) for arm in ARMS}
    starts = {round(log["evals"][0]["test_pass"], 6) for log in logs.values()}
    assert len(starts) == 1
    for log in logs.values():
        assert len(log["steps"]["nonzero_fraction"]) == TINY.steps
        assert sum(log["signal_by_pass_rate"]["groups"]) == TINY.steps * TINY.batch_tasks


def test_run_is_reproducible(setup):
    assert run_arm(setup, "sweet", seed=3) == run_arm(setup, "sweet", seed=3)


def test_sweet_arm_starts_with_more_signal_than_the_extremes(setup):
    first = {arm: run_arm(setup, arm, seed=0)["evals"][0]["predicted_signal"] for arm in BIN_NAMES}
    assert first["sweet"] > first["hard"]
    assert first["sweet"] > first["easy"]


def test_select_lr_picks_a_candidate(setup):
    best, table = select_lr(setup, verbose=False)
    assert best in TINY.lr_candidates
    assert [row["lr"] for row in table] == list(TINY.lr_candidates)


def test_ablations_keep_samples_per_step_constant():
    samples = TINY.batch_tasks * TINY.group_size
    for variant in ablation_variants(TINY).values():
        assert variant.batch_tasks * variant.group_size == samples


def test_whole_pipeline_runs_and_makes_figures(tmp_path, monkeypatch):
    results = run_pipeline(TINY, ablation_seeds=(0,), replicate_world_seeds=(1,),
                           replicate_seeds=(0,), verbose=False)
    json.dumps(results)  # must be plain data that can be saved
    assert set(results["runs"]) == set(ARMS) | set(FOLLOWUP_ARMS)
    assert len(results["replicates"]) == 1

    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs")
    with open("outputs/results.json", "w") as f:
        json.dump(results, f)
    plots.main()
    for name in ("heldout_accuracy.png", "signal_fraction.png", "gain_by_difficulty.png"):
        assert os.path.getsize(os.path.join("figures", name)) > 5000
    assert "Main result" in open("outputs/summary.md").read()
