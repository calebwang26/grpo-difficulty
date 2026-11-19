import pytest
import torch

from grpo_difficulty.policy import (
    assign_bins,
    equal_size_bins,
    greedy_accuracy,
    make_policy,
    pass_probability,
    pretrain,
    sampled_pass_counts,
)
from grpo_difficulty.world import make_world


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def pretrained(world):
    policy = make_policy(world.config.num_answers, seed=0)
    X, y = world.split("pretrain")
    losses = pretrain(policy, X, y, steps=500, seed=0)
    return policy, losses


def test_policy_outputs_one_logit_per_answer(world):
    policy = make_policy(world.config.num_answers)
    X, _ = world.split("test")
    assert policy(X).shape == (len(X), world.config.num_answers)


def test_same_seed_same_policy(world):
    p1, p2 = make_policy(16, seed=3), make_policy(16, seed=3)
    for a, b in zip(p1.parameters(), p2.parameters()):
        assert torch.equal(a, b)


def test_untrained_policy_is_near_chance(world):
    policy = make_policy(world.config.num_answers, seed=0)
    X, y = world.split("test")
    chance = 1 / world.config.num_answers
    assert pass_probability(policy, X, y).mean().item() == pytest.approx(chance, abs=0.04)


def test_pretraining_lowers_loss_and_generalises(world, pretrained):
    policy, losses = pretrained
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20 - 0.3
    X, y = world.split("test")  # never seen in pretraining
    chance = 1 / world.config.num_answers
    assert pass_probability(policy, X, y).mean().item() > 2 * chance


def test_pass_probability_is_a_probability(world, pretrained):
    policy, _ = pretrained
    X, y = world.split("pool")
    p = pass_probability(policy, X, y)
    assert p.shape == (len(X),)
    assert p.min() >= 0 and p.max() <= 1


def test_sampled_counts_agree_with_exact_probability(world, pretrained):
    policy, _ = pretrained
    X, y = world.split("pool")
    generator = torch.Generator().manual_seed(0)
    counts = sampled_pass_counts(policy, X, y, n=2000, generator=generator)
    exact = pass_probability(policy, X, y)
    assert torch.allclose(counts / 2000, exact, atol=0.06)


def test_sampling_is_reproducible(world, pretrained):
    policy, _ = pretrained
    X, y = world.split("pool")
    c1 = sampled_pass_counts(policy, X, y, n=32, generator=torch.Generator().manual_seed(5))
    c2 = sampled_pass_counts(policy, X, y, n=32, generator=torch.Generator().manual_seed(5))
    assert torch.equal(c1, c2)


def test_greedy_accuracy_range(world, pretrained):
    policy, _ = pretrained
    X, y = world.split("test")
    assert 0.0 <= greedy_accuracy(policy, X, y) <= 1.0


def test_assign_bins_edges():
    rates = torch.tensor([0.0, 0.09, 0.10, 0.15, 0.20, 0.5, 0.80, 0.85, 0.90, 0.95, 1.0])
    bins = assign_bins(rates)
    assert bins["hard"].tolist() == [0, 1]
    assert bins["sweet"].tolist() == [4, 5, 6]
    assert bins["easy"].tolist() == [9, 10]


def test_bins_never_overlap():
    rates = torch.rand(500, generator=torch.Generator().manual_seed(0))
    bins = assign_bins(rates)
    seen = set()
    for idx in bins.values():
        assert not seen & set(idx.tolist())
        seen |= set(idx.tolist())


def test_equal_size_bins():
    bins = {"hard": torch.arange(0, 30), "sweet": torch.arange(100, 400), "easy": torch.arange(500, 620)}
    cut = equal_size_bins(bins, seed=0)
    assert {len(v) for v in cut.values()} == {30}
    capped = equal_size_bins(bins, size=10, seed=0)
    assert {len(v) for v in capped.values()} == {10}
    for name in bins:
        assert set(capped[name].tolist()) <= set(bins[name].tolist())
    again = equal_size_bins(bins, size=10, seed=0)
    assert all(torch.equal(capped[k], again[k]) for k in bins)


def test_equal_size_bins_rejects_empty_bin():
    with pytest.raises(ValueError):
        equal_size_bins({"hard": torch.arange(0), "sweet": torch.arange(5), "easy": torch.arange(5)})
