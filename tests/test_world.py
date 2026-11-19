import pytest
import torch

from grpo_difficulty.world import WorldConfig, encode, make_world, rule


def test_default_rule_by_hand():
    # defaults: stripe width 2, 16 answers. 3*15 + 3*15 = 90; 90 // 2 = 45; 45 mod 16 = 13
    c = WorldConfig()
    assert rule(3, 3, 15, 15, c.stripe_width, c.num_answers) == 13
    assert rule(1, 2, 3, 4, c.stripe_width, c.num_answers) == 5  # 11 // 2 = 5


def test_rule_by_hand():
    # (a*x + b*y) = 2*5 + 1*3 = 13; 13 // 4 = 3; 3 mod 8 = 3
    assert rule(2, 1, 5, 3, stripe_width=4, num_answers=8) == 3
    # 3*15 + 3*15 = 90; 90 // 4 = 22; 22 mod 8 = 6
    assert rule(3, 3, 15, 15, stripe_width=4, num_answers=8) == 6
    assert rule(1, 0, 0, 9, stripe_width=4, num_answers=8) == 0


def test_world_sizes_and_ranges():
    world = make_world()
    c = world.config
    assert len(world.tasks) == ((c.max_coef + 1) ** 2 - 1) * c.grid**2
    assert len(world.pretrain_idx) == c.n_pretrain
    assert len(world.pool_idx) == c.n_pool
    assert len(world.test_idx) == c.n_test
    assert world.answers.min() >= 0 and world.answers.max() < c.num_answers
    assert world.features.min() >= -1 and world.features.max() <= 1


def test_splits_never_overlap():
    world = make_world()
    as_set = lambda idx: {tuple(t) for t in world.tasks[idx].tolist()}
    pretrain, pool, test = as_set(world.pretrain_idx), as_set(world.pool_idx), as_set(world.test_idx)
    assert not pretrain & pool
    assert not pretrain & test
    assert not pool & test
    # every task is unique, so no task can hide in two splits under another index
    assert len({tuple(t) for t in world.tasks.tolist()}) == len(world.tasks)


def test_answers_follow_the_rule():
    world = make_world()
    c = world.config
    for (a, b, x, y), answer in zip(world.tasks[:200].tolist(), world.answers[:200].tolist()):
        assert answer == ((a * x + b * y) // c.stripe_width) % c.num_answers


def test_same_seed_same_world_and_different_seed_differs():
    w1, w2 = make_world(WorldConfig(seed=0)), make_world(WorldConfig(seed=0))
    assert torch.equal(w1.tasks, w2.tasks)
    w3 = make_world(WorldConfig(seed=1))
    assert not torch.equal(w1.tasks, w3.tasks)


def test_no_degenerate_rule():
    world = make_world()
    assert ((world.tasks[:, 0] == 0) & (world.tasks[:, 1] == 0)).sum() == 0


def test_encode_maps_corners_to_plus_minus_one():
    c = WorldConfig()
    corners = torch.tensor([[0, 0, 0, 0], [c.max_coef, c.max_coef, c.grid - 1, c.grid - 1]])
    features = encode(corners, c)
    assert torch.allclose(features[0], -torch.ones(4))
    assert torch.allclose(features[1], torch.ones(4))


def test_split_returns_matching_features_and_answers():
    world = make_world()
    X, y = world.split("test")
    assert X.shape == (world.config.n_test, 4)
    assert y.shape == (world.config.n_test,)


def test_too_many_requested_tasks_raises():
    with pytest.raises(ValueError):
        make_world(WorldConfig(n_pool=100000))
