import pytest
import torch
import torch.nn.functional as F

from grpo_difficulty.grpo import (
    group_advantages,
    grpo_loss,
    grpo_step,
    nonzero_group_fraction,
    sample_groups,
    train_grpo,
)
from grpo_difficulty.policy import make_policy, pass_probability
from grpo_difficulty.signal import signal_fraction


def test_advantage_is_reward_minus_group_mean():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv, torch.tensor([[0.75, -0.25, -0.25, -0.25], [0.5, 0.5, -0.5, -0.5]]))
    assert torch.allclose(adv.sum(dim=1), torch.zeros(2))


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_all_same_group_gives_zero_loss_and_zero_gradient(value, normalize):
    logits = torch.randn(3, 5, generator=torch.Generator().manual_seed(0), requires_grad=True)
    sampled = torch.randint(0, 5, (3, 8), generator=torch.Generator().manual_seed(1))
    rewards = torch.full((3, 8), value)
    loss = grpo_loss(logits, sampled, rewards, normalize)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(logits.grad == 0)


def test_gradient_matches_hand_calculation():
    """One task, 3 answers, group of 2: one correct sample (token 0), one wrong (token 2)."""
    logits = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
    sampled = torch.tensor([[0, 2]])
    rewards = torch.tensor([[1.0, 0.0]])
    grpo_loss(logits, sampled, rewards).backward()
    # d log p(token) / d logits = onehot(token) - softmax(logits)
    probs = F.softmax(logits.detach(), dim=1)[0]
    adv = torch.tensor([0.5, -0.5])
    expected = torch.zeros(3)
    for a, token in zip(adv, [0, 2]):
        expected += -a * (F.one_hot(torch.tensor(token), 3).float() - probs) / 2
    assert torch.allclose(logits.grad[0], expected, atol=1e-6)
    # the correct answer's logit should go up under gradient descent
    assert logits.grad[0, 0] < 0 and logits.grad[0, 2] > 0


def test_normalized_advantages_have_unit_std_in_mixed_groups():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    adv = group_advantages(rewards, normalize=True)
    assert torch.allclose(adv.std(dim=1, correction=0), torch.ones(2), atol=1e-4)


def test_nonzero_group_fraction_counts_mixed_groups():
    rewards = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert nonzero_group_fraction(rewards) == 0.5


def test_sampled_groups_match_the_theory_curve():
    """Fix the pass rate at p, sample many groups, compare with 1 - p^G - (1-p)^G."""
    p, G, tasks = 0.2, 8, 20000
    logits = torch.log(torch.tensor([[p, 1 - p]])).repeat(tasks, 1)
    answers = torch.zeros(tasks, dtype=torch.long)
    _, rewards = sample_groups(logits, answers, G, torch.Generator().manual_seed(0))
    assert rewards.mean().item() == pytest.approx(p, abs=0.01)
    assert nonzero_group_fraction(rewards) == pytest.approx(float(signal_fraction(p, G)), abs=0.01)


def test_a_step_changes_weights_only_when_there_is_signal():
    policy = make_policy(4, hidden=16, seed=0)
    features = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    answers = torch.zeros(6, dtype=torch.long)
    before = [p.clone() for p in policy.parameters()]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    stats = grpo_step(policy, optimizer, features, answers, 8, torch.Generator().manual_seed(0))
    assert stats["nonzero_fraction"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))


def _toy_problem():
    features = torch.randn(40, 4, generator=torch.Generator().manual_seed(0))
    answers = (features[:, 0] > 0).long() + 2 * (features[:, 1] > 0).long()
    return features, answers


def test_training_raises_the_pass_rate():
    features, answers = _toy_problem()
    policy = make_policy(4, hidden=32, seed=0)
    before = pass_probability(policy, features, answers).mean().item()
    train_grpo(policy, features, answers, steps=300, batch_tasks=16, group_size=8, lr=3e-3, seed=0)
    after = pass_probability(policy, features, answers).mean().item()
    assert after > before + 0.3


def test_training_is_reproducible_and_logs_the_right_amounts():
    features, answers = _toy_problem()
    logs = []
    for _ in range(2):
        policy = make_policy(4, hidden=32, seed=0)
        evaluate = lambda pol: {"pass": pass_probability(pol, features, answers).mean().item()}
        logs.append(train_grpo(policy, features, answers, steps=25, seed=7, eval_fn=evaluate, eval_every=10))
    assert logs[0] == logs[1]
    assert len(logs[0]["steps"]["nonzero_fraction"]) == 25
    assert [e["step"] for e in logs[0]["evals"]] == [0, 10, 20, 25]


def test_different_seeds_give_different_runs():
    features, answers = _toy_problem()
    out = []
    for seed in (0, 1):
        policy = make_policy(4, hidden=32, seed=0)
        out.append(train_grpo(policy, features, answers, steps=10, seed=seed)["steps"]["mean_reward"])
    assert out[0] != out[1]
