"""GRPO written from scratch (no RL library).

GRPO = Group Relative Policy Optimization. One training step:

1. Pick a batch of tasks.
2. For each task, sample a *group* of G answers from the policy.
3. Score each answer: reward 1 if correct, 0 if wrong.
4. Advantage = reward − the group's mean reward. "Was this answer better or worse
   than my other attempts at the same task?"
5. Loss = −mean(advantage × log-probability of the sampled answer).
   Minimising it raises the probability of answers that beat the group average &
   lowers the probability of answers that fell below it.
6. One Adam step.

If a group is all correct or all wrong, every advantage is 0, the loss is 0 & the
task contributes no gradient. ``signal.py`` has the formula for how often that happens.
"""

import torch
import torch.nn.functional as F

SIGNAL_BUCKETS = 20  # pass-rate buckets (0–5%, 5–10%, ...) used to check the theory


def group_advantages(rewards, normalize=False, eps=1e-6):
    """Advantage of each answer relative to its own group.

    ``rewards`` has shape (tasks, G). With ``normalize=True`` the advantages are also
    divided by the group's standard deviation, which makes every mixed group push
    equally hard no matter how lopsided it is. An all-same group stays exactly zero.
    """
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize:
        advantages = advantages / (rewards.std(dim=1, keepdim=True, correction=0) + eps)
    return advantages


def grpo_loss(logits, sampled, rewards, normalize=False):
    """The GRPO loss for a batch of groups.

    logits:  (tasks, num_answers) policy outputs, one row per task
    sampled: (tasks, G) the answer tokens that were sampled
    rewards: (tasks, G) 1.0 where the sampled answer was correct, else 0.0
    """
    log_probs = F.log_softmax(logits, dim=1)        # log-probability of every answer
    sampled_log_probs = log_probs.gather(1, sampled)  # ... of the answers we sampled
    advantages = group_advantages(rewards, normalize)
    return -(advantages * sampled_log_probs).mean()


def sample_groups(logits, answers, group_size, generator=None):
    """Sample G answers per task from the policy & score them."""
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        sampled = torch.multinomial(probs, group_size, replacement=True, generator=generator)
        rewards = (sampled == answers[:, None]).float()
    return sampled, rewards


def mixed_groups(rewards):
    """True for every group that is mixed, i.e. has at least one nonzero advantage."""
    n_correct = rewards.sum(dim=1)
    return (n_correct > 0) & (n_correct < rewards.shape[1])


def nonzero_group_fraction(rewards):
    """Share of groups that are mixed."""
    return mixed_groups(rewards).float().mean().item()


def grpo_step(policy, optimizer, features, answers, group_size, generator=None, normalize=False):
    """One GRPO update on a batch of tasks. Returns numbers worth logging."""
    logits = policy(features)
    sampled, rewards = sample_groups(logits, answers, group_size, generator)
    loss = grpo_loss(logits, sampled, rewards, normalize)
    with torch.no_grad():
        # each task's true pass rate at the moment its group was sampled
        pass_rate = F.softmax(logits, dim=1).gather(1, answers[:, None])[:, 0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.item(),
        "nonzero_fraction": nonzero_group_fraction(rewards),
        "mean_reward": rewards.mean().item(),
        "pass_rate": pass_rate,
        "mixed": mixed_groups(rewards),
    }


def train_grpo(policy, features, answers, steps, batch_tasks=32, group_size=8, lr=1e-3,
               seed=0, normalize=False, eval_fn=None, eval_every=10):
    """Train ``policy`` with GRPO on the given tasks.

    Every step draws ``batch_tasks`` tasks at random (with replacement) from
    ``features``/``answers`` & samples ``group_size`` answers for each, so every run
    with the same settings uses exactly ``steps × batch_tasks × group_size`` samples.

    ``eval_fn(policy)`` is called before training & then every ``eval_every`` steps;
    it should return a dict of numbers.

    Returns ``{"steps": per-step logs, "evals": evaluation logs, "signal_by_pass_rate":
    counts for checking the theory curve}``. The last item sorts every sampled group
    into one of ``SIGNAL_BUCKETS`` buckets by its task's true pass rate at that moment
    & counts how many groups fell in each bucket & how many of those were mixed.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    step_logs = {"loss": [], "nonzero_fraction": [], "mean_reward": []}
    eval_logs = []
    groups_seen = torch.zeros(SIGNAL_BUCKETS)
    groups_mixed = torch.zeros(SIGNAL_BUCKETS)

    def evaluate(step):
        if eval_fn is not None:
            eval_logs.append({"step": step, **eval_fn(policy)})

    evaluate(0)
    for step in range(1, steps + 1):
        batch = torch.randint(0, len(features), (batch_tasks,), generator=generator)
        stats = grpo_step(policy, optimizer, features[batch], answers[batch],
                          group_size, generator, normalize)
        for key in step_logs:
            step_logs[key].append(stats[key])
        bucket = (stats["pass_rate"] * SIGNAL_BUCKETS).long().clamp(max=SIGNAL_BUCKETS - 1)
        groups_seen += torch.bincount(bucket, minlength=SIGNAL_BUCKETS)
        groups_mixed += torch.bincount(bucket[stats["mixed"]], minlength=SIGNAL_BUCKETS)
        if step % eval_every == 0 or step == steps:
            evaluate(step)
    return {
        "steps": step_logs,
        "evals": eval_logs,
        "signal_by_pass_rate": {"groups": groups_seen.tolist(), "mixed": groups_mixed.tolist()},
    }
