"""The task world: a small synthetic family of tasks with one hidden rule.

One task is four small integers ``(a, b, x, y)`` & its answer is one of
``num_answers`` tokens:

    answer = ((a*x + b*y) // stripe_width) mod num_answers

Picture a 16-by-16 grid of points ``(x, y)``. The rule paints diagonal stripes
across the grid & the answer is the stripe a point falls in. The pair ``(a, b)``
is the task's rule setting: it turns the stripes & sets how thin they are.
Small ``a`` & ``b`` give wide stripes (easy to learn), large ones give thin
stripes (hard to learn), & points next to a stripe edge are harder than points in
the middle of a stripe.

All four numbers are given to the policy as input features, so what it learns on
one task carries over to nearby tasks.

The world is cut into three groups of tasks that never overlap:

* ``pretrain``: used for supervised pretraining (the policy sees the answers).
* ``pool``: the tasks we could "buy" for RL training. Never seen in pretraining.
* ``test``: held out for evaluation only. Never trained on in any way.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorldConfig:
    num_answers: int = 8    # how many answer tokens (stripe colours)
    stripe_width: int = 4   # how wide a stripe is, in units of a*x + b*y
    max_coef: int = 3       # a & b each run from 0 to this
    grid: int = 16          # x & y each run from 0 to grid - 1
    n_pretrain: int = 800
    n_pool: int = 1500
    n_test: int = 800
    seed: int = 0


@dataclass(frozen=True)
class World:
    config: WorldConfig
    tasks: torch.Tensor      # (N, 4) integers: a, b, x, y
    features: torch.Tensor   # (N, 4) floats in [-1, 1], the policy's input
    answers: torch.Tensor    # (N,) integers in [0, num_answers)
    pretrain_idx: torch.Tensor
    pool_idx: torch.Tensor
    test_idx: torch.Tensor

    def split(self, name):
        """Return (features, answers) for 'pretrain', 'pool' or 'test'."""
        idx = {"pretrain": self.pretrain_idx, "pool": self.pool_idx, "test": self.test_idx}[name]
        return self.features[idx], self.answers[idx]


def rule(a, b, x, y, stripe_width, num_answers):
    """The hidden rule. Works on plain integers or on tensors."""
    return ((a * x + b * y) // stripe_width) % num_answers


def encode(tasks, config):
    """Turn integer tasks into input features scaled to the range [-1, 1]."""
    tasks = tasks.float()
    coefs = tasks[:, :2] / config.max_coef * 2 - 1
    points = tasks[:, 2:] / (config.grid - 1) * 2 - 1
    return torch.cat([coefs, points], dim=1)


def make_world(config=WorldConfig()):
    """Build every task, shuffle with a fixed seed & cut into the three splits."""
    coef_values = range(config.max_coef + 1)
    grid_values = range(config.grid)
    all_tasks = [
        (a, b, x, y)
        for a in coef_values
        for b in coef_values
        if (a, b) != (0, 0)  # a = b = 0 makes every answer 0, which is no task at all
        for x in grid_values
        for y in grid_values
    ]
    tasks = torch.tensor(all_tasks)
    needed = config.n_pretrain + config.n_pool + config.n_test
    if needed > len(tasks):
        raise ValueError(f"splits need {needed} tasks but the world only has {len(tasks)}")

    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(tasks), generator=generator)
    tasks = tasks[order]

    answers = rule(tasks[:, 0], tasks[:, 1], tasks[:, 2], tasks[:, 3],
                   config.stripe_width, config.num_answers)
    features = encode(tasks, config)

    a, b = config.n_pretrain, config.n_pretrain + config.n_pool
    return World(
        config=config,
        tasks=tasks,
        features=features,
        answers=answers,
        pretrain_idx=torch.arange(0, a),
        pool_idx=torch.arange(a, b),
        test_idx=torch.arange(b, b + config.n_test),
    )


def main():
    world = make_world()
    c = world.config
    print(f"World: {len(world.tasks)} tasks in total, {c.num_answers} answer tokens")
    print(f"Splits: pretrain {len(world.pretrain_idx)}, pool {len(world.pool_idx)}, "
          f"test {len(world.test_idx)}")
    print("First five tasks (a, b, x, y) -> answer:")
    for task, answer in zip(world.tasks[:5].tolist(), world.answers[:5].tolist()):
        print(f"  {tuple(task)} -> {answer}")
    counts = torch.bincount(world.answers, minlength=c.num_answers).tolist()
    print(f"How often each answer token occurs: {counts}")


if __name__ == "__main__":
    main()
