# Plan

## The question

Does training on tasks the model solves "sometimes" (20–80% pass rate) teach it more
than training on tasks it always or never solves?

## Approach in one paragraph

Build a small synthetic task family, partially pretrain a small MLP policy on it with
supervised learning, measure each unseen task's pass rate, sort tasks into three bins
(too hard, sweet spot, too easy), then train three GRPO runs from the same checkpoint,
one per bin, with identical compute. Compare them on one shared held-out test set.

## File layout

| File | What it does |
| --- | --- |
| `grpo_difficulty/passk.py` | Unbiased pass@k estimator, the naive estimator, a bootstrap confidence interval & a demo of the naive estimator's bias |
| `grpo_difficulty/signal.py` | The theory: fraction of GRPO groups that carry any gradient, `1 - p^G - (1-p)^G` |
| `grpo_difficulty/world.py` | The synthetic task family & its three disjoint splits (pretrain, RL pool, test) |
| `grpo_difficulty/policy.py` | The MLP policy, supervised pretraining, pass-rate measurement & binning |
| `grpo_difficulty/grpo.py` | GRPO from scratch: sample G answers per task, advantage = reward − group mean |
| `grpo_difficulty/experiment.py` | The three-bin experiment over several seeds, plus ablations |
| `grpo_difficulty/plots.py` | All figures used in the README |
| `run_all.py` | One command that runs everything end to end |
| `tests/` | Unit tests for every module |

The code lives in a package folder (`grpo_difficulty/`) rather than loose files because a
top-level file named `signal.py` would shadow Python's built-in `signal` module & break
other libraries.

## Build order (one branch & pull request per slice)

1. Repo safety files, licence, CI (initial commit on `main`)
2. `passk.py` + tests + bias demo
3. `signal.py` + tests + theory plot
4. `world.py` + tests
5. `policy.py` (MLP, pretraining, pass rates, bins) + tests
6. `grpo.py` + tests
7. `experiment.py` (main experiment, then ablations)
8. `plots.py` + `run_all.py`
9. README with real results, `EXPLAIN.md`, final `BUILDLOG.md`
10. Pre-publish checklist

## Rules I'm holding myself to

- Fixed seeds everywhere; `make all` rebuilds every number & plot.
- The task world & the main experiment settings are chosen *before* looking at which
  bin wins. Settings are never tuned to make the sweet spot look good.
- Whatever the result is, it goes in the README as it came out.
