# Build log

A running record of what was built, in order, with the decisions made & the problems hit.
Newest entries are at the bottom.

## Step 0: environment & repo setup

**Built:** a dedicated conda environment (`grpo-difficulty`, Python 3.11), the git repo,
`.gitignore`, the gitleaks config & pre-commit hook, the CI workflow, an MIT licence,
a `Makefile` & pinned `requirements.txt`.

**Decisions**

- *Package folder instead of loose files.* The spec names a file `signal.py`. A loose file
  with that name at the top of the repo would shadow Python's built-in `signal` module,
  which pytest & other libraries import. So all code lives in `grpo_difficulty/` & is
  run as `python -m grpo_difficulty.signal`. The file names from the spec are unchanged.
- *Pre-commit hook as a plain script.* `scripts/pre-commit` is versioned in the repo &
  copied into `.git/hooks/` by `make hooks`. It runs `gitleaks protect --staged`.
- *CI installs CPU-only PyTorch* so the GitHub runner doesn't download GPU libraries.

**Problems hit**

- `conda create` refused to run because the default Anaconda channels need their Terms of
  Service accepted. Accepting terms is Caleb's call, not mine, so I built the environment
  from the community `conda-forge` channel instead
  (`conda create -n grpo-difficulty -c conda-forge --override-channels python=3.11`).
  No global conda setting was changed.

**Checked:** the secret scanner blocks an AWS-shaped key without the `FAKE` marker &
lets the same key through when it carries the marker.

## Step 1: `passk.py` (pass@k estimators)

**Built:** the unbiased pass@k estimator from the Codex paper, the naive plug-in
estimator, a bootstrap confidence interval over tasks & a simulation that shows the
naive estimator's bias. Also `style.py`, one shared look for all figures.

**Decisions**

- *Stable computation.* `C(n-c, k) / C(n, k)` is computed as the product
  `prod (1 - k/i)` for `i` from `n-c+1` to `n`, so no huge binomial numbers are ever
  formed. A test runs it with n = 100,000.
- *The bootstrap resamples tasks,* because a benchmark is a sample of tasks & that is
  the main source of uncertainty in a benchmark score.
- *The unbiasedness test is exact, not simulated.* It sums the estimator over every
  possible number of correct answers, weighted by the binomial probability, & checks the
  total equals `1 - (1-p)^k` to 12 decimal places.

**Result of the demo** (true pass rate 10%, k = 10, true pass@10 = 0.651): with n = 10
samples the naive estimator averages 0.494, a 24% relative under-read. The unbiased
estimator averages 0.652.

**Problems hit:** the first plot had a title wider than the figure & a label sitting
on top of a line. Shortened the title & moved the label.

**CI problem on the first pull request:** the gitleaks check failed with
`403 Resource not accessible by integration`. It was not a secret finding. The gitleaks
action lists the pull request's commits, & the workflow's token had no permission to
read pull requests. Fixed by adding an explicit `permissions` block (`contents: read`,
`pull-requests: read`) to `ci.yml`.

## Step 2: `signal.py` (the theory)

**Built:** `signal_fraction(p, G) = 1 - p^G - (1-p)^G`, the chance that a GRPO group is
mixed & therefore has a nonzero gradient. Also `mean_squared_advantage`, which gives
the average *size* of the advantages (`p(1-p)(G-1)/G`), a brute-force simulation that
checks the formula, & the plot for G = 4, 8 & 16.

**Decisions**

- *The three G lines share one colour, light to dark,* because group size is an ordered
  quantity, not three unrelated categories.
- *The difficulty bins are shaded on the plot* so the theory & the experiment use the
  same picture.

**Worth noticing already:** the "too hard" bin is not a dead zone by this formula. At
p = 10% & G = 8, 57% of groups still carry signal. The signal only truly vanishes for
tasks far below 10% (p = 1% gives 8%). So the theory predicts a real cost for hard
tasks only when they are *very* hard, & a similar cost for easy tasks only when they are
*very* easy.

## Step 3: `world.py` (the task world)

**Built:** a family of 3,840 tasks. Each task is four small integers `(a, b, x, y)` &
the answer is `((a*x + b*y) // 4) mod 8`, one of 8 tokens. Think of diagonal stripes
painted over a 16×16 grid: `(a, b)` sets the stripes' direction & thickness, `(x, y)` is
a point, & the answer is the stripe the point is in. The tasks are shuffled with a fixed
seed & cut into three non-overlapping splits: 800 for pretraining, 1,500 as the RL
"pool" & 800 held out for testing.

**This was the hardest design decision in the project, & the first three attempts
failed.** The spec suggested `(a*x + b) mod m`. I prototyped that before writing any repo
code, measuring one thing only: after partial pretraining, do *unseen* tasks show a wide
spread of pass rates?

| Attempt | Input features | What happened on unseen tasks |
| --- | --- | --- |
| 1. `(a*x + b) mod m`, m = 11–17 | one-hot | The network memorised its training tasks (100% there) but stayed near chance on unseen ones. Only the trivial `a = 0` rule carried over. Of 444 pool tasks: 365 too hard, 2 sweet spot, 70 too easy. |
| 2. Same rule | thermometer codes, Fourier features | Same failure. Almost no tasks in the middle. |
| 3. Same rule | plain scaled numbers | Some spread appeared, but even the easiest rule topped out near 55–85% on unseen tasks & every rule with `a ≥ 3` stayed at ~0%. |
| 4. Stripe rule `((a*x + b*y) // 4) mod 8` | plain scaled numbers | Unseen-task pass rates track training pass rates closely & spread smoothly from 0 to 1. After 1,000 steps on a 900-task pool: 66 too hard, 364 sweet spot, 302 too easy. |

**Why attempts 1–3 failed:** exact modular arithmetic on small integers changes its
answer completely when any input moves by one. A small MLP given a few hundred examples
has nothing smooth to latch onto, so it memorises. That gives two piles (memorised vs
never seen) & nothing in between, which is useless for a question about the middle.

**Why the stripe rule works:** dividing by the stripe width before taking the remainder
makes the answer stay the same across several neighbouring points. The function is
still the same family (a linear rule, then a remainder) but now it is learnable from
examples, & difficulty comes out graded: thin stripes are harder than wide ones, & points
at a stripe edge are harder than points in the middle.

**What I was careful about:** the world was picked using only the spread of pass rates
after pretraining. I had not run any GRPO training when I fixed it, so it could not have
been chosen to favour one bin.

**Other decisions**

- *Three splits, not two.* The RL pool is never seen during pretraining. That mirrors
  the real situation: a lab has a pretrained model & is deciding which new tasks to buy.
- *`a = b = 0` is left out* because its answer is always 0.

## Step 4: `policy.py` (the MLP, pretraining, pass rates & bins)

**Built:** a two-hidden-layer MLP (4 inputs → 128 → 128 → one logit per answer),
supervised pretraining with cross-entropy, two ways to measure a task's pass rate
(exact, from the softmax, & sampled, n = 32, as a real evaluation would), the three
bins & a helper that cuts all bins to the same number of tasks.

**The world changed in this step, & here is why.** With the first settings (8 answer
tokens, stripe width 4) I inspected the "too hard" bin & found it was not really hard:

- Blind guessing among 8 tokens is already right 12.5% of the time, which is *above*
  the 10% line. A task could only land in the bin if the policy was confidently wrong.
- The bin's median true pass rate was 8.6%, & only 10 of 1,500 pool tasks were below 1%.

So the bin held tasks the policy solves "rarely", not "never", & the experiment could
not have tested the "never solves" half of the question. I scanned five settings &
looked only at the bin sizes & how hard the hard bin was:

| Answer tokens | Stripe width | Pretrain steps | Hard / sweet / easy | Tasks at 0 of 32 | Hard bin median true pass rate |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 | 1,000 | 129 / 592 / 523 | 21 | 8.6% |
| **16** | **2** | **2,000** | **318 / 597 / 317** | **133** | **3.3%** |
| 16 | 3 | 2,000 | 116 / 483 / 640 | 47 | 5.4% |
| 32 | 2 | 2,000 | 238 / 631 / 337 | 108 | 2.5% |
| 32 | 1 | 2,000 | 726 / 459 / 128 | 477 | 0.7% |

I chose **16 tokens, stripe width 2, 2,000 pretraining steps**: the hard & easy bins
are almost the same size (318 & 317), & the hard bin is properly hard without swallowing
the whole pool. This was again decided before any GRPO run existed.

**After pretraining (final settings):** mean pass rate 76% on pretraining tasks, 50% on
the unseen pool & 51% on the test set. Pool & test agree, as they should, since both are
unseen. The gap to 76% means the policy partly memorised its pretraining tasks.

**A detail worth knowing:** bins are assigned from the *sampled* pass rate (n = 32), so
they carry sampling noise. The hard bin's true mean pass rate is 4.9%, the sweet spot's
51.1% & the easy bin's 94.2%. Tasks that got lucky or unlucky in 32 samples are in the
"wrong" bin, exactly as they would be in a real evaluation.

## Step 5: `grpo.py` (GRPO from scratch)

**Built:** the GRPO loss & training loop with no RL library. For each task in a batch:
sample G answers, reward 1 or 0, advantage = reward − group mean (optionally ÷ the
group's standard deviation), loss = −mean(advantage × log-prob of the sampled answer),
one Adam step. Every step logs the share of groups with a nonzero advantage.

**Decisions**

- *Tasks are drawn with replacement each step,* so every run with the same settings uses
  exactly `steps × batch_tasks × group_size` samples. That is what "equal compute" means
  in the experiment.
- *The plain advantage is the default; ÷ std is an option.* The spec lists it as
  optional, so the main experiment uses the plain form & the ÷ std form is an ablation.
- *One random generator per run* drives both which tasks are picked & which answers are
  sampled, so a run is fully determined by its seed.

**Tests that matter most**

- An all-correct or all-wrong group gives a loss of exactly 0 & a gradient of exactly 0,
  with & without ÷ std.
- The gradient on a tiny hand-worked example matches the formula
  `−advantage × (one-hot(sampled) − softmax)`.
- With the pass rate pinned at 20% & G = 8, the measured share of mixed groups matches
  `signal.py`'s formula to within 1%.
- Two runs with the same seed produce identical logs.
