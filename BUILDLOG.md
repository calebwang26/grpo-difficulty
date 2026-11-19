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
