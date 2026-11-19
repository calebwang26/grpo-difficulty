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

## Step 6: `experiment.py`, `plots.py`, `run_all.py` (the experiment itself)

This step took three attempts. The first two are recorded here because they changed the
design, & because the first one is a result in its own right.

### Attempt 1: 300 tasks per bin, settings fixed in advance (failed, informatively)

Settings chosen before running anything: G = 8, 32 tasks per step, Adam lr 1e-3, 1,000
steps, 300 tasks per bin, world of 3,840 tasks.

**What happened:** held-out pass rate went *down* for every arm, including the
difficulty-blind random mix.

| Trained on | Held-out pass rate, start → end |
| --- | --- |
| Too hard | 51.1% → 34% |
| Sweet spot | 51.1% → 46% |
| Too easy | 51.1% → 42% |
| Random mix | 51.1% → 45% |

**Diagnosis (it was not a bug):** each run's pass rate on its *own* 300 training tasks
climbed (the mix went 52% → 81%) while held-out fell. The policy was memorising its 300
tasks. I then tried four learning rates from 3e-5 to 1e-3 on the random-mix arm only, &
all four showed the same thing: training up, held-out down.

**Why this matters:** if even difficulty-blind RL can't improve held-out accuracy, then
"which bin teaches more?" has no meaningful answer. The experiment needed a world where
RL on new tasks generalises at all.

### Attempt 2: a denser world, & a rule for choosing the learning rate

Two changes, both made by looking only at the random-mix arm & a new validation split.
The three bins & the test set were not consulted.

1. **Denser world.** Grid 16 → 32 & stripe width 2 → 4. This is the same stripe pattern
   sampled twice as finely in each direction, so there are 15,360 tasks instead of
   3,840. The pool is now 8,000 tasks, & each arm gets 1,500 tasks instead of 300. With
   1,500 tasks a run sees each task about 64 times instead of 107 & has far more distinct
   examples to generalise from. On this world, RL on the random mix raised validation
   pass rate from 46.5% to 56.2%.
2. **A fourth split, `val` (2,000 tasks),** used only to choose the learning rate. The
   rule, written into `select_lr`: train the random-mix arm at each of four learning
   rates (3 seeds each) & keep the one with the best final validation pass rate. Result:
   1e-4 → 50.5%, 3e-4 → 53.1%, **1e-3 → 56.2%**, 3e-3 → 50.9%.

Steps went from 1,000 to 3,000 because validation was still improving at 1,000.

**Is this tuning toward a desired answer?** I don't think so, for three reasons: the
choice used an arm that has no difficulty filter, it used a split the final comparison
never touches, & the ablation table shows the ranking of the arms is the same at all
four learning rates anyway.

### Attempt 3: the full pipeline (this is what `make all` runs)

1. World, pretraining, pass rates, bins (cut to 1,500 tasks each).
2. Learning-rate selection as above.
3. Main experiment: 4 arms × 5 seeds, 3,000 steps × 32 tasks × G = 8 = 768,000 samples
   per run, all from the same checkpoint, all scored on the same 2,000 test tasks.
4. Follow-up arms, added *after* seeing the main result: random mix with the too-hard
   bin removed, & random mix with the too-easy bin removed.
5. Ablations: G = 4 & G = 16 (with tasks per step adjusted so samples per step stay at
   256), the ÷ std advantage, & the three other learning rates.
6. Replicates: four reshuffled worlds, each with its own freshly pretrained checkpoint.

### Results

| Trained on | Held-out pass rate (5 seeds, mean ± sd) | Change |
| --- | --- | --- |
| Nothing (pretrained checkpoint) | 47.2% | |
| Too hard (<10%) | 38.0% ± 1.1% | −9.2 pts |
| Sweet spot (20–80%) | 54.2% ± 0.5% | +7.0 pts |
| Too easy (>90%) | 44.8% ± 0.4% | −2.4 pts |
| Random mix | 57.1% ± 0.8% | +9.9 pts |
| Random mix minus too-hard | 56.5% ± 0.3% | +9.3 pts |
| Random mix minus too-easy | 54.1% ± 0.5% | +6.9 pts |

- **Among the three bins the sweet spot wins clearly,** & it is the only bin that
  improves held-out accuracy at all. The ranking sweet > easy > hard held in all 6
  ablations & all 4 replicate worlds.
- **But the random mix beats the sweet spot,** in the main run, in all 6 ablations &
  in all 4 replicates. I did not expect this.
- **The "zero gradient" story is true at the level of a single step & measured exactly:**
  the share of mixed groups sits on the `1 − p^G − (1−p)^G` curve (right panel of
  `figures/signal_fraction.png`). At the start 94% of sweet-spot groups carried signal
  against 21% for too-hard & 30% for too-easy.
- **But it is not what decided the outcome.** The too-hard run was not starved: its own
  tasks went from 3.6% to 51.6%, & test tasks that started too hard went from 2.5% to
  49.6%, the biggest gain anywhere in the experiment. It lost overall because it
  *forgot*: test tasks that started too easy fell from 95.9% to 25.3%. Every single-bin
  run gains on its own difficulty band & loses on others (`figures/gain_by_difficulty.png`).
- **Too-easy tasks are not worthless.** On their own they teach nothing new (their run
  barely moves). Inside a mix they stop the policy forgetting what it already knows:
  removing them from the mix cost 3 points.

**Other things noticed**

- Outputs are identical across two full runs on this machine (same numbers to every
  printed digit), so the seeding works.
- The pool, scored like a benchmark with `passk.py`: pass@8 unbiased 0.758 (95% CI
  0.750–0.766) against a true value of 0.757; the naive estimator gives 0.747. So the
  estimator from Step 1 checks out on the experiment's own data, & the naive one is low
  even at n = 32.

**Problems hit**

- I piped the first full run through `head`, which killed the process before it saved
  its results. Reran it without the pipe.
- A full run took 4 minutes when the laptop was idle & 11 minutes while the other two
  project sessions were using the CPU.
- First versions of the plots had overlapping labels, & one chart title claimed more
  than the data showed ("each run helps its own difficulty & hurts the rest": the
  sweet-spot run also helps hard tasks). Replaced with a neutral title.

## Step 7: README & EXPLAIN

**Built:** `README.md` in the required structure, with every number copied from
`outputs/summary.md` (which `make all` regenerates), & `EXPLAIN.md`, a walkthrough of the
pipeline & every file plus written answers to the five "Defend it" questions.

**Decisions**

- *The README leads with the unexpected part.* The honest one-line answer is "partly":
  the sweet spot beat both extremes, but a random mix beat the sweet spot.
- *Limitations are specific to this setup,* above all that a tiny network with no drift
  penalty probably exaggerates forgetting, & that the too-hard bin here (median true pass
  rate 1.7%) is not the same as tasks at a true 0%.
- *The business context is described generically* ("AI training-data brokers"), with no
  company, customer or partner names & no prices.

## Step 8: pre-publish checklist

Results of each check when first run, with the repo still **private**:

| # | Check | Result |
| --- | --- | --- |
| 1 | `gitleaks detect` over the entire git history | Pass: no leaks found |
| 2 | Only the no-reply email in `git log` | **Fail on this first run: see below.** Fixed in Step 9 |
| 3 | No confidential business terms or personal details in the repo | Pass: none found in any tracked file. The only personal detail is the name in `LICENSE` |
| 4 | No file over 50 MB, small repo | Pass: 1.2 MB of git objects, largest tracked file is a 172 KB PNG |
| 5 | CI passes on `main`, & a fresh clone runs end to end | Pass: CI is green on `main`. A fresh `git clone` into a temp folder followed by `make all` finished in 327 seconds & produced a `summary.md` identical, character for character, to the one the README was written from |
| 6 | Owner reviews the README & approves | Waiting at this point. Approved in Step 9 |
| 7 | Switch the repo to public | Not done at this point, waiting on 2 & 6. See Step 9 |

**Check 2 failed, & why.** Every commit I made locally carries the no-reply address.
But the seven merge commits were created by GitHub when each pull request was merged, &
GitHub signed those with the account's primary email address rather than the no-reply
one. Local git settings don't apply to commits GitHub creates.

- *Stopping it happening again:* I first tried `gh pr merge --author-email <no-reply
  address>` on pull request 8. GitHub refused it ("Invalid email address"), which
  suggests "Keep my email addresses private" is switched off for the account. So pull
  requests 8 & 9 were merged on the laptop instead (`git merge --no-ff` after CI passed,
  then a normal push), which signs the merge commit with the local no-reply address.
  GitHub still marks the pull request as merged. Turning on "Keep my email addresses
  private" in GitHub → Settings → Emails would make ordinary merges safe.
- *Fixing the seven existing commits* means rewriting history & force-pushing `main`.
  That is on the "stop & ask" list, so I stopped & asked. Step 9 has what happened next.

**Other notes**

- `HANDOFF.md` is in `.gitignore` & has never been committed.
- Branch protection is not available on private repos on a free plan, so "a PR merges
  only when both checks pass" was enforced by hand while the repo was private: every
  merge waited for the tests & gitleaks checks to go green. The one failing check (pull request 1, a token-permission
  error) was fixed before merging.
- Removed one unused import from `tests/test_experiment.py`.

## Step 9: fixing the email & publishing through a fresh repo

I stopped after Step 8 & asked the owner. His decisions:

- He switched on "Keep my email addresses private" in his GitHub settings himself.
- He approved rewriting the seven merge commits & force-pushing `main`, with a backup
  taken first.
- He approved the README.
- He chose to publish through a **fresh repo** rather than flipping the build repo to
  public. The reason is below.

**The rewrite.** After saving a backup of every branch (`git bundle create ... --all`),
I ran `git filter-branch --env-filter` over `main` with one rule: where a commit's
author email is the personal address, replace it with the no-reply address. Then I
checked that nothing else had moved:

| Check | Result |
| --- | --- |
| Number of commits | 41 before, 41 after |
| The file tree recorded by each commit | Identical for all 41 |
| Names, dates & commit messages | Identical for all 41 |
| Emails left in `git log` | Only the no-reply address (plus GitHub's own `noreply@github.com` as committer on the seven merges it made) |

The push used `git push --force-with-lease`, which refuses to overwrite anything if the
remote has changed since I last looked. 35 of the 41 commit ids changed. The first six
commits come before the first merge, so they kept their ids.

**Why a fresh repo.** Rewriting history does not delete the old commits from GitHub.
Each one stays retrievable by its id, & every merged pull request page links straight
to its old merge commit. Making the build repo public would have published the personal
address anyway. So:

1. The build repo was renamed to `grpo-difficulty-dev` & stays private. It is the
   archive: it keeps pull requests 1–9 & their CI runs.
2. A new, empty repo was created under the original name, & only the rewritten `main`
   was pushed to it. It has no other branches.
3. I asked the new repo for each of the 35 replaced commit ids through GitHub's API.
   All 35 came back "not found".

One side effect: the merge commits in this repo's history say "Merge pull request #1"
to "#9", but those pull request pages live in the private archive. Pull request numbers
start again from 1 here, beginning with the one that added this section.

**Does email privacy work now?** Tested before relying on it. In the private archive I
opened a throwaway pull request into a throwaway branch (not `main`), merged it with an
ordinary `gh pr merge`, & read the merge commit: the author was the no-reply address.
Both branches were then deleted. Ordinary merges are safe again, so the laptop-merge
workaround from Step 8 is no longer needed.

**The checklist again, on a fresh clone of the new repo** (still private at this point):

| # | Check | Result |
| --- | --- | --- |
| 1 | `gitleaks detect` over the entire git history | Pass: no leaks found |
| 2 | Only the no-reply email in `git log` | Pass |
| 3 | No confidential business terms or personal details | Pass: none in any tracked file or commit message, in any commit |
| 4 | No file over 50 MB, small repo | Pass: 836 KB of git data, largest tracked file is a 172 KB PNG |
| 5 | CI passes on `main`, & a fresh clone runs end to end | Pass: CI is green on the new repo's `main`. A fresh `git clone` of the new repo followed by `make all` finished in 276 seconds & produced a `summary.md` identical, character for character, to the one the README was written from |
| 6 | Owner reviews the README & approves | Pass: approved |
| 7 | Switch the repo to public | The last step. Done only after this section is merged, CI is green on `main` & checks 1 & 2 pass once more on the final history |
