# EXPLAIN: how this project works

This is a plain-language walkthrough for someone who knows neural networks, softmax, cross-entropy & the basics of RL, but is new to reading research code. Read it top to bottom once, then use the "Defend it" section to practise.

Contents:

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Words used here](#2-words-used-here)
3. [The pipeline, step by step](#3-the-pipeline-step-by-step)
4. [File by file](#4-file-by-file)
5. [How to read the results](#5-how-to-read-the-results)
6. [Defend it](#6-defend-it)
7. [Questions you will probably get](#7-questions-you-will-probably-get)

---

## 1. The project in one minute

GRPO trains a model by comparing several of its own answers to the same task. If all the answers are right, or all are wrong, there is nothing to compare & the model learns nothing from that task on that step. So the theory says: tasks the model solves *sometimes* should teach the most.

I tested that on a small synthetic world. I partly trained a small network, measured how often it solves each of 8,000 new tasks, sorted the tasks into too hard, sweet spot & too easy, & then trained a copy of the network on each pile with exactly the same compute.

What came out:

- The sweet-spot pile was the only one that made the model better on new tasks (+7.0 points). The too-hard pile made it worse (−9.2) & so did the too-easy pile (−2.4).
- A random mix of all tasks did better than any single pile (+9.9).
- The theory about zero gradients is correct & I measured it. But it is not the reason the extreme piles lost. They lost because training on one narrow kind of task made the model forget the other kinds.

---

## 2. Words used here

| Word | Meaning in this project |
| --- | --- |
| **Task** | Four small integers `(a, b, x, y)`. The model must output the right answer token. |
| **Answer token** | One of 16 possible outputs, numbered 0 to 15. |
| **Policy** | The model. It takes a task & outputs a probability for each of the 16 answers. |
| **Pass rate** | How often the policy answers a task correctly when it samples an answer. |
| **Group** | G answers sampled for the same task in one GRPO step. Here G = 8. |
| **Reward** | 1 if a sampled answer is correct, 0 if not. |
| **Advantage** | A sampled answer's reward minus its group's average reward. |
| **Mixed group** | A group with at least one right & at least one wrong answer. Only mixed groups give a gradient. |
| **Arm** | One version of the experiment. Here: train on too hard, on sweet spot, on too easy, or on a random mix. |
| **Seed** | The number that fixes the random choices, so a run can be repeated exactly. |
| **Held-out test set** | Tasks that are never trained on, used only to score the final models. |
| **Ablation** | Re-running the experiment with one setting changed, to see if the result survives. |
| **Replicate** | Re-running the whole experiment from scratch with a different shuffle of the world. |

---

## 3. The pipeline, step by step

### Step 1: build the world

There are 15,360 tasks. A task is `(a, b, x, y)` with `a, b` from 0 to 3 & `x, y` from 0 to 31. The answer is:

```
answer = ((a*x + b*y) // 4) mod 16
```

Worked example: `(a, b, x, y) = (1, 2, 3, 4)`. Then `a*x + b*y = 3 + 8 = 11`, `11 // 4 = 2`, `2 mod 16 = 2`. The answer is token 2.

A picture helps. Fix `(a, b)` & look at the 32×32 grid of points `(x, y)`. The quantity `a*x + b*y` grows steadily as you move across the grid in one direction. Dividing by 4 & rounding down chops it into stripes, 4 units wide. Taking `mod 16` reuses the 16 colours once you run out. So each `(a, b)` paints the grid with diagonal stripes, & the task is "which stripe is this point in?"

- Small `a` & `b` → wide stripes → easy.
- Large `a` & `b` → thin stripes → hard.
- A point in the middle of a stripe is easier than a point at its edge.

That gives a smooth range of difficulty, which is what the experiment needs.

The tasks are shuffled once with a fixed seed & cut into four piles that never share a task:

| Split | Size | Used for |
| --- | --- | --- |
| `pretrain` | 800 | Supervised pretraining |
| `pool` | 8,000 | The tasks we choose RL training sets from |
| `val` | 2,000 | Choosing the learning rate, nothing else |
| `test` | 2,000 | The final comparison, nothing else |

### Step 2: pretrain a policy, but not too well

The policy is an MLP: 4 inputs → 128 → 128 → 16 outputs, with ReLU in between. The 16 outputs are logits, & softmax turns them into probabilities.

It is trained for 2,000 steps with cross-entropy on the 800 pretraining tasks, the same way makemore trains a next-character model. Stopping at 2,000 steps is deliberate. Afterwards the policy passes 77% on the tasks it trained on & 47% on tasks it has never seen. Some unseen tasks it nearly always gets, some it never gets, & many are in between.

### Step 3: measure every pool task's pass rate

For each of the 8,000 pool tasks, sample 32 answers from the policy & count how many are correct. That count divided by 32 is the measured pass rate. This is exactly what a real evaluation of a coding model does: run each task n times & count passes.

Because this policy is just a softmax, we can also read off the *true* pass rate (the probability it puts on the right answer). A real coding model can't do that, but here it lets us check the measurements.

### Step 4: bin the tasks

| Bin | Measured pass rate | Tasks in the pool | True average pass rate |
| --- | --- | --- | --- |
| Too hard | < 10% (0–3 correct of 32) | 2,154 | 3.6% |
| Sweet spot | 20–80% (7–25 of 32) | 2,939 | 51.6% |
| Too easy | > 90% (29–32 of 32) | 1,645 | 94.8% |

Tasks at 10–20% & 80–90% are left out, so the bins don't touch. Each bin is then cut to 1,500 randomly chosen tasks. That way no arm gets more tasks, or more variety, than another.

A fourth training set, the **random mix**, is 1,500 tasks drawn at random from the whole pool with no filter.

### Step 5: choose the learning rate fairly

The learning rate matters a lot in RL, & I didn't want to pick one that happened to favour a bin. So the rule is: train the random-mix arm at four learning rates, score each on the `val` split, keep the best. The three bins & the test set are never looked at during this choice. The winner was 0.001.

### Step 6: train with GRPO

Each arm starts from a copy of the *same* pretrained checkpoint. One GRPO step:

1. Pick 32 tasks at random from the arm's 1,500.
2. For each task, sample 8 answers from the policy.
3. Reward each answer 1 or 0.
4. Advantage = reward − the average reward of those 8.
5. Loss = −average of (advantage × log-probability of the sampled answer).
6. One Adam step.

Every arm runs 3,000 steps, so every arm uses 3,000 × 32 × 8 = 768,000 sampled answers. Each arm is run 5 times with different seeds.

### Step 7: score every run on the same test set

Every 25 steps, each run's pass rate on the 2,000 test tasks is recorded. The headline number is that pass rate at step 3,000, averaged over the 5 seeds.

### Step 8: check it wasn't a fluke

- **Ablations:** repeat with G = 4, G = 16, with advantage ÷ standard deviation, & with three other learning rates.
- **Replicates:** repeat everything four more times with a different shuffle of the world & a freshly pretrained checkpoint.
- **Follow-up arms:** the random mix with the too-hard bin removed, & the random mix with the too-easy bin removed.

---

## 4. File by file

All the code is in the `grpo_difficulty/` folder. Each file can be run on its own, for example `python -m grpo_difficulty.signal`.

### `passk.py`: measuring pass@k

**pass@k** is the chance that at least one of k sampled answers is correct.

- `pass_at_k_unbiased(n, c, k)`: you sampled n answers & c were correct. It returns `1 − C(n−c, k) / C(n, k)`. In words: of all the ways to pick k of your n answers, what share contains at least one correct answer? The code computes this as a product of small fractions, because the binomial numbers themselves get astronomically large.
- `pass_at_k_naive(n, c, k)`: the obvious shortcut. Estimate the pass rate as `c/n`, plug it into `1 − (1 − c/n)^k`. It is biased low (see "Defend it").
- `bootstrap_ci(...)`: a confidence interval for a benchmark's average pass@k. It redraws the list of tasks with replacement 2,000 times & sees how much the average moves.
- `bias_demo()`: simulates both estimators 200,000 times. With a true pass rate of 10% & only n = 10 samples, the true pass@10 is 0.651, the unbiased estimator averages 0.652 & the naive one averages 0.494.

How it connects to the experiment: Step 3 above is a pass@1 measurement with n = 32. The experiment also scores the whole pool with pass@8 as a check: unbiased 0.758 (95% interval 0.750–0.766), true value 0.757, naive 0.747.

### `signal.py`: the theory

One function carries the idea of the whole project:

```python
def signal_fraction(p, G):
    return 1.0 - p**G - (1.0 - p)**G
```

For a task with pass rate `p`, the chance all G answers are right is `p^G`, the chance all are wrong is `(1−p)^G`, & in both cases the group gives no gradient. What's left is the chance the group is mixed.

| Pass rate p | G = 4 | G = 8 | G = 16 |
| --- | --- | --- | --- |
| 1% | 4% | 8% | 15% |
| 5% | 19% | 34% | 56% |
| 10% | 34% | 57% | 82% |
| 50% | 88% | 99% | 100% |
| 90% | 34% | 57% | 82% |
| 99% | 4% | 8% | 15% |

Notice what this table already says: at a 10% pass rate with G = 8, more than half the groups still carry signal. The dead zone is narrower than "below 10%" suggests. It is really "below 1–2%".

The file also has `mean_squared_advantage`, which says how *big* the advantages are on average (`p(1−p)(G−1)/G`), & a simulation that confirms the formula.

### `world.py`: the tasks

- `rule(a, b, x, y, ...)`: the hidden rule from Step 1.
- `encode(tasks, config)`: scales the four integers to the range −1 to 1, which is what the network sees.
- `make_world(config)`: lists every task, shuffles with a fixed seed, computes the answers & cuts the four splits.
- `WorldConfig`: all the world's settings in one place.

### `policy.py`: the model & everything before RL

- `Policy`: the MLP.
- `pretrain(...)`: the supervised training loop (cross-entropy, Adam).
- `pass_probability(...)`: the true pass rate of each task, read from the softmax.
- `sampled_pass_counts(...)`: the measured pass rate, by sampling n answers.
- `assign_bins(...)`: sorts tasks into the three bins.
- `equal_size_bins(...)`: cuts every bin to the same size.

### `grpo.py`: the RL algorithm

The heart of it is short enough to read in full:

```python
def group_advantages(rewards, normalize=False, eps=1e-6):
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize:
        advantages = advantages / (rewards.std(dim=1, keepdim=True, correction=0) + eps)
    return advantages

def grpo_loss(logits, sampled, rewards, normalize=False):
    log_probs = F.log_softmax(logits, dim=1)
    sampled_log_probs = log_probs.gather(1, sampled)
    advantages = group_advantages(rewards, normalize)
    return -(advantages * sampled_log_probs).mean()
```

`rewards` has one row per task & one column per sampled answer. `rewards.mean(dim=1, keepdim=True)` is each row's average, so the subtraction gives "reward minus group mean". `gather` picks out the log-probability of each answer that was actually sampled.

- `sample_groups(...)`: samples G answers per task & scores them.
- `grpo_step(...)`: one update.
- `train_grpo(...)`: the loop. It also logs, at every step, the share of groups that were mixed, & it sorts every sampled group into a bucket by its task's pass rate at that moment, so the theory curve can be checked against what really happened.

### `experiment.py`: putting it together

- `prepare(config)`: Steps 1–4 (world, checkpoint, pass rates, bins).
- `select_lr(setup)`: Step 5.
- `run_arm(setup, arm, seed)`: one GRPO run with its evaluations.
- `run_experiment`, `run_ablations`, `run_replicates`: Steps 6–8.
- `run_pipeline()`: all of it in order. The result is saved to `outputs/results.json`.

### `plots.py` & `style.py`

`plots.py` reads `outputs/results.json` & makes the three result figures plus `outputs/summary.md`, the tables the README's numbers come from. `style.py` holds the colours & fonts so all figures match.

### `run_all.py`

Runs every stage in order. `make all` calls it.

### `tests/`

75 tests. The ones most worth knowing about:

- The unbiased pass@k estimator is checked against brute-force counting, & its average over every possible outcome equals the true value to 12 decimal places.
- An all-correct or all-wrong group gives a loss of exactly 0 & a gradient of exactly 0.
- The GRPO gradient on a tiny example matches a hand calculation.
- No task appears in two splits.
- Every arm has the same number of tasks & uses the same number of samples.
- The same seed gives the same run.

---

## 5. How to read the results

### The main chart: `figures/heldout_accuracy.png`

The flat grey line at 47.2% is the pretrained checkpoint. Above it is better than doing nothing, below it is worse.

- **Random mix** climbs steadily to 57.1%.
- **Sweet spot** climbs to 54.2%.
- **Too easy** drifts slightly down to 44.8%. It has almost nothing to learn from its tasks, & the small updates it does make nudge other tasks the wrong way.
- **Too hard** drops to about 33% within the first 50 steps, then slowly recovers to 38.0%.

The shaded bands are ± 1 standard deviation over 5 seeds. They are narrow compared with the gaps between lines, so the ordering is not luck.

### The signal chart: `figures/signal_fraction.png`

**Left panel:** the share of groups with a nonzero advantage at each step. Solid lines are measured, dotted lines are what `signal_fraction` predicts from each task's pass rate at that moment. They lie on top of each other. The sweet-spot run starts at 94% & falls as its tasks get solved (they become too easy). The two extreme runs start near 20–30% & fall towards 6–8%.

**Right panel:** every group sampled during training, sorted by its task's pass rate at the time. The dots are the measured share of mixed groups. The line is the formula. This is the theory, confirmed.

### The surprise: `figures/gain_by_difficulty.png`

Split the test tasks by how hard they were for the pretrained checkpoint, & look at each group separately:

| Trained on | Test tasks that started too hard | ...in the sweet spot | ...too easy |
| --- | --- | --- | --- |
| Too hard | 2.5% → 49.6% | 49.8% → 35.8% | 95.9% → 25.3% |
| Sweet spot | 2.5% → 16.0% | 49.8% → 68.7% | 95.9% → 69.2% |
| Too easy | 2.5% → 2.3% | 49.8% → 43.2% | 95.9% → 98.0% |
| Random mix | 2.5% → 30.3% | 49.8% → 62.1% | 95.9% → 79.6% |

Read the first row. Training on too-hard tasks took hard test tasks from 2.5% to 49.6%. That is the biggest improvement anywhere in the project. So the too-hard run *did* learn. The theory predicted it would be starved of gradient, & at the first step only 21% of its groups carried signal, but 21% was enough to start, & each success made the next more likely.

Then look at what it cost: tasks the model used to solve 95.9% of the time fell to 25.3%. The network is small & every task shares the same weights. Pushing hard to fix one kind of task, with no examples of the other kinds in the batch to hold them in place, overwrote what it knew. This is called **catastrophic forgetting**.

The random mix avoids this because every batch contains all kinds of tasks.

### The follow-up: which filter is worth applying?

| Trained on | Held-out pass rate |
| --- | --- |
| Random mix | 57.1% |
| Random mix minus too-hard | 56.5% |
| Random mix minus too-easy | 54.1% |

Dropping the too-hard tasks changed nothing beyond noise. Dropping the too-easy tasks cost 3 points. Easy tasks don't teach anything new, but they act as an anchor that stops the model drifting away from what it already does well.

### How sure can we be?

- 5 seeds per arm, with gaps much larger than the spread.
- The ranking (random mix > sweet spot > too easy > too hard) held in all 6 ablations & all 4 replicate worlds.
- It is still one toy world. See the Limitations in the README before repeating any of this as a general fact.

---

## 6. Defend it

### Write the GRPO loss for one group & explain each term.

For one task, sample G answers `a_1 ... a_G` from the policy. Then:

```
loss = − (1/G) × Σ_i  A_i × log π(a_i | task)

A_i   = r_i − r̄                (optionally divided by the group's standard deviation)
r_i   = 1 if a_i is correct, else 0
r̄     = (r_1 + ... + r_G) / G
```

- **`π(a_i | task)`** is the probability the policy gives to answer `a_i`. Its log is what we differentiate.
- **`r_i`** is the reward: did this answer pass the check?
- **`r̄`** is the group's average reward, the baseline. It answers "how well does the policy usually do on this task right now?"
- **`A_i`** is the advantage: was this answer better or worse than usual for this task? It is treated as a fixed number. No gradient flows through it.
- **`A_i × log π`**: minimising the loss raises the log-probability of answers with a positive advantage & lowers it for answers with a negative one. This is the policy-gradient idea from REINFORCE.
- **The minus sign** is there because optimisers minimise, & we want to maximise reward.
- **`1/G`** averages over the group. In the code the average also runs over all tasks in the batch.

Worked example with G = 4 & rewards `[1, 0, 0, 0]`: the mean is 0.25, so the advantages are `[+0.75, −0.25, −0.25, −0.25]`. The one correct answer gets pushed up strongly, & the three wrong ones get pushed down a little each.

Subtracting the group mean, rather than using the raw reward, makes the signal about *this* task: an easy task where everything scores 1 does not flood the gradient just because its rewards are large.

### Why does an all-correct group give zero gradient? Use the advantage formula.

If all G answers are correct, every `r_i = 1`, so `r̄ = 1` & every `A_i = 1 − 1 = 0`. Every term in the loss is `0 × log π(a_i)`, so the loss is 0 no matter what the weights are, & the gradient of a constant 0 is 0.

The same happens when all are wrong: `r_i = 0`, `r̄ = 0`, `A_i = 0`.

The intuition: the advantage measures an answer *relative to its group*. When every answer got the same score, none is better than another, so there is nothing to push towards & nothing to push away from. Dividing by the standard deviation does not rescue it, because the numerator is already exactly 0.

A test in `tests/test_grpo.py` checks this: the loss & every entry of the gradient come out as exactly 0.

How often does it happen? For a task with pass rate p, with probability `p^G + (1−p)^G`. At p = 99% & G = 8 that is 92% of groups.

### Why must the three runs use equal compute & the same test set?

**Equal compute**, because more training almost always gives a better model. If the sweet-spot run got more steps or more samples, we could not tell "the sweet spot teaches more" apart from "more training teaches more". Here every run gets the same starting checkpoint, the same number of tasks to draw from (1,500), the same steps (3,000), the same tasks per step (32), the same group size (8), the same optimiser & the same learning rate. The only thing that differs is which tasks are in the training set. Then any difference in the result must come from that.

**The same test set**, because scores on different test sets are not comparable. If each run were tested on tasks like its own training tasks, the too-easy run would score about 99% & the too-hard run about 50%, & that would say nothing about which one learned more. One shared test set, covering every difficulty & never trained on, puts all runs on the same scale. The question is about what a model *gains in general*, so the test must not be tilted towards any arm.

Two smaller points in the same spirit: the learning rate was chosen on a separate validation split using only the random-mix arm, so the test set was never used to make a choice, & every run starts from the same checkpoint, so no run begins ahead.

### What does pass@k measure, & why is the naive estimator biased?

**What it measures:** the probability that at least one of k independent attempts at a task is correct. If the true pass rate is p, that is `1 − (1−p)^k`. pass@1 is the plain pass rate. pass@10 answers "if the model gets ten tries, does it solve it?"

**The naive estimator:** sample n answers, count c correct, estimate `p̂ = c/n`, then compute `1 − (1 − p̂)^k`.

**Why it is biased:** `p̂` itself is fine on average. The problem is putting a noisy number through a curved function. `(1 − p̂)^k` is a curve that bends upward, & for a curve like that, the average of the curve's values is larger than the curve's value at the average. So on average `(1 − p̂)^k` comes out too big, which makes `1 − (1 − p̂)^k` too small. The naive estimator reads low.

A concrete case: true p = 10%, k = 10, n = 10. With probability `0.9^10 ≈ 35%` you see 0 correct, & the naive estimate is 0. Most of the remaining time you see exactly 1 correct, `p̂ = 0.1`, & the naive estimate is `1 − 0.9^10 = 0.65`. To average out at the true 0.65 it would need to sometimes be far above 0.65, & it can't make up for all those zeros. Its average is 0.49.

**The unbiased estimator** avoids estimating p at all. It asks directly: of all `C(n, k)` ways to choose k of my n samples, what share has at least one correct answer? That is `1 − C(n−c, k) / C(n, k)`, & its average is exactly `1 − (1−p)^k` for any n ≥ k.

The bias shrinks as n grows (n = 200 gives 0.643 against 0.651), but real evaluations often use small n because samples are expensive, which is exactly when it matters.

### If this result held on real coding tasks, how should a training-data broker price or filter traces?

Taking the result at face value, & keeping in mind it comes from a toy:

1. **Measure difficulty, & sell the measurement.** For every task, report its pass rate against the buyer's own model, from at least 32 samples, with the unbiased pass@k estimator & a confidence interval. Difficulty is a property of the task *& the model together*, so it has to be measured per buyer & re-measured as models improve. This metadata is the product. Everything below depends on it.
2. **Don't sell "sweet spot only" as a replacement for a broad set.** In this experiment a random mix beat the sweet-spot bin. A buyer who trains on a narrow difficulty band risks the model forgetting what it already does. Sell balanced bundles & show the difficulty breakdown of each one.
3. **Price the sweet spot highest per task.** It was the only bin that helped on its own & it carries signal in nearly every group (94% against 21–30%). If a buyer can only afford a small set, this is the set.
4. **Keep some easy tasks, priced low.** They taught nothing new, but removing them from the mix cost 3 points. Position them as a cheap anchor or regression set, not as waste to delete.
5. **Be careful with too-hard tasks, & split them in two.** Tasks the model passes rarely (1–3 times in 32) still trained in this experiment & gave the largest gains on hard test tasks. Tasks at a true 0% give no gradient at all, & in real data many of those are broken verifiers or impossible specs. So: verify that a too-hard task is solvable at all, price solvable-but-rare ones as specialist stock for buyers targeting hard problems, & price never-solved ones near zero until someone shows they can be solved.
6. **Price by composition, not by volume.** Two bundles of the same size were worth very different amounts here: one raised held-out accuracy by 9.9 points, another lowered it by 9.2.
7. **Hold on to hard inventory.** Today's too-hard task becomes next year's sweet-spot task as models improve, so its value rises over time.

What I would say out loud to a researcher: "In my toy setup the sweet spot beat both extremes, as the zero-gradient argument predicts, but a plain random mix beat the sweet spot, because single-difficulty training caused forgetting. So I would use difficulty to *describe & balance* a dataset rather than to filter it down to one band. I'd want to see whether that survives with a real model & a drift penalty before putting numbers on a price list."

---

## 7. Questions you will probably get

**"Isn't the forgetting just because your network is tiny?"**
Probably in part, yes. A 19,216-weight MLP shares every weight across every task, & the GRPO here has no clipping & no penalty for drifting from the starting policy. Real pipelines have both. That is the first limitation in the README & the second thing on the "next" list. What I can say is that in this setup the effect was large & consistent across 5 seeds, 6 ablations & 4 reshuffled worlds.

**"Why did the too-hard run learn at all, if most of its groups had zero gradient?"**
Because "most" is not "all". Its tasks averaged a 3.6% true pass rate, which at G = 8 means about 21% of groups were mixed. Each mixed group pushed a correct answer up, & since tasks share features, that raised the pass rate of similar tasks too. The wall the theory describes is real but sits lower, around 1% & below, where under one group in twelve carries signal.

**"Did you tune things until you got the answer you wanted?"**
No, & the answer I got was not the one I expected. The world's settings were fixed by looking only at the spread of pass rates after pretraining. The learning rate was chosen with a written rule, on the random-mix arm & a validation split. The build log records the first attempt, where every arm got worse, & what was changed as a result. The two follow-up arms are labelled as added after seeing the main result.

**"Why a synthetic world & not real coding tasks?"**
Control & cost. Here the true pass rate of every task is known exactly, the verifier is perfect, a full run takes about 5 minutes on a laptop, & I could repeat it 100+ times. The price is that it is a toy, & the README says so.

**"Why is the score a pass rate & not greedy accuracy?"**
Pass rate (the probability of sampling the right answer) is what GRPO directly optimises & what an evaluation that samples would measure. Greedy accuracy is in `outputs/summary.md` too, & tells the same story: 51.6% at the start, then 38.7% (too hard), 57.6% (sweet spot), 45.7% (too easy) & 61.1% (random mix).

**"What does ÷ std change?"**
It rescales each group's advantages to have standard deviation 1, so a group with one success in eight pushes as hard as a group with four in eight. In this experiment it changed the final numbers by less than 1.5 points & did not change the ranking.

**"What was the hardest part?"**
Getting a world where the question could be asked at all. Plain modular arithmetic was memorised rather than learned, so there was no middle. Then with only 300 tasks per arm, RL memorised its training tasks & nothing generalised. Both are written up in `BUILDLOG.md`.
