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
