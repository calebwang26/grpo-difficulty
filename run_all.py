"""One command that rebuilds every number & every figure in this repo.

    python run_all.py

Takes a few minutes on a laptop CPU. Everything is seeded, so two runs on the same
machine give the same numbers.
"""

import time

from grpo_difficulty import experiment, passk, plots, policy, signal


def main():
    start = time.time()
    print("=== 1/5 pass@k: the naive estimator's bias ===")
    passk.main()
    print("\n=== 2/5 Theory: share of GRPO groups that carry signal ===")
    signal.main()
    print("\n=== 3/5 Pretrained policy & difficulty bins ===")
    policy.main()
    print("\n=== 4/5 The experiment ===")
    experiment.main()
    print("\n=== 5/5 Figures & summary tables ===")
    plots.main()
    print(f"\nAll done in {time.time() - start:.0f}s. Figures are in figures/, numbers in outputs/.")


if __name__ == "__main__":
    main()
