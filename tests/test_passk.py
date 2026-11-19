import itertools
import math

import numpy as np
import pytest

from grpo_difficulty.passk import (
    bias_demo,
    bootstrap_ci,
    mean_pass_at_k,
    pass_at_k_naive,
    pass_at_k_unbiased,
    true_pass_at_k,
)


def brute_force(n, c, k):
    """Count size-k subsets of n samples (c correct) with at least one correct."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(samples, k))
    return sum(1 for s in subsets if any(s)) / len(subsets)


@pytest.mark.parametrize("n,c,k", [(5, 2, 2), (6, 1, 3), (7, 3, 1), (8, 0, 4), (8, 8, 4), (6, 5, 3)])
def test_unbiased_matches_brute_force(n, c, k):
    assert pass_at_k_unbiased(n, c, k) == pytest.approx(brute_force(n, c, k))


def test_unbiased_matches_binomial_formula():
    n, c, k = 30, 7, 5
    expected = 1 - math.comb(n - c, k) / math.comb(n, k)
    assert pass_at_k_unbiased(n, c, k) == pytest.approx(expected)


def test_edge_cases():
    assert pass_at_k_unbiased(10, 0, 3) == 0.0
    assert pass_at_k_unbiased(10, 10, 3) == 1.0
    assert pass_at_k_unbiased(10, 8, 3) == 1.0  # only 2 wrong, so any 3 include a correct
    assert pass_at_k_unbiased(10, 4, 1) == pytest.approx(0.4)  # pass@1 is the pass rate


def test_stable_for_large_n():
    value = pass_at_k_unbiased(100000, 500, 100)
    assert 0.0 < value < 1.0
    assert value == pytest.approx(true_pass_at_k(0.005, 100), abs=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pass_at_k_unbiased(5, 2, 6)
    with pytest.raises(ValueError):
        pass_at_k_unbiased(5, 6, 2)
    with pytest.raises(ValueError):
        pass_at_k_naive(5, -1, 2)


@pytest.mark.parametrize("p,n,k", [(0.1, 10, 5), (0.3, 8, 8), (0.05, 20, 10)])
def test_unbiased_is_exactly_unbiased(p, n, k):
    """Average the estimator over every possible outcome, weighted by its probability."""
    expectation = sum(
        math.comb(n, c) * p**c * (1 - p) ** (n - c) * pass_at_k_unbiased(n, c, k)
        for c in range(n + 1)
    )
    assert expectation == pytest.approx(true_pass_at_k(p, k), abs=1e-12)


@pytest.mark.parametrize("p,n,k", [(0.1, 10, 5), (0.3, 8, 8), (0.05, 20, 10)])
def test_naive_is_biased_low(p, n, k):
    expectation = sum(
        math.comb(n, c) * p**c * (1 - p) ** (n - c) * pass_at_k_naive(n, c, k)
        for c in range(n + 1)
    )
    assert expectation < true_pass_at_k(p, k) - 0.01


def test_naive_equals_unbiased_at_k_1():
    assert pass_at_k_naive(20, 7, 1) == pytest.approx(pass_at_k_unbiased(20, 7, 1))


def test_bias_demo_shows_shrinking_bias():
    rows = bias_demo(p=0.1, k=10, sample_sizes=(10, 100), trials=20000, seed=0)
    small, large = rows
    assert abs(small["unbiased_mean"] - small["truth"]) < 0.01
    assert small["naive_mean"] < small["truth"] - 0.05
    assert abs(large["naive_mean"] - large["truth"]) < abs(small["naive_mean"] - small["truth"])


def test_bootstrap_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    counts = rng.binomial(32, 0.3, size=200)
    point, low, high = bootstrap_ci(32, counts, k=4, n_boot=500, seed=1)
    assert low < point < high
    assert point == pytest.approx(mean_pass_at_k(32, counts, 4))
    assert (point, low, high) == bootstrap_ci(32, counts, k=4, n_boot=500, seed=1)


def test_bootstrap_ci_narrows_with_more_tasks():
    rng = np.random.default_rng(0)
    few = rng.binomial(32, 0.3, size=30)
    many = rng.binomial(32, 0.3, size=3000)
    _, lo_few, hi_few = bootstrap_ci(32, few, k=4, n_boot=500)
    _, lo_many, hi_many = bootstrap_ci(32, many, k=4, n_boot=500)
    assert (hi_many - lo_many) < (hi_few - lo_few)
