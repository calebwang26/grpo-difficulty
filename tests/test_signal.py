import numpy as np
import pytest

from grpo_difficulty.signal import (
    mean_squared_advantage,
    signal_fraction,
    simulated_signal_fraction,
)


def test_known_values():
    assert signal_fraction(0.5, 2) == pytest.approx(0.5)
    assert signal_fraction(0.5, 8) == pytest.approx(1 - 2 * 0.5**8)
    assert signal_fraction(0.1, 4) == pytest.approx(1 - 0.1**4 - 0.9**4)


def test_zero_signal_at_the_extremes():
    for G in (1, 4, 8, 16):
        assert signal_fraction(0.0, G) == 0.0
        assert signal_fraction(1.0, G) == 0.0


def test_group_of_one_never_has_signal():
    p = np.linspace(0, 1, 11)
    assert np.allclose(signal_fraction(p, 1), 0.0)


def test_symmetric_and_peaks_at_half():
    p = np.linspace(0, 1, 101)
    for G in (4, 8, 16):
        f = signal_fraction(p, G)
        assert np.allclose(f, f[::-1])
        assert np.argmax(f) == 50


def test_bigger_groups_carry_signal_more_often():
    p = np.linspace(0.01, 0.99, 50)
    assert np.all(signal_fraction(p, 8) > signal_fraction(p, 4))
    assert np.all(signal_fraction(p, 16) > signal_fraction(p, 8))


@pytest.mark.parametrize("p,G", [(0.05, 8), (0.5, 4), (0.9, 16)])
def test_formula_matches_simulation(p, G):
    assert simulated_signal_fraction(p, G, trials=200000, seed=1) == pytest.approx(
        signal_fraction(p, G), abs=0.005
    )


def test_mean_squared_advantage_matches_simulation():
    p, G = 0.3, 8
    rng = np.random.default_rng(0)
    rewards = (rng.random((200000, G)) < p).astype(float)
    advantages = rewards - rewards.mean(axis=1, keepdims=True)
    assert (advantages**2).mean() == pytest.approx(mean_squared_advantage(p, G), abs=0.002)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        signal_fraction(1.2, 4)
    with pytest.raises(ValueError):
        signal_fraction(0.5, 0)
