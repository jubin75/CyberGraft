import pytest

from cybergraft.experiment.bypass_experiment import bootstrap_ci, permutation_p, summary_stats


def test_summary_stats_positive_effect():
    stats = summary_stats([1.0, 1.1, 0.9, 1.0, 1.1])
    assert stats["mean"] == pytest.approx(1.02, abs=1e-6)
    assert stats["std"] > 0.0
    assert stats["cohen_d"] > 0.0
    assert stats["ci_lo"] > 0.0
    assert stats["ci_lo"] < stats["mean"] < stats["ci_hi"]
    assert stats["n"] == 5


def test_summary_stats_insignificant_effect():
    stats = summary_stats([-0.2, 0.1, -0.1, 0.0, 0.2])
    assert stats["mean"] == pytest.approx(0.0, abs=1e-6)
    assert stats["ci_lo"] < 0.0
    assert stats["n"] == 5


def test_summary_stats_zero_std():
    stats = summary_stats([0.5, 0.5, 0.5])
    assert stats["std"] == 0.0
    assert stats["ci_lo"] == 0.5
    assert stats["ci_hi"] == 0.5
    assert stats["cohen_d"] == float("inf")


def test_bootstrap_ci_positive_effect():
    ci_lo, ci_hi = bootstrap_ci([1.0, 1.1, 0.9, 1.0, 1.1], seed=0)
    assert ci_lo > 0.0
    assert ci_hi > 0.0
    assert ci_lo < ci_hi


def test_permutation_p_positive_effect():
    p = permutation_p([0.5, 0.6, 0.5, 0.5, 0.5], seed=0)
    assert p < 0.05


def test_permutation_p_zero_mean_effect():
    p = permutation_p([-0.1, 0.1, -0.2, 0.1, 0.0], seed=0)
    assert p > 0.05
