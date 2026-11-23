import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from hedgerow.risk import cvar, normal_cvar, oce_objective, paired_block_difference


def test_cvar_is_the_tail_mean() -> None:
    x = np.arange(1.0, 101.0)
    assert cvar(x, 0.9) == pytest.approx(np.mean(x[-10:]))
    assert cvar(x, 0.5) == pytest.approx(np.mean(x[-50:]))
    assert cvar(x, 0.0) == pytest.approx(x.mean())


def test_cvar_is_the_minimum_of_the_oce_objective() -> None:
    # Buehler et al. (2019), Eq. (3.5) with Example 3.9.
    rng = np.random.default_rng(0)
    x = rng.standard_t(5, size=20_001)
    for alpha in (0.5, 0.9, 0.99):
        best = minimize_scalar(lambda w, a=alpha: oce_objective(x, w, a), bounds=(-20, 20), method="bounded")
        assert cvar(x, alpha) == pytest.approx(best.fun, abs=1e-6)
        assert cvar(x, alpha) <= oce_objective(x, 0.0, alpha) + 1e-12


def test_cvar_matches_normal_closed_form() -> None:
    rng = np.random.default_rng(1)
    x = 1.0 + 2.0 * rng.standard_normal(2_000_000)
    for alpha in (0.5, 0.95, 0.99):
        assert cvar(x, alpha) == pytest.approx(normal_cvar(1.0, 2.0, alpha), rel=0.01)


def test_cvar_properties() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    y = rng.standard_normal(5000)
    # cash invariance and positive homogeneity
    assert cvar(x + 3.0, 0.9) == pytest.approx(cvar(x, 0.9) + 3.0)
    assert cvar(2.0 * x, 0.9) == pytest.approx(2.0 * cvar(x, 0.9))
    # monotone in alpha, and subadditive
    assert cvar(x, 0.5) <= cvar(x, 0.9) <= cvar(x, 0.99) <= x.max()
    assert cvar(x + y, 0.9) <= cvar(x, 0.9) + cvar(y, 0.9) + 1e-12


def test_cvar_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        cvar(np.ones(3), 1.0)


def test_paired_block_difference() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(40_000)
    mean, se = paired_block_difference(a + 0.5, a, 0.5)
    assert mean == pytest.approx(0.5)
    assert se == pytest.approx(0.0, abs=1e-12)
    mean2, se2 = paired_block_difference(a, rng.standard_normal(40_000), 0.5)
    assert abs(mean2) < 4 * se2
