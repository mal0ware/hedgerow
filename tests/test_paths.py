import numpy as np
import pytest

from hedgerow import black_scholes as bs
from hedgerow.paths import (
    HestonParams,
    cir_mean,
    cir_variance,
    expected_integrated_variance,
    gbm_paths,
    heston_paths,
)

T = 30 / 365


def test_gbm_same_seed_same_paths() -> None:
    a = gbm_paths(500, 30, T, 0.2, seed=7)
    b = gbm_paths(500, 30, T, 0.2, seed=7)
    c = gbm_paths(500, 30, T, 0.2, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gbm_moments() -> None:
    n = 200_000
    paths = gbm_paths(n, 10, 1.0, 0.3, seed=1, s0=100.0, drift=0.05)
    log_ret = np.log(paths[:, -1] / 100.0)
    assert log_ret.mean() == pytest.approx(0.05 - 0.045, abs=4 * 0.3 / np.sqrt(n))
    assert log_ret.std() == pytest.approx(0.3, rel=0.01)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=0.005)


def test_gbm_increments_are_independent_of_grid() -> None:
    # Realised variance on the grid recovers sigma^2 T.
    paths = gbm_paths(20_000, 50, 1.0, 0.2, seed=3)
    rv = (np.diff(np.log(paths), axis=1) ** 2).sum(axis=1)
    assert rv.mean() == pytest.approx(0.04, rel=0.01)


def test_gbm_call_price_matches_black_scholes() -> None:
    n = 400_000
    paths = gbm_paths(n, 1, T, 0.2, seed=11)
    pay = np.maximum(paths[:, -1] - 100.0, 0.0)
    assert pay.mean() == pytest.approx(
        bs.price(100.0, 100.0, T, 0.2), abs=4 * pay.std() / np.sqrt(n)
    )


def test_heston_same_seed_same_paths() -> None:
    p = HestonParams()
    s1, v1 = heston_paths(300, 30, T, p, seed=5)
    s2, v2 = heston_paths(300, 30, T, p, seed=5)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(v1, v2)
    assert s1.shape == v1.shape == (300, 31)
    assert np.all(v1 >= 0.0)


@pytest.mark.parametrize("params", [HestonParams(), HestonParams(kappa=2.0, theta=0.06, xi=0.5)])
def test_cir_mean_and_variance(params: HestonParams) -> None:
    n = 400_000
    _, v = heston_paths(n, 5, 0.5, params, seed=2)
    for k in (1, 5):
        t = 0.5 * k / 5
        var = cir_variance(t, params)
        assert v[:, k].mean() == pytest.approx(cir_mean(t, params), abs=5 * np.sqrt(var / n))
        assert v[:, k].var() == pytest.approx(var, rel=0.05)


def test_heston_integrated_variance_matches_variance_swap() -> None:
    # Buehler et al. (2019), Eq. (5.4): S2_0 = L(0, v0).
    params = HestonParams()
    n = 200_000
    _, v = heston_paths(n, 30, T, params, seed=4, substeps=8)
    # trapezoid rule on the reporting grid
    integral = (0.5 * (v[:, :-1] + v[:, 1:])).sum(axis=1) * T / 30
    assert integral.mean() == pytest.approx(
        expected_integrated_variance(T, params), abs=5 * integral.std() / np.sqrt(n)
    )


def test_heston_price_is_close_to_a_martingale() -> None:
    n = 400_000
    s, _ = heston_paths(n, 30, T, HestonParams(), seed=6)
    for k in (10, 30):
        assert s[:, k].mean() == pytest.approx(100.0, abs=5 * s[:, k].std() / np.sqrt(n) + 0.02)


def test_heston_reduces_to_black_scholes_when_variance_is_frozen() -> None:
    # No vol of vol and V0 at its mean: the variance stays put and S is lognormal.
    params = HestonParams(kappa=1.0, theta=0.04, xi=1e-6, rho=0.0, v0=0.04)
    n = 200_000
    s, v = heston_paths(n, 30, T, params, seed=9)
    np.testing.assert_allclose(v, 0.04, atol=1e-6)
    pay = np.maximum(s[:, -1] - 100.0, 0.0)
    assert pay.mean() == pytest.approx(
        bs.price(100.0, 100.0, T, 0.2), abs=4 * pay.std() / np.sqrt(n)
    )


def test_heston_call_price_matches_buehler() -> None:
    # Buehler et al. (2019), Section 5.2: risk-neutral price q = 1.69 for the
    # at-the-money call, T = 30/365, with the printed parameters.
    n = 1_000_000
    s, _ = heston_paths(n, 30, T, HestonParams(), seed=10)
    pay = np.maximum(s[:, -1] - 100.0, 0.0)
    assert pay.mean() == pytest.approx(1.69, abs=0.02)


def test_substeps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        heston_paths(10, 5, T, HestonParams(), seed=0, substeps=0)
