import numpy as np
import pytest

from hedgerow import black_scholes as bs
from hedgerow.hedging import payoff
from hedgerow.heston import (
    GridSpec,
    atm_implied_vol,
    call_price,
    cf_exponents,
    delta_table,
    fourier_slice,
    implied_vol,
)
from hedgerow.paths import HestonParams, heston_paths

PAPER = HestonParams()
T = 30.0 / 365.0
MID = GridSpec(log2_n=15)


def test_characteristic_function_is_one_at_zero_and_is_a_martingale() -> None:
    # exp(A + B V) must equal 1 at u = 0 (total mass) and at u = -i (E[S_T] = S_t).
    a, b = cf_exponents(np.array([0j, -1j]), T, PAPER)
    np.testing.assert_allclose(a, 0.0, atol=1e-12)
    np.testing.assert_allclose(b, 0.0, atol=1e-12)


def test_price_matches_the_deep_hedging_paper() -> None:
    # Buehler et al. (2019), Section 5.2, print q = 1.69 for this contract.
    assert call_price(100.0, 100.0, T, PAPER.v0, PAPER, MID) == pytest.approx(1.69, abs=0.005)


def test_price_is_close_to_the_simulator() -> None:
    # The simulator holds the variance fixed inside each daily step, so it is
    # not exactly the continuous model. The two prices agree to about 1%.
    prices, _ = heston_paths(400_000, 30, T, PAPER, seed=11)
    mc = float(payoff(prices[:, -1], 100.0).mean())
    assert call_price(100.0, 100.0, T, PAPER.v0, PAPER, MID) == pytest.approx(mc, rel=0.015)


def test_small_vol_of_variance_gives_black_scholes() -> None:
    calm = HestonParams(xi=0.01, rho=0.0)
    for strike in (95.0, 100.0, 105.0):
        expected = float(bs.price(100.0, strike, T, 0.2)[()])
        assert call_price(100.0, strike, T, 0.04, calm, MID) == pytest.approx(expected, abs=2e-4)
    s = fourier_slice(T, np.array([0.04]), calm, MID)
    delta = float(np.interp(np.log(105.0 / 100.0), s.x, s.delta[0]))
    assert delta == pytest.approx(float(bs.delta(100.0, 105.0, T, 0.2)[()]), abs=2e-4)


def test_delta_is_the_derivative_of_the_price() -> None:
    s = fourier_slice(T, np.array([0.04, 0.2]), PAPER, MID)
    h = 1e-3
    for row, v in enumerate((0.04, 0.2)):
        for strike in (97.0, 100.0, 104.0):
            up = call_price(100.0 * (1 + h), strike, T, v, PAPER, MID)
            down = call_price(100.0 * (1 - h), strike, T, v, PAPER, MID)
            table = float(np.interp(np.log(strike / 100.0), s.x, s.delta[row]))
            assert table == pytest.approx((up - down) / (200.0 * h), abs=2e-3)


def test_variance_sensitivity_is_the_derivative_of_the_price() -> None:
    s = fourier_slice(T, np.array([0.04]), PAPER, MID)
    dv = 5e-4
    up = call_price(100.0, 100.0, T, 0.04 + dv, PAPER, MID)
    down = call_price(100.0, 100.0, T, 0.04 - dv, PAPER, MID)
    table = 100.0 * float(np.interp(0.0, s.x, s.dprice_dv_over_spot[0]))
    assert table == pytest.approx((up - down) / (2 * dv), rel=1e-3)


def test_implied_vol_round_trip() -> None:
    price = float(bs.price(100.0, 100.0, T, 0.17)[()])
    assert implied_vol(price, 100.0, 100.0, T) == pytest.approx(0.17, abs=1e-9)
    # The Heston call is worth less than the Black-Scholes call at 20%.
    assert 0.13 < atm_implied_vol(100.0, 100.0, T, PAPER, MID) < 0.16


def test_table_lookup_matches_direct_inversion() -> None:
    spec = GridSpec(log2_n=14, n_money=801, n_var=64)
    table = delta_table(PAPER, T, 3, spec)
    rng = np.random.default_rng(3)
    var = np.array([0.0, 0.003, 0.04, 0.3])
    spot = 100.0 * np.exp(rng.uniform(-0.08, 0.08, size=var.size))
    prices = np.tile(spot[:, None], (1, 4))
    variances = np.tile(var[:, None], (1, 4))
    got = table.deltas(prices, variances, 100.0)
    got_mv = table.deltas(prices, variances, 100.0, min_var=True)
    for k in range(3):
        s = fourier_slice(T * (1 - k / 3), var, PAPER, spec)
        for i in range(var.size):
            x = float(np.log(100.0 / spot[i]))
            d = float(np.interp(x, s.x, s.delta[i]))
            dv = float(np.interp(x, s.x, s.dprice_dv_over_spot[i]))
            assert got[i, k] == pytest.approx(d, abs=0.02)
            assert got_mv[i, k] == pytest.approx(d + PAPER.rho * PAPER.xi * dv, abs=0.02)


def test_table_refuses_the_wrong_number_of_dates() -> None:
    table = delta_table(PAPER, T, 3, GridSpec.coarse())
    with pytest.raises(ValueError, match="rebalance dates"):
        table.deltas(np.full((2, 6), 100.0), np.full((2, 6), 0.04), 100.0)


def test_minimum_variance_delta_has_the_smaller_hedging_error() -> None:
    # With negative correlation the stock-only hedge should hold fewer shares
    # than dC/dS, and doing so should cut the spread of the hedging error.
    spec = GridSpec(log2_n=14, n_money=601, n_var=48)
    table = delta_table(PAPER, T, 30, spec)
    prices, variances = heston_paths(20_000, 30, T, PAPER, seed=5)
    moves = np.diff(prices, axis=1)
    liability = payoff(prices[:, -1], 100.0)
    plain = table.deltas(prices, variances, 100.0)
    min_var = table.deltas(prices, variances, 100.0, min_var=True)
    assert min_var[:, 0].mean() < plain[:, 0].mean()
    err_plain = (plain * moves).sum(axis=1) - liability
    err_mv = (min_var * moves).sum(axis=1) - liability
    assert err_mv.std() < 0.85 * err_plain.std()
    # Any stock-only strategy has the option value as its mean loss.
    assert -err_mv.mean() == pytest.approx(liability.mean(), abs=0.05)
