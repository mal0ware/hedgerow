import numpy as np
import pytest

from hedgerow import black_scholes as bs


def test_known_value_hull() -> None:
    # Hull, Options, Futures and Other Derivatives, the standard worked example:
    # S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5 gives c = 4.76, p = 0.81.
    assert bs.price(42.0, 40.0, 0.5, 0.2, 0.10, "call") == pytest.approx(4.76, abs=0.005)
    assert bs.price(42.0, 40.0, 0.5, 0.2, 0.10, "put") == pytest.approx(0.81, abs=0.005)


def test_derman_kamal_option_value_and_vega() -> None:
    # Derman and Kamal (1999): S = K = 100, T = 1/12, r = 5%, sigma = 20%.
    # Printed value 2.512 and vega 0.115 per volatility point.
    assert bs.price(100.0, 100.0, 1 / 12, 0.2, 0.05, "call") == pytest.approx(2.512, abs=0.001)
    assert bs.vega(100.0, 100.0, 1 / 12, 0.2, 0.05) / 100 == pytest.approx(0.115, abs=0.001)


@pytest.mark.parametrize("spot", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_put_call_parity(spot: float, rate: float) -> None:
    tau, vol, strike = 0.7, 0.3, 100.0
    call = bs.price(spot, strike, tau, vol, rate, "call")
    put = bs.price(spot, strike, tau, vol, rate, "put")
    assert call - put == pytest.approx(spot - strike * np.exp(-rate * tau), abs=1e-10)
    assert bs.delta(spot, strike, tau, vol, rate, "call") - bs.delta(
        spot, strike, tau, vol, rate, "put"
    ) == pytest.approx(1.0)


def test_greeks_match_finite_differences() -> None:
    s, k, tau, vol, r = 103.0, 100.0, 0.4, 0.25, 0.03
    h = 1e-3
    up = bs.price(s + h, k, tau, vol, r)
    mid = bs.price(s, k, tau, vol, r)
    dn = bs.price(s - h, k, tau, vol, r)
    assert bs.delta(s, k, tau, vol, r) == pytest.approx((up - dn) / (2 * h), abs=1e-7)
    assert bs.gamma(s, k, tau, vol, r) == pytest.approx((up - 2 * mid + dn) / h**2, abs=1e-5)
    v_fd = (bs.price(s, k, tau, vol + h, r) - bs.price(s, k, tau, vol - h, r)) / (2 * h)
    assert bs.vega(s, k, tau, vol, r) == pytest.approx(v_fd, abs=1e-4)


def test_expiry_limits() -> None:
    spots = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(bs.price(spots, 100.0, 0.0, 0.2), [0.0, 0.0, 10.0])
    np.testing.assert_allclose(bs.delta(spots, 100.0, 0.0, 0.2), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(bs.gamma(spots, 100.0, 0.0, 0.2), 0.0)
    np.testing.assert_allclose(bs.vega(spots, 100.0, 0.0, 0.2), 0.0)


def test_deep_in_and_out_of_the_money_limits() -> None:
    assert bs.delta(1e4, 100.0, 0.5, 0.2) == pytest.approx(1.0)
    assert bs.delta(1.0, 100.0, 0.5, 0.2) == pytest.approx(0.0)
    assert bs.price(1e4, 100.0, 0.5, 0.2) == pytest.approx(1e4 - 100.0)


def test_price_increases_with_volatility() -> None:
    vols = np.linspace(0.05, 1.0, 30)
    prices = bs.price(100.0, 105.0, 0.25, vols)
    assert np.all(np.diff(prices) > 0)
