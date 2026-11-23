import numpy as np
import pytest

from hedgerow import black_scholes as bs
from hedgerow.hedging import (
    boyle_emanuel_scale,
    bs_deltas,
    derman_kamal_std,
    discrete_hedge_std,
    hedge_pnl,
    leland_vol,
    payoff,
    time_grid,
)
from hedgerow.paths import gbm_paths

S0 = K = 100.0
VOL = 0.2
T = 30 / 365


def _bs_error(
    n_steps: int, n_paths: int, seed: int, cost: float = 0.0, hedge_vol: float = VOL
) -> np.ndarray:
    paths = gbm_paths(n_paths, n_steps, T, VOL, seed)
    deltas = bs_deltas(paths, K, T, hedge_vol)
    premium = float(bs.price(S0, K, T, hedge_vol))
    return hedge_pnl(paths, deltas, payoff(paths[:, -1], K), premium, cost, maturity=T).pnl


# ---------------------------------------------------------------- accounting


def test_time_grid() -> None:
    np.testing.assert_allclose(time_grid(4, 1.0), [1.0, 0.75, 0.5, 0.25])


def test_hand_built_path() -> None:
    prices = np.array([[100.0, 110.0, 105.0]])
    deltas = np.array([[0.5, 1.0]])
    liability = np.array([5.0])
    res = hedge_pnl(prices, deltas, liability, premium=3.0, cost_rate=0.01, maturity=1.0)
    gains = 0.5 * 10.0 + 1.0 * (-5.0)
    cost = 0.01 * (0.5 * 100.0 + 0.5 * 110.0)
    assert res.pnl[0] == pytest.approx(3.0 + gains - cost - 5.0)
    assert res.cost[0] == pytest.approx(cost)
    assert res.turnover[0] == pytest.approx(1.0)


def test_interest_accrues_on_cash() -> None:
    prices = np.array([[100.0, 100.0, 100.0]])
    deltas = np.zeros((1, 2))
    res = hedge_pnl(prices, deltas, np.array([0.0]), premium=10.0, rate=0.05, maturity=2.0)
    assert res.pnl[0] == pytest.approx(10.0 * np.exp(0.10))


def test_matches_gains_process_formula_at_zero_rate() -> None:
    # Buehler et al. (2019), Eq. (2.1).
    paths = gbm_paths(200, 12, T, VOL, seed=1)
    rng = np.random.default_rng(0)
    deltas = rng.uniform(-1, 2, size=(200, 12))
    z = payoff(paths[:, -1], K)
    eps = 0.003
    res = hedge_pnl(paths, deltas, z, premium=1.5, cost_rate=eps, maturity=T)
    gains = (deltas * np.diff(paths, axis=1)).sum(axis=1)
    trades = np.diff(deltas, axis=1, prepend=0.0)
    costs = (eps * np.abs(trades) * paths[:, :-1]).sum(axis=1)
    np.testing.assert_allclose(res.pnl, -z + 1.5 + gains - costs, atol=1e-10)


def test_static_full_hedge_of_a_forward_is_exact() -> None:
    paths = gbm_paths(50, 5, T, VOL, seed=2)
    res = hedge_pnl(paths, np.ones((50, 5)), paths[:, -1] - K, premium=S0 - K, maturity=T)
    np.testing.assert_allclose(res.pnl, 0.0, atol=1e-10)


def test_shape_check() -> None:
    with pytest.raises(ValueError):
        hedge_pnl(np.ones((3, 5)), np.ones((3, 5)), np.zeros(3))


def test_put_and_call_hedging_errors_are_identical() -> None:
    paths = gbm_paths(1000, 20, T, VOL, seed=3, drift=0.05)
    out = []
    for kind in ("call", "put"):
        deltas = bs_deltas(paths, K, T, VOL, 0.05, kind)  # type: ignore[arg-type]
        prem = float(bs.price(S0, K, T, VOL, 0.05, kind))  # type: ignore[arg-type]
        liab = payoff(paths[:, -1], K, kind)  # type: ignore[arg-type]
        out.append(hedge_pnl(paths, deltas, liab, prem, rate=0.05, maturity=T).pnl)
    np.testing.assert_allclose(out[0], out[1], atol=1e-9)


# ------------------------------------------------- discrete hedging, no costs


def test_mean_error_is_zero() -> None:
    n = 100_000
    pnl = _bs_error(30, n, seed=4)
    assert pnl.mean() == pytest.approx(0.0, abs=4 * pnl.std() / np.sqrt(n))


def test_error_shrinks_as_one_over_root_n() -> None:
    steps = np.array([8, 16, 32, 64, 128, 256])
    stds = np.array([_bs_error(int(n), 40_000, seed=5).std() for n in steps])
    slope = np.polyfit(np.log(steps), np.log(stds), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.02)
    # quadrupling the rebalances halves the error
    assert stds[2] / stds[4] == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("n_steps", [30, 120, 480])
def test_error_std_matches_summed_one_interval_variances(n_steps: int) -> None:
    sim = _bs_error(n_steps, 60_000, seed=6).std()
    assert sim == pytest.approx(discrete_hedge_std(S0, K, T, VOL, n_steps), rel=0.04)


def test_derman_kamal_published_simulation() -> None:
    # Derman and Kamal (1999), Table 1: one-month at-the-money option, r = 5%,
    # sigma = 20%, 50,000 paths. Standard deviation 0.41 with 21 rebalances and
    # 0.20 with 84. Their rule of thumb gives 0.443 and 0.222.
    # This simulation gives 0.428 and 0.218: within 2 to 4% of their rule and
    # 4 to 9% above their printed simulation. PAPER.md records the difference.
    mat = 1 / 12
    for n_steps, printed_sim, printed_rule in ((21, 0.41, 0.443), (84, 0.20, 0.222)):
        paths = gbm_paths(200_000, n_steps, mat, VOL, seed=7, drift=0.05)
        deltas = bs_deltas(paths, K, mat, VOL, 0.05)
        prem = float(bs.price(S0, K, mat, VOL, 0.05))
        pnl = hedge_pnl(paths, deltas, payoff(paths[:, -1], K), prem, rate=0.05, maturity=mat).pnl
        assert pnl.mean() == pytest.approx(0.0, abs=0.005)
        assert pnl.std() == pytest.approx(printed_sim, rel=0.10)
        assert pnl.std() == pytest.approx(printed_rule, rel=0.04)
        assert pnl.std() == pytest.approx(discrete_hedge_std(S0, K, mat, VOL, n_steps), rel=0.04)
        assert derman_kamal_std(S0, K, mat, VOL, n_steps, 0.05) == pytest.approx(
            printed_rule, abs=0.002
        )


def test_one_interval_error_is_a_shifted_chi_square() -> None:
    # Hold the Black-Scholes hedge for one short interval. The gain of
    # long option, short delta shares is lambda (u^2 - 1) to leading order.
    tau, dt, n = 0.25, 1 / 365, 400_000
    rng = np.random.default_rng(8)
    u = rng.standard_normal(n)
    s1 = S0 * np.exp(-0.5 * VOL**2 * dt + VOL * np.sqrt(dt) * u)
    gain = (
        bs.price(s1, K, tau - dt, VOL)
        - bs.price(S0, K, tau, VOL)
        - bs.delta(S0, K, tau, VOL) * (s1 - S0)
    )
    lam = boyle_emanuel_scale(S0, K, tau, VOL, dt)
    predicted = lam * (u**2 - 1.0)
    assert np.corrcoef(gain, predicted)[0, 1] > 0.995
    assert gain.mean() == pytest.approx(0.0, abs=5 * gain.std() / np.sqrt(n))
    assert gain.std() == pytest.approx(np.sqrt(2.0) * lam, rel=0.03)
    skew = float(np.mean((gain - gain.mean()) ** 3) / gain.std() ** 3)
    assert skew == pytest.approx(np.sqrt(8.0), rel=0.1)
    # most outcomes are small losses: P(u^2 < 1) = 68.3%
    assert np.mean(gain < 0) == pytest.approx(0.683, abs=0.01)


# -------------------------------------------------------------------- Leland


def test_leland_vol_formula() -> None:
    k, dt = 0.005, 1 / 365
    expected = VOL * np.sqrt(1 + 2 * np.sqrt(2 / np.pi) * k / (VOL * np.sqrt(dt)))
    assert leland_vol(VOL, k, dt) == pytest.approx(expected)
    assert leland_vol(VOL, 0.0, dt) == pytest.approx(VOL)
    assert leland_vol(VOL, k, dt) > VOL > leland_vol(VOL, 0.001, dt, "long")


def test_leland_vol_scale_invariance() -> None:
    # The correction depends on k / sqrt(dt) only.
    assert leland_vol(VOL, 0.002, 1 / 100) == pytest.approx(leland_vol(VOL, 0.001, 1 / 400))


def test_leland_vol_long_breaks_down() -> None:
    with pytest.raises(ValueError):
        leland_vol(VOL, 0.05, 1 / 365, "long")


def test_expected_cost_of_one_rebalance_is_the_extra_theta() -> None:
    # Leland's argument: over one interval the expected cost of re-hedging,
    # k S E|d delta| = k Gamma S^2 sigma sqrt(2 dt / pi), equals the extra time
    # decay Gamma S^2 (sigma_hat^2 - sigma^2) dt / 2 earned by pricing at sigma_hat.
    k, dt, tau = 0.002, 1 / 365, 0.25
    sig_hat = leland_vol(VOL, k, dt)
    rng = np.random.default_rng(9)
    s1 = S0 * np.exp(-0.5 * VOL**2 * dt + VOL * np.sqrt(dt) * rng.standard_normal(400_000))
    d_delta = bs.delta(s1, K, tau - dt, sig_hat) - bs.delta(S0, K, tau, sig_hat)
    cost = k * s1 * np.abs(d_delta)
    extra_theta = 0.5 * bs.gamma(S0, K, tau, sig_hat) * S0**2 * (sig_hat**2 - VOL**2) * dt
    assert cost.mean() == pytest.approx(float(extra_theta), rel=0.03)


def test_leland_hedge_is_roughly_self_financing_under_costs() -> None:
    # Charge the Leland price, hedge with the Leland delta, exclude the entry
    # trade (as Kabanov and Safarian do): the mean error is small. The
    # Black-Scholes hedge charged the Black-Scholes price loses the costs.
    k, n_steps, n = 0.002, 60, 100_000
    paths = gbm_paths(n, n_steps, T, VOL, seed=10)
    z = payoff(paths[:, -1], K)
    sig_hat = leland_vol(VOL, k, T / n_steps)
    out = {}
    for name, vol in (("bs", VOL), ("leland", sig_hat)):
        deltas = bs_deltas(paths, K, T, vol)
        res = hedge_pnl(paths, deltas, z, float(bs.price(S0, K, T, vol)), k, maturity=T)
        entry = k * deltas[:, 0] * S0
        out[name] = res.pnl + entry
    premium = float(bs.price(S0, K, T, VOL))
    assert abs(out["leland"].mean()) < 0.02 * premium
    assert out["bs"].mean() < -0.1 * premium
    assert out["leland"].std() < out["bs"].std()


def test_leland_trades_less_than_black_scholes() -> None:
    k, n_steps = 0.005, 30
    paths = gbm_paths(20_000, n_steps, T, VOL, seed=11)
    z = payoff(paths[:, -1], K)
    turn = {}
    for name, vol in (("bs", VOL), ("leland", leland_vol(VOL, k, T / n_steps))):
        turn[name] = hedge_pnl(paths, bs_deltas(paths, K, T, vol), z, 0.0, k, maturity=T).turnover
    assert turn["leland"].mean() < turn["bs"].mean()
