"""Discrete hedging: the accounting, the classical strategies and their theory.

Conventions
-----------
The hedger is short one European option and holds ``deltas[:, k]`` shares over
the interval ``[t_k, t_{k+1}]``. Trading ``n`` shares at price ``S`` costs
``cost_rate * |n| * S`` (Buehler et al. 2019, Eq. (5.7)). ``cost_rate`` is the
one-way fraction: half of Leland's round-trip coefficient. The first trade,
from zero shares, pays costs. The position is not charged for liquidation at
expiry, as in the numerical examples of Buehler et al. (Section 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hedgerow import black_scholes as bs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HedgeResult:
    """Per-path outcome of a hedged short option.

    ``pnl`` is terminal wealth after paying the payoff. ``cost`` is the total
    transaction cost paid and ``turnover`` the total number of shares traded;
    both are undiscounted sums.
    """

    pnl: FloatArray
    cost: FloatArray
    turnover: FloatArray


def payoff(terminal: FloatArray, strike: float, kind: bs.OptionKind = "call") -> FloatArray:
    """European option payoff."""
    if kind == "call":
        return np.asarray(np.maximum(terminal - strike, 0.0), dtype=np.float64)
    return np.asarray(np.maximum(strike - terminal, 0.0), dtype=np.float64)


def time_grid(n_steps: int, maturity: float) -> FloatArray:
    """Time to expiry at each of the ``n_steps`` rebalance dates."""
    return np.asarray(maturity * (1.0 - np.arange(n_steps) / n_steps), dtype=np.float64)


def hedge_pnl(
    prices: FloatArray,
    deltas: FloatArray,
    liability: FloatArray,
    premium: float = 0.0,
    cost_rate: float = 0.0,
    rate: float = 0.0,
    maturity: float = 1.0,
) -> HedgeResult:
    """Terminal wealth of a self-financing hedge of a short liability.

    With ``rate == 0`` this is Buehler et al. (2019), Eq. (2.1):
    ``PL = -Z + p0 + sum_k delta_k (S_{k+1} - S_k) - C_T``.
    With a positive rate the cash account earns interest between trades.

    Parameters
    ----------
    prices : shape ``(n_paths, n_steps + 1)``.
    deltas : shape ``(n_paths, n_steps)``; holdings over each interval.
    liability : shape ``(n_paths,)``; the payoff owed at expiry.
    premium : cash received at time zero.
    """
    n_paths, n_cols = prices.shape
    n_steps = n_cols - 1
    if deltas.shape != (n_paths, n_steps):
        raise ValueError(f"deltas has shape {deltas.shape}, expected {(n_paths, n_steps)}")
    growth = float(np.exp(rate * maturity / n_steps))
    trades = np.diff(deltas, axis=1, prepend=0.0)
    costs = cost_rate * np.abs(trades) * prices[:, :-1]
    cash = np.full(n_paths, premium, dtype=np.float64)
    for k in range(n_steps):
        cash = cash - trades[:, k] * prices[:, k] - costs[:, k]
        cash = cash * growth
    pnl = cash + deltas[:, -1] * prices[:, -1] - liability
    return HedgeResult(
        pnl=np.asarray(pnl, dtype=np.float64),
        cost=np.asarray(costs.sum(axis=1), dtype=np.float64),
        turnover=np.asarray(np.abs(trades).sum(axis=1), dtype=np.float64),
    )


def bs_deltas(
    prices: FloatArray,
    strike: float,
    maturity: float,
    vol: float | FloatArray,
    rate: float = 0.0,
    kind: bs.OptionKind = "call",
) -> FloatArray:
    """Black-Scholes delta at every rebalance date.

    ``vol`` is a scalar, or an array of shape ``(n_paths, n_steps)`` for a
    hedger who plugs a time-varying volatility into the formula.
    """
    n_steps = prices.shape[1] - 1
    tau = time_grid(n_steps, maturity)[None, :]
    return bs.delta(prices[:, :-1], strike, tau, vol, rate, kind)


def leland_vol(
    vol: float,
    cost_rate: float,
    dt: float,
    position: Literal["short", "long"] = "short",
) -> float:
    """Leland's adjusted volatility.

    For the hedger of a short option, in the notation of Kabanov and Safarian
    (1997), Eq. (2), with one-way cost fraction k and revision interval dt:

        sigma_hat^2 = sigma^2 (1 + 2 sqrt(2 / pi) k / (sigma sqrt(dt))).

    Leland (1985) writes the same thing with the round-trip cost 2k. For a long
    option the sign of the correction flips, and the formula breaks down when
    the bracket turns negative.
    """
    leland_number = 2.0 * np.sqrt(2.0 / np.pi) * cost_rate / (vol * np.sqrt(dt))
    sign = 1.0 if position == "short" else -1.0
    bracket = 1.0 + sign * leland_number
    if bracket <= 0.0:
        raise ValueError("Leland's adjusted variance is not positive for these inputs")
    return float(vol * np.sqrt(bracket))


def boyle_emanuel_scale(spot: float, strike: float, tau: float, vol: float, dt: float) -> float:
    """lambda = Gamma S^2 sigma^2 dt / 2, the scale of the one-interval hedging error.

    Over one short interval a hedged long option earns approximately
    ``lambda * (u^2 - 1)`` with ``u`` standard normal: a shifted chi-square
    with one degree of freedom, mean zero and variance ``2 lambda^2``.
    """
    return float(0.5 * bs.gamma(spot, strike, tau, vol)[()] * spot**2 * vol**2 * dt)


def discrete_hedge_std(
    s0: float,
    strike: float,
    maturity: float,
    vol: float,
    n_steps: int,
    quad_points: int = 200,
) -> float:
    """Leading-order standard deviation of the total discrete hedging error.

    Sums the one-interval variances ``2 lambda_k^2`` over the rebalance dates,
    taking the expectation of ``(Gamma S^2)^2`` over the lognormal law of
    ``S_k`` by Gauss-Hermite quadrature. Zero rates and zero drift. Errors in
    different intervals are uncorrelated to this order, so the variances add.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(quad_points)
    weights = weights / weights.sum()
    dt = maturity / n_steps
    total = 0.0
    for k in range(n_steps):
        t = k * dt
        tau = maturity - t
        s = s0 * np.exp(-0.5 * vol**2 * t + vol * np.sqrt(t) * nodes)
        dollar_gamma = bs.gamma(s, strike, tau, vol) * s**2
        total += 0.5 * (vol**2 * dt) ** 2 * float(np.sum(weights * dollar_gamma**2))
    return float(np.sqrt(total))


def derman_kamal_std(
    s0: float, strike: float, maturity: float, vol: float, n_steps: int, rate: float = 0.0
) -> float:
    """sqrt(pi / 4) * vega * sigma / sqrt(N): Derman and Kamal (1999), Eq. (1)."""
    v = float(bs.vega(s0, strike, maturity, vol, rate)[()])
    return float(np.sqrt(np.pi / 4.0) * v * vol / np.sqrt(n_steps))
