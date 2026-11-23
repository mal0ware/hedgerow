"""Black-Scholes prices and Greeks for European calls and puts.

No dividends. ``tau`` is time to expiry in years, ``rate`` is the continuously
compounded risk-free rate. Everything broadcasts over numpy arrays.

At ``tau == 0`` the functions return the expiry limits: the payoff for the
price, a step function for the delta, zero for gamma and vega.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]
OptionKind = Literal["call", "put"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def norm_pdf(x: ArrayLike) -> FloatArray:
    """Standard normal density."""
    z = np.asarray(x, dtype=np.float64)
    return np.asarray(np.exp(-0.5 * z * z) / _SQRT_2PI, dtype=np.float64)


def norm_cdf(x: ArrayLike) -> FloatArray:
    """Standard normal distribution function."""
    return np.asarray(ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def d1(spot: ArrayLike, strike: float, tau: ArrayLike, vol: ArrayLike, rate: float = 0.0) -> FloatArray:
    """The Black-Scholes d1. Returns +/- infinity at ``tau == 0`` or ``vol == 0``."""
    s = np.asarray(spot, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)
    total = v * np.sqrt(t)
    forward_log = np.log(s / strike) + rate * t
    with np.errstate(divide="ignore", invalid="ignore"):
        out = forward_log / total + 0.5 * total
    degenerate = total <= 0.0
    if np.any(degenerate):
        limit = np.where(forward_log > 0.0, np.inf, np.where(forward_log < 0.0, -np.inf, 0.0))
        out = np.where(degenerate, limit, out)
    return np.asarray(out, dtype=np.float64)


def price(
    spot: ArrayLike,
    strike: float,
    tau: ArrayLike,
    vol: ArrayLike,
    rate: float = 0.0,
    kind: OptionKind = "call",
) -> FloatArray:
    """Black-Scholes price of a European call or put."""
    s = np.asarray(spot, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)
    a = d1(s, strike, t, v, rate)
    b = a - v * np.sqrt(t)
    disc = strike * np.exp(-rate * t)
    call = s * norm_cdf(a) - disc * norm_cdf(b)
    if kind == "call":
        return np.asarray(call, dtype=np.float64)
    return np.asarray(call - s + disc, dtype=np.float64)


def delta(
    spot: ArrayLike,
    strike: float,
    tau: ArrayLike,
    vol: ArrayLike,
    rate: float = 0.0,
    kind: OptionKind = "call",
) -> FloatArray:
    """Derivative of the price with respect to spot."""
    call = norm_cdf(d1(spot, strike, tau, vol, rate))
    if kind == "call":
        return call
    return np.asarray(call - 1.0, dtype=np.float64)


def gamma(spot: ArrayLike, strike: float, tau: ArrayLike, vol: ArrayLike, rate: float = 0.0) -> FloatArray:
    """Second derivative of the price with respect to spot (same for calls and puts)."""
    s = np.asarray(spot, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)
    total = v * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = norm_pdf(d1(s, strike, t, v, rate)) / (s * total)
    return np.asarray(np.where(total > 0.0, out, 0.0), dtype=np.float64)


def vega(spot: ArrayLike, strike: float, tau: ArrayLike, vol: ArrayLike, rate: float = 0.0) -> FloatArray:
    """Derivative of the price with respect to volatility, per unit of volatility."""
    s = np.asarray(spot, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    out = s * norm_pdf(d1(s, strike, t, vol, rate)) * np.sqrt(t)
    return np.asarray(np.where(t > 0.0, out, 0.0), dtype=np.float64)
