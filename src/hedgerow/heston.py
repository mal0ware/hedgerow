"""Heston call price and deltas by Fourier inversion.

The simulator in :mod:`hedgerow.paths` gives paths. This module gives what a
hedger who knows the Heston model would compute from it: the call price, the
model delta dC/dS, and the minimum-variance delta for a hedger who can trade
the stock only,

    delta_mv = dC/dS + rho xi (dC/dV) / S,

which is the regression coefficient of dC on dS when dS and dV are correlated.

Method
------
Let X = log(S_T / S_t). Under the measure that uses the stock as numeraire,
X has characteristic function phi(u - i), where phi is the Heston
characteristic function under the martingale measure (rates are zero, so
phi(-i) = 1). Its density f is recovered on a fine grid by one FFT. Then, with
k = log(K / S_t),

    dC/dS = P(X > k)                       (the price is homogeneous in S and K)
    C / S = integral over x > k of (1 - exp(k - x)) f(x) dx.

The characteristic function is exp(A(u) + B(u) V), so the derivative of the
density with respect to V has transform B(u) times that of the density, and
dC/dV comes from the same two integrals applied to it.

With the parameters used here the variance sits near zero on most paths, and
the density of X is then close to a point mass. To keep the inversion stable
the characteristic function is multiplied by exp(-eps^2 u^2 / 2), which adds
independent normal noise of standard deviation ``eps`` to X. At the default
resolution eps is 0.02% of the spot price.

The characteristic function is written in the form of Albrecher, Mayer,
Schoutens and Tistaert (2007), which stays on the principal branch of the
complex logarithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from hedgerow import black_scholes as bs
from hedgerow.paths import HestonParams

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class GridSpec:
    """Resolution of the Fourier inversion and of the lookup table."""

    log2_n: int = 17
    """The FFT uses 2**log2_n points."""
    du: float = 0.5
    """Spacing in the transform variable. The x grid then spans +/- pi / du."""
    n_money: int = 1201
    """Points in log-moneyness, packed around the strike."""
    n_var: int = 96
    """Points in variance, packed around zero."""
    v_max: float = 8.0
    """Largest variance in the table. Larger values are clipped to it."""
    m_max: float = 1.0
    """Largest absolute log-moneyness in the table. Larger values are clipped."""

    @property
    def n(self) -> int:
        return int(2**self.log2_n)

    @property
    def eps(self) -> float:
        """Smoothing width: the damping factor is exp(-18) at the largest u."""
        return 6.0 / (0.5 * self.n * self.du)

    @staticmethod
    def coarse() -> GridSpec:
        """A small grid for smoke runs and tests."""
        return GridSpec(log2_n=13, n_money=301, n_var=32)


def cf_exponents(
    u: ComplexArray, tau: float, params: HestonParams
) -> tuple[ComplexArray, ComplexArray]:
    """A(u) and B(u) with E[exp(i u X)] = exp(A + B V) for X = log(S_T / S_t)."""
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    big_b = (b - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
    big_a = kappa * theta / (xi * xi) * ((b - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.asarray(big_a, dtype=np.complex128), np.asarray(big_b, dtype=np.complex128)


@dataclass(frozen=True)
class Slice:
    """Price and deltas at one time to expiry, on the x grid, for each variance."""

    x: FloatArray
    """Grid of k = log(K / S), shape ``(n,)``."""
    delta: FloatArray
    """dC/dS, shape ``(n_v, n)``."""
    price_over_spot: FloatArray
    """C / S, shape ``(n_v, n)``."""
    dprice_dv_over_spot: FloatArray
    """(dC/dV) / S, shape ``(n_v, n)``."""


def _tail(values: FloatArray, dx: float) -> FloatArray:
    """Integral from each grid point to the right end, by the trapezoid rule."""
    rev = np.cumsum(values[:, ::-1], axis=1)[:, ::-1]
    return np.asarray(dx * (rev - 0.5 * values), dtype=np.float64)


def fourier_slice(tau: float, variances: FloatArray, params: HestonParams, spec: GridSpec) -> Slice:
    """Invert the characteristic function at one ``tau`` for a vector of variances."""
    n, du = spec.n, spec.du
    dx = 2.0 * np.pi / (n * du)
    u = np.asarray((np.arange(n) - n // 2) * du, dtype=np.float64)
    x = np.asarray((np.arange(n) - n // 2) * dx, dtype=np.float64)
    big_a, big_b = cf_exponents(np.asarray(u - 1j, dtype=np.complex128), tau, params)
    damp = np.exp(-0.5 * (spec.eps * u) ** 2)
    v = np.asarray(variances, dtype=np.float64)[:, None]
    psi = np.exp(big_a[None, :] + big_b[None, :] * v) * damp[None, :]
    scale = du / (2.0 * np.pi)

    def invert(transform: ComplexArray) -> FloatArray:
        shifted = np.fft.ifftshift(transform, axes=1)
        return np.asarray(
            scale * np.fft.fftshift(np.fft.fft(shifted, axis=1), axes=1).real, dtype=np.float64
        )

    density = invert(psi)
    d_density = invert(psi * big_b[None, :])
    decay = np.exp(-x)[None, :]
    grow = np.exp(x)[None, :]
    prob = _tail(density, dx)
    price = prob - grow * _tail(density * decay, dx)
    dprice = _tail(d_density, dx) - grow * _tail(d_density * decay, dx)
    return Slice(x, prob, price, dprice)


def call_price(
    spot: float,
    strike: float,
    tau: float,
    variance: float,
    params: HestonParams,
    spec: GridSpec | None = None,
) -> float:
    """Heston price of a European call, zero rates."""
    s = fourier_slice(tau, np.array([variance]), params, spec or GridSpec())
    k = float(np.log(strike / spot))
    return float(spot * np.interp(k, s.x, s.price_over_spot[0]))


def implied_vol(price: float, spot: float, strike: float, tau: float) -> float:
    """The Black-Scholes volatility that gives ``price``. Zero rates."""

    def gap(vol: float) -> float:
        return float(bs.price(spot, strike, tau, vol)[()]) - price

    return float(brentq(gap, 1e-4, 5.0, xtol=1e-12))


def atm_implied_vol(
    spot: float, strike: float, tau: float, params: HestonParams, spec: GridSpec | None = None
) -> float:
    """Implied volatility of the call at the starting variance."""
    return implied_vol(call_price(spot, strike, tau, params.v0, params, spec), spot, strike, tau)


@dataclass(frozen=True)
class DeltaTable:
    """Deltas on a (date, log-moneyness, variance) grid, looked up by interpolation.

    ``money`` is log(S / K). The variance axis is stored as V**(1/4) so that
    linear interpolation is fine near zero, where the delta changes fastest.
    """

    money: FloatArray
    root_var: FloatArray
    delta: FloatArray
    min_var: FloatArray
    v_max: float

    def _lookup(
        self, table: FloatArray, step: int, spot_over_strike: FloatArray, var: FloatArray
    ) -> FloatArray:
        m = np.clip(np.log(spot_over_strike), self.money[0], self.money[-1])
        q = np.clip(var, 0.0, self.v_max) ** 0.25
        i = np.clip(np.searchsorted(self.money, m) - 1, 0, len(self.money) - 2)
        j = np.clip(np.searchsorted(self.root_var, q) - 1, 0, len(self.root_var) - 2)
        wm = (m - self.money[i]) / (self.money[i + 1] - self.money[i])
        wq = (q - self.root_var[j]) / (self.root_var[j + 1] - self.root_var[j])
        t = table[step]
        out = (
            (1.0 - wm) * (1.0 - wq) * t[i, j]
            + wm * (1.0 - wq) * t[i + 1, j]
            + (1.0 - wm) * wq * t[i, j + 1]
            + wm * wq * t[i + 1, j + 1]
        )
        return np.asarray(out, dtype=np.float64)

    def deltas(
        self, prices: FloatArray, variances: FloatArray, strike: float, min_var: bool = False
    ) -> FloatArray:
        """Holdings at every rebalance date. Shapes as in :func:`hedgerow.hedging.bs_deltas`."""
        n_steps = prices.shape[1] - 1
        if n_steps != self.delta.shape[0]:
            raise ValueError("the table was built for a different number of rebalance dates")
        table = self.min_var if min_var else self.delta
        out = np.empty((prices.shape[0], n_steps))
        for k in range(n_steps):
            out[:, k] = self._lookup(table, k, prices[:, k] / strike, variances[:, k])
        return out


@lru_cache(maxsize=4)
def delta_table(
    params: HestonParams, maturity: float, n_steps: int, spec: GridSpec | None = None
) -> DeltaTable:
    """Build the lookup table for ``n_steps`` evenly spaced rebalance dates."""
    spec = spec or GridSpec()
    t = np.linspace(-5.5, 5.5, spec.n_money)
    money = spec.m_max * np.sinh(t) / np.sinh(5.5)
    root_var = np.linspace(0.0, spec.v_max**0.25, spec.n_var)
    var = root_var**4
    delta = np.empty((n_steps, spec.n_money, spec.n_var))
    min_var = np.empty_like(delta)
    chunk = 16
    for k in range(n_steps):
        tau = maturity * (1.0 - k / n_steps)
        for start in range(0, spec.n_var, chunk):
            s = fourier_slice(tau, var[start : start + chunk], params, spec)
            for row in range(s.delta.shape[0]):
                # k = log(K / S) = -money; np.interp needs increasing x, which s.x is.
                d = np.interp(-money, s.x, s.delta[row])
                dv = np.interp(-money, s.x, s.dprice_dv_over_spot[row])
                delta[k, :, start + row] = d
                min_var[k, :, start + row] = d + params.rho * params.xi * dv
    return DeltaTable(money, root_var, delta, min_var, spec.v_max)
