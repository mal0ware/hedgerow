"""Price-path simulators. Every path set is a pure function of its seed.

Paths are generated once with numpy and handed to every strategy, so a
comparison between two strategies is always a comparison on the same paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters in the notation of Buehler et al. (2019), Eq. (5.2).

    dS = sqrt(V) S dB,  dV = kappa (theta - V) dt + xi sqrt(V) dW,  d<B, W> = rho dt.

    The paper writes alpha for ``kappa``, b for ``theta`` and sigma for ``xi``.
    The defaults are the values printed below Eq. (5.2).
    """

    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 2.0
    rho: float = -0.7
    v0: float = 0.04


def gbm_paths(
    n_paths: int,
    n_steps: int,
    maturity: float,
    vol: float,
    seed: int,
    s0: float = 100.0,
    drift: float = 0.0,
) -> FloatArray:
    """Geometric Brownian motion sampled exactly on an even grid.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first column is ``s0``.
    """
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (drift - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return np.asarray(s0 * np.exp(log_paths), dtype=np.float64)


def cir_step(
    v: FloatArray, dt: float, params: HestonParams, rng: np.random.Generator
) -> FloatArray:
    """One exact draw from the CIR transition law (Glasserman 2004, Section 3.4).

    V(t + dt) given V(t) is c times a noncentral chi-square with
    d = 4 kappa theta / xi^2 degrees of freedom and noncentrality V(t) exp(-kappa dt) / c,
    where c = xi^2 (1 - exp(-kappa dt)) / (4 kappa).
    """
    decay = np.exp(-params.kappa * dt)
    c = params.xi**2 * (1.0 - decay) / (4.0 * params.kappa)
    dof = 4.0 * params.kappa * params.theta / params.xi**2
    return np.asarray(c * rng.noncentral_chisquare(dof, v * decay / c), dtype=np.float64)


def heston_paths(
    n_paths: int,
    n_steps: int,
    maturity: float,
    params: HestonParams,
    seed: int,
    s0: float = 100.0,
    substeps: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Heston price and variance paths under the martingale measure.

    The variance is drawn exactly from the CIR transition law. The price uses
    the simplified Broadie-Kaya step of Buehler et al. (2019), footnote 5: over
    each step the variance inside the price equation is held at its value at
    the start of the step,

        log S' = log S - V h / 2 + (rho / xi) (V' - V - kappa (theta - V) h)
                 + sqrt((1 - rho^2) V h) Z.

    ``substeps`` splits each reporting step into that many simulation steps.
    Returns ``(prices, variances)``, each of shape ``(n_paths, n_steps + 1)``.
    """
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    rng = np.random.default_rng(seed)
    h = maturity / (n_steps * substeps)
    log_s = np.full(n_paths, np.log(s0))
    v = np.full(n_paths, params.v0)
    prices = np.empty((n_paths, n_steps + 1))
    variances = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = s0
    variances[:, 0] = params.v0
    orth = np.sqrt(1.0 - params.rho**2)
    for k in range(1, n_steps + 1):
        for _ in range(substeps):
            v_next = cir_step(v, h, params, rng)
            z = rng.standard_normal(n_paths)
            log_s = (
                log_s
                - 0.5 * v * h
                + (params.rho / params.xi) * (v_next - v - params.kappa * (params.theta - v) * h)
                + orth * np.sqrt(v * h) * z
            )
            v = v_next
        prices[:, k] = np.exp(log_s)
        variances[:, k] = v
    return prices, variances


def cir_mean(t: float, params: HestonParams) -> float:
    """E[V_t] for the CIR variance."""
    return float(params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * t))


def cir_variance(t: float, params: HestonParams) -> float:
    """Var[V_t] for the CIR variance (Cox, Ingersoll and Ross 1985, Eq. (19))."""
    decay = np.exp(-params.kappa * t)
    a = params.v0 * params.xi**2 / params.kappa * (decay - decay**2)
    b = params.theta * params.xi**2 / (2.0 * params.kappa) * (1.0 - decay) ** 2
    return float(a + b)


def expected_integrated_variance(maturity: float, params: HestonParams) -> float:
    """E of the integral of V over [0, maturity]: L(0, v0) in Buehler et al. Eq. (5.4)."""
    k = params.kappa
    return float(
        (params.v0 - params.theta) / k * (1.0 - np.exp(-k * maturity)) + params.theta * maturity
    )
