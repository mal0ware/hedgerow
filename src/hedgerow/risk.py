"""Risk measures on a sample of losses. Positive numbers are losses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def oce_objective(losses: FloatArray, w: float, alpha: float) -> float:
    """w + E[max(L - w, 0)] / (1 - alpha).

    This is Buehler et al. (2019), Eq. (3.5), with the loss function of
    Example 3.9. Its minimum over ``w`` is the CVaR of the losses.
    """
    return float(w + np.mean(np.maximum(losses - w, 0.0)) / (1.0 - alpha))


def cvar(losses: FloatArray, alpha: float) -> float:
    """Conditional value at risk: the mean of the worst ``1 - alpha`` share of losses.

    Computed as the minimum of :func:`oce_objective`, which an empirical
    ``alpha``-quantile attains. ``alpha = 0`` gives the mean loss.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    x = np.asarray(losses, dtype=np.float64)
    if alpha == 0.0:
        return float(np.mean(x))
    w = float(np.quantile(x, alpha, method="inverted_cdf"))
    return oce_objective(x, w, alpha)


def normal_cvar(mean: float, std: float, alpha: float) -> float:
    """CVaR of a normal loss: mean + std * pdf(z_alpha) / (1 - alpha)."""
    from scipy.stats import norm

    z = float(norm.ppf(alpha))
    return float(mean + std * float(norm.pdf(z)) / (1.0 - alpha))


def paired_block_difference(a: FloatArray, b: FloatArray, alpha: float, n_blocks: int = 20) -> tuple[float, float]:
    """Mean and standard error of CVaR(a) - CVaR(b) across equal blocks of paths.

    ``a`` and ``b`` are losses of two strategies on the same paths, in the same
    order. The paths are independent, so the block differences are too.
    """
    size = len(a) // n_blocks
    diffs = np.array(
        [
            cvar(a[i * size : (i + 1) * size], alpha) - cvar(b[i * size : (i + 1) * size], alpha)
            for i in range(n_blocks)
        ]
    )
    return float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(n_blocks))
