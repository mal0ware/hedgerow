"""The experiments behind every table and figure.

Each function takes a :class:`RunConfig` and returns plain rows (lists of
dicts). Nothing here writes files or draws; :mod:`hedgerow.report` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hedgerow import black_scholes as bs
from hedgerow.deep import Contract, TrainConfig, TrainedHedge, train
from hedgerow.hedging import (
    boyle_emanuel_scale,
    bs_deltas,
    derman_kamal_std,
    discrete_hedge_std,
    hedge_pnl,
    leland_vol,
    payoff,
)
from hedgerow.paths import HestonParams, gbm_paths, heston_paths
from hedgerow.risk import cvar, paired_block_difference

FloatArray = NDArray[np.float64]
Row = dict[str, float | int | str]
World = Literal["black_scholes", "heston"]

WORLDS: tuple[World, ...] = ("black_scholes", "heston")


@dataclass(frozen=True)
class RunConfig:
    """Everything that sizes a run. ``full()`` made the committed results."""

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30.0 / 365.0
    n_steps: int = 30
    vol: float = 0.2
    heston: HestonParams = field(default_factory=HestonParams)
    cost_rates: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01)
    alphas: tuple[float, ...] = (0.5, 0.99)
    n_train: int = 200_000
    n_test: int = 200_000
    n_blocks: int = 20
    iterations: tuple[int, ...] = (3000, 2000)
    batch_sizes: tuple[int, ...] = (1024, 4096)
    extra_seeds: tuple[int, ...] = (1, 2)
    seed_study_cost: float = 0.005
    n_classical: int = 100_000
    rebalance_grid: tuple[int, ...] = (5, 10, 20, 30, 60, 120, 250, 500)
    leland_cost: float = 0.005
    device: str = "cpu"
    seed: int = 20261123

    @staticmethod
    def full() -> RunConfig:
        return RunConfig()

    @staticmethod
    def fast() -> RunConfig:
        """A smoke run: same code path, small sizes, under two minutes on a laptop CPU."""
        return RunConfig(
            cost_rates=(0.0, 0.005),
            n_train=20_000,
            n_test=20_000,
            iterations=(200, 120),
            batch_sizes=(1024, 2048),
            extra_seeds=(1,),
            n_classical=20_000,
            rebalance_grid=(5, 10, 30, 120),
        )

    def train_config(self, alpha: float, seed: int = 0) -> TrainConfig:
        i = self.alphas.index(alpha)
        return TrainConfig(
            alpha=alpha,
            iterations=self.iterations[i],
            batch_size=self.batch_sizes[i],
            seed=seed,
            device=self.device,
        )


# --------------------------------------------------------------- experiment 1


def discretisation_table(cfg: RunConfig) -> list[Row]:
    """Hedging error against the number of rebalances, no costs, Black-Scholes world."""
    premium = float(bs.price(cfg.s0, cfg.strike, cfg.maturity, cfg.vol)[()])
    rows: list[Row] = []
    for i, n in enumerate(cfg.rebalance_grid):
        paths = gbm_paths(cfg.n_classical, n, cfg.maturity, cfg.vol, cfg.seed + 100 + i, cfg.s0)
        deltas = bs_deltas(paths, cfg.strike, cfg.maturity, cfg.vol)
        pnl = hedge_pnl(
            paths, deltas, payoff(paths[:, -1], cfg.strike), premium, maturity=cfg.maturity
        ).pnl
        std = float(pnl.std(ddof=1))
        rows.append(
            {
                "rebalances": n,
                "mean_error": float(pnl.mean()),
                "mean_se": std / float(np.sqrt(cfg.n_classical)),
                "std_error": std,
                "std_se": std / float(np.sqrt(2.0 * cfg.n_classical)),
                "predicted_std": discrete_hedge_std(cfg.s0, cfg.strike, cfg.maturity, cfg.vol, n),
                "derman_kamal_std": derman_kamal_std(cfg.s0, cfg.strike, cfg.maturity, cfg.vol, n),
                "std_times_root_n": std * float(np.sqrt(n)),
                "std_pct_of_premium": 100.0 * std / premium,
            }
        )
    return rows


def loglog_slope(rows: list[Row], x: str = "rebalances", y: str = "std_error") -> float:
    """Least-squares slope of log y on log x."""
    xs = np.log([float(r[x]) for r in rows])
    ys = np.log([float(r[y]) for r in rows])
    return float(np.polyfit(xs, ys, 1)[0])


# --------------------------------------------------------------- experiment 2


def one_interval_sample(
    cfg: RunConfig, tau: float = 0.25, dt: float = 1.0 / 365.0
) -> tuple[FloatArray, float]:
    """Gain of a hedged long call over one interval, in units of lambda.

    Returns the sample of ``gain / lambda`` and lambda itself. The
    Boyle-Emanuel prediction for the sample is ``u^2 - 1``.
    """
    rng = np.random.default_rng(cfg.seed + 200)
    u = rng.standard_normal(cfg.n_classical)
    s1 = cfg.s0 * np.exp(-0.5 * cfg.vol**2 * dt + cfg.vol * np.sqrt(dt) * u)
    gain = (
        bs.price(s1, cfg.strike, tau - dt, cfg.vol)
        - bs.price(cfg.s0, cfg.strike, tau, cfg.vol)
        - bs.delta(cfg.s0, cfg.strike, tau, cfg.vol) * (s1 - cfg.s0)
    )
    lam = boyle_emanuel_scale(cfg.s0, cfg.strike, tau, cfg.vol, dt)
    return np.asarray(gain / lam, dtype=np.float64), lam


def one_interval_table(sample: FloatArray) -> list[Row]:
    """Moments of the one-interval gain against those of u^2 - 1."""
    centred = sample - sample.mean()
    std = float(sample.std())
    return [
        {"statistic": "mean / lambda", "simulated": float(sample.mean()), "chi_square": 0.0},
        {"statistic": "std / lambda", "simulated": std, "chi_square": float(np.sqrt(2.0))},
        {
            "statistic": "skewness",
            "simulated": float(np.mean(centred**3) / std**3),
            "chi_square": float(np.sqrt(8.0)),
        },
        {
            "statistic": "share of intervals with a loss",
            "simulated": float(np.mean(sample < 0.0)),
            "chi_square": 0.6826894921370859,
        },
        {
            "statistic": "worst outcome / lambda",
            "simulated": float(sample.min()),
            "chi_square": -1.0,
        },
    ]


# --------------------------------------------------------------- experiment 3


def leland_table(cfg: RunConfig) -> list[Row]:
    """Black-Scholes hedge and Leland hedge under costs, across rebalance counts.

    Each hedger charges the price at its own volatility. The entry trade is
    left out of the costs here, as in Kabanov and Safarian (1997), Remark 2,
    so that the mean error isolates the cost of re-hedging.
    """
    k = cfg.leland_cost
    rows: list[Row] = []
    for i, n in enumerate(cfg.rebalance_grid):
        paths = gbm_paths(cfg.n_classical, n, cfg.maturity, cfg.vol, cfg.seed + 300 + i, cfg.s0)
        liability = payoff(paths[:, -1], cfg.strike)
        sig_hat = leland_vol(cfg.vol, k, cfg.maturity / n)
        row: Row = {
            "rebalances": n,
            "leland_number": 2.0
            * float(np.sqrt(2.0 / np.pi))
            * k
            / (cfg.vol * float(np.sqrt(cfg.maturity / n))),
            "leland_vol": sig_hat,
            "leland_premium": float(bs.price(cfg.s0, cfg.strike, cfg.maturity, sig_hat)[()]),
        }
        for name, vol in (("bs", cfg.vol), ("leland", sig_hat)):
            deltas = bs_deltas(paths, cfg.strike, cfg.maturity, vol)
            premium = float(bs.price(cfg.s0, cfg.strike, cfg.maturity, vol)[()])
            res = hedge_pnl(paths, deltas, liability, premium, k, maturity=cfg.maturity)
            pnl = res.pnl + k * np.abs(deltas[:, 0]) * cfg.s0
            row[f"{name}_mean_error"] = float(pnl.mean())
            row[f"{name}_std_error"] = float(pnl.std(ddof=1))
            row[f"{name}_mean_cost"] = float(
                res.cost.mean() - k * np.abs(deltas[:, 0]).mean() * cfg.s0
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------- experiment 4


@dataclass
class WorldData:
    """Training and test paths for one world. ``*_var`` is None under Black-Scholes."""

    world: World
    train: FloatArray
    train_var: FloatArray | None
    test: FloatArray
    test_var: FloatArray | None


def make_world(cfg: RunConfig, world: World) -> WorldData:
    """Draw the training and test paths. Test paths use a different seed."""
    offset = 0 if world == "black_scholes" else 50
    if world == "black_scholes":
        train_p = gbm_paths(
            cfg.n_train, cfg.n_steps, cfg.maturity, cfg.vol, cfg.seed + 1 + offset, cfg.s0
        )
        test_p = gbm_paths(
            cfg.n_test, cfg.n_steps, cfg.maturity, cfg.vol, cfg.seed + 2 + offset, cfg.s0
        )
        return WorldData(world, train_p, None, test_p, None)
    train_p, train_v = heston_paths(
        cfg.n_train, cfg.n_steps, cfg.maturity, cfg.heston, cfg.seed + 1 + offset, cfg.s0
    )
    test_p, test_v = heston_paths(
        cfg.n_test, cfg.n_steps, cfg.maturity, cfg.heston, cfg.seed + 2 + offset, cfg.s0
    )
    return WorldData(world, train_p, train_v, test_p, test_v)


def contract_for(cfg: RunConfig, cost_rate: float) -> Contract:
    return Contract(
        strike=cfg.strike,
        maturity=cfg.maturity,
        cost_rate=cost_rate,
        vol_scale=cfg.vol,
        var_scale=cfg.vol**2,
    )


def classical_deltas(cfg: RunConfig, data: WorldData, cost_rate: float) -> dict[str, FloatArray]:
    """Holdings of the formula-based strategies on the test paths."""
    dt = cfg.maturity / cfg.n_steps
    out = {
        "unhedged": np.zeros((data.test.shape[0], cfg.n_steps)),
        "bs_delta": bs_deltas(data.test, cfg.strike, cfg.maturity, cfg.vol),
        "leland": bs_deltas(
            data.test, cfg.strike, cfg.maturity, leland_vol(cfg.vol, cost_rate, dt)
        ),
    }
    if data.test_var is not None:
        spot_vol = np.sqrt(np.maximum(data.test_var[:, :-1], 1e-8))
        out["bs_delta_spot_vol"] = bs_deltas(data.test, cfg.strike, cfg.maturity, spot_vol)
    return out


def strategy_losses(
    cfg: RunConfig, data: WorldData, cost_rate: float, deltas: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Loss, cost and turnover per test path. Loss is minus terminal wealth with no premium."""
    liability = payoff(data.test[:, -1], cfg.strike)
    res = hedge_pnl(data.test, deltas, liability, 0.0, cost_rate, maturity=cfg.maturity)
    return -res.pnl, res.cost, res.turnover


def summarise(
    world: str,
    cost_rate: float,
    strategy: str,
    loss: FloatArray,
    cost: FloatArray,
    turnover: FloatArray,
) -> Row:
    return {
        "world": world,
        "cost_rate": cost_rate,
        "strategy": strategy,
        "mean_loss": float(loss.mean()),
        "std_loss": float(loss.std(ddof=1)),
        "cvar_50": cvar(loss, 0.5),
        "cvar_99": cvar(loss, 0.99),
        "mean_cost": float(cost.mean()),
        "mean_turnover": float(turnover.mean()),
    }


@dataclass
class Comparison:
    """Output of :func:`run_comparison`."""

    rows: list[Row]
    headline: list[Row]
    seeds: list[Row]
    losses: dict[tuple[str, float, str], FloatArray]
    hedges: dict[tuple[str, float, float], TrainedHedge]


def deep_name(alpha: float) -> str:
    return f"deep_cvar_{round(100 * alpha):d}"


def run_comparison(cfg: RunConfig, log: bool = False) -> Comparison:
    """Every strategy on the same test paths, in both worlds, at every cost rate."""
    rows: list[Row] = []
    headline: list[Row] = []
    seeds: list[Row] = []
    losses: dict[tuple[str, float, str], FloatArray] = {}
    hedges: dict[tuple[str, float, float], TrainedHedge] = {}
    for world in WORLDS:
        data = make_world(cfg, world)
        for cost_rate in cfg.cost_rates:
            for name, deltas in classical_deltas(cfg, data, cost_rate).items():
                loss, cost, turn = strategy_losses(cfg, data, cost_rate, deltas)
                losses[(world, cost_rate, name)] = loss
                rows.append(summarise(world, cost_rate, name, loss, cost, turn))
            contract = contract_for(cfg, cost_rate)
            for alpha in cfg.alphas:
                hedge = train(data.train, data.train_var, contract, cfg.train_config(alpha))
                hedges[(world, cost_rate, alpha)] = hedge
                loss, cost, turn = strategy_losses(
                    cfg, data, cost_rate, hedge.deltas(data.test, data.test_var)
                )
                name = deep_name(alpha)
                losses[(world, cost_rate, name)] = loss
                rows.append(summarise(world, cost_rate, name, loss, cost, turn))
                headline.append(headline_row(cfg, world, cost_rate, alpha, losses))
                if log:
                    print(f"  trained {world} cost={cost_rate} alpha={alpha}", flush=True)
            if cost_rate == cfg.seed_study_cost:
                alpha = cfg.alphas[0]
                base = losses[(world, cost_rate, deep_name(alpha))]
                seeds.append(
                    seed_row(world, cost_rate, alpha, 0, base, losses[(world, cost_rate, "leland")])
                )
                for seed in cfg.extra_seeds:
                    other = train(
                        data.train, data.train_var, contract, cfg.train_config(alpha, seed)
                    )
                    loss, _, _ = strategy_losses(
                        cfg, data, cost_rate, other.deltas(data.test, data.test_var)
                    )
                    seeds.append(
                        seed_row(
                            world,
                            cost_rate,
                            alpha,
                            seed,
                            loss,
                            losses[(world, cost_rate, "leland")],
                        )
                    )
    return Comparison(rows, headline, seeds, losses, hedges)


def headline_row(
    cfg: RunConfig,
    world: str,
    cost_rate: float,
    alpha: float,
    losses: dict[tuple[str, float, str], FloatArray],
) -> Row:
    """Network against Leland on the CVaR the network was trained on."""
    deep = losses[(world, cost_rate, deep_name(alpha))]
    leland = losses[(world, cost_rate, "leland")]
    diff, se = paired_block_difference(deep, leland, alpha, cfg.n_blocks)
    if abs(diff) < 2.0 * se:
        verdict = "tie"
    elif diff < 0:
        verdict = "network"
    else:
        verdict = "leland"
    return {
        "world": world,
        "cost_rate": cost_rate,
        "alpha": alpha,
        "bs_delta_cvar": cvar(losses[(world, cost_rate, "bs_delta")], alpha),
        "leland_cvar": cvar(leland, alpha),
        "network_cvar": cvar(deep, alpha),
        "network_minus_leland": cvar(deep, alpha) - cvar(leland, alpha),
        "block_mean_difference": diff,
        "block_se": se,
        "lower_cvar": verdict,
    }


def seed_row(
    world: str, cost_rate: float, alpha: float, seed: int, loss: FloatArray, leland: FloatArray
) -> Row:
    return {
        "world": world,
        "cost_rate": cost_rate,
        "alpha": alpha,
        "training_seed": seed,
        "network_cvar": cvar(loss, alpha),
        "leland_cvar": cvar(leland, alpha),
        "network_std": float(loss.std(ddof=1)),
    }


def with_device(cfg: RunConfig, device: str) -> RunConfig:
    return replace(cfg, device=device)
