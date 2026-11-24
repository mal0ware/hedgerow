"""The experiments behind every table and figure.

Each function takes a :class:`RunConfig` and returns plain rows (lists of
dicts). Nothing here writes files or draws; :mod:`hedgerow.report` does that.

The strategy comparison keeps three sets of paths apart in each world:
training paths for the networks, validation paths on which every choice is
made (training length, and the target and width of the no-trade band), and
test paths that are only ever scored.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hedgerow import black_scholes as bs
from hedgerow.deep import Contract, TrainConfig, TrainedHedge, set_determinism, train
from hedgerow.hedging import (
    boyle_emanuel_scale,
    bs_deltas,
    derman_kamal_std,
    discrete_hedge_std,
    hedge_pnl,
    leland_vol,
    payoff,
)
from hedgerow.heston import GridSpec, atm_implied_vol, delta_table
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
    heston_grid: GridSpec = field(default_factory=GridSpec)
    cost_rates: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01)
    alphas: tuple[float, ...] = (0.5, 0.99)
    n_train: int = 200_000
    n_val: int = 200_000
    n_test: int = 200_000
    n_blocks: int = 20
    length_grid: tuple[tuple[int, ...], ...] = ((1500, 3000, 6000), (1000, 2000, 4000))
    """Training lengths tried on validation paths, one tuple per CVaR level."""
    batch_sizes: tuple[int, ...] = (1024, 4096)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    """Training seeds for every network in the comparison."""
    study_seeds: tuple[int, ...] = (0, 1, 2)
    """Training seeds used in the training-length study."""
    study_cost: float = 0.005
    """Cost level of the training-length study and of the detail figures."""
    band_widths: tuple[float, ...] = (
        0.0,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.75,
        1.0,
    )
    """Half-widths of the no-trade band tried on validation paths, in shares."""
    vol_shifts: tuple[float, ...] = (0.15, 0.25)
    """Volatilities of the Black-Scholes test paths that nobody was built for."""
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
        """A smoke run: same code path, small sizes, a few minutes on a laptop CPU."""
        return RunConfig(
            heston_grid=GridSpec.coarse(),
            cost_rates=(0.0, 0.005),
            n_train=20_000,
            n_val=20_000,
            n_test=20_000,
            length_grid=((100, 200), (60, 120)),
            batch_sizes=(1024, 2048),
            seeds=(0, 1),
            study_seeds=(0,),
            band_widths=(0.0, 0.05, 0.15),
            vol_shifts=(0.25,),
            n_classical=20_000,
            rebalance_grid=(5, 10, 30, 120),
        )

    def train_config(self, alpha: float, seed: int, iterations: int) -> TrainConfig:
        return TrainConfig(
            alpha=alpha,
            iterations=iterations,
            batch_size=self.batch_sizes[self.alphas.index(alpha)],
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
#
# Strategies are "built for" a world and a cost rate and then run, unchanged,
# on test paths. When the test paths come from the world and cost the strategy
# was built for, that is the in-model comparison. When they do not, it is the
# misspecification study.

SEED_TRAIN, SEED_VAL, SEED_TEST, SEED_VOL_SHIFT = 1, 2, 4, 60


@dataclass(frozen=True)
class PathSet:
    """Prices and, in the Heston world, the variance at each date."""

    prices: FloatArray
    variances: FloatArray | None


def draw_paths(cfg: RunConfig, world: World, n: int, seed_offset: int) -> PathSet:
    """Paths of one world. The Heston seeds are shifted by 50."""
    if world == "black_scholes":
        prices = gbm_paths(n, cfg.n_steps, cfg.maturity, cfg.vol, cfg.seed + seed_offset, cfg.s0)
        return PathSet(prices, None)
    prices, variances = heston_paths(
        n, cfg.n_steps, cfg.maturity, cfg.heston, cfg.seed + 50 + seed_offset, cfg.s0
    )
    return PathSet(prices, variances)


@lru_cache(maxsize=2)
def training_paths(cfg: RunConfig, world: World) -> PathSet:
    return draw_paths(cfg, world, cfg.n_train, SEED_TRAIN)


def validation_paths(cfg: RunConfig, world: World) -> PathSet:
    return draw_paths(cfg, world, cfg.n_val, SEED_VAL)


def holdout_paths(cfg: RunConfig, world: World) -> PathSet:
    """The test paths. Nothing is chosen on them."""
    return draw_paths(cfg, world, cfg.n_test, SEED_TEST)


def contract_for(cfg: RunConfig, cost_rate: float) -> Contract:
    return Contract(
        strike=cfg.strike,
        maturity=cfg.maturity,
        cost_rate=cost_rate,
        vol_scale=cfg.vol,
        var_scale=cfg.vol**2,
    )


def heston_implied_vol(cfg: RunConfig) -> float:
    """Implied volatility of the contract in the Heston world at time zero."""
    return atm_implied_vol(cfg.s0, cfg.strike, cfg.maturity, cfg.heston, cfg.heston_grid)


def variances_or_flat(cfg: RunConfig, paths: PathSet) -> FloatArray:
    """The variance a Heston-built strategy is told. On Black-Scholes paths it is sigma^2."""
    if paths.variances is not None:
        return paths.variances
    return np.full(paths.prices.shape, cfg.vol**2)


def formula_deltas(
    cfg: RunConfig, design: World, cost_rate: float, paths: PathSet
) -> dict[str, FloatArray]:
    """Holdings of the formula hedges built for ``design`` at ``cost_rate``.

    Built for Black-Scholes: the delta and the Leland delta at ``cfg.vol``.
    Built for Heston: both of those at the Heston implied volatility, the
    Black-Scholes delta with the current variance plugged in, the Heston
    delta, and the Heston minimum-variance delta.
    """
    dt = cfg.maturity / cfg.n_steps
    p = paths.prices
    if design == "black_scholes":
        return {
            "bs_delta": bs_deltas(p, cfg.strike, cfg.maturity, cfg.vol),
            "leland": bs_deltas(p, cfg.strike, cfg.maturity, leland_vol(cfg.vol, cost_rate, dt)),
        }
    iv = heston_implied_vol(cfg)
    var = variances_or_flat(cfg, paths)
    table = delta_table(cfg.heston, cfg.maturity, cfg.n_steps, cfg.heston_grid)
    spot_vol = np.sqrt(np.maximum(var[:, :-1], 1e-8))
    return {
        "bs_delta_iv": bs_deltas(p, cfg.strike, cfg.maturity, iv),
        "leland_iv": bs_deltas(p, cfg.strike, cfg.maturity, leland_vol(iv, cost_rate, dt)),
        "bs_delta_spot_vol": bs_deltas(p, cfg.strike, cfg.maturity, spot_vol),
        "heston_delta": table.deltas(p, var, cfg.strike),
        "heston_min_var": table.deltas(p, var, cfg.strike, min_var=True),
    }


def leland_name(design: World) -> str:
    """The Leland hedge at the volatility that prices the option in ``design``."""
    return "leland" if design == "black_scholes" else "leland_iv"


def band_deltas(target: FloatArray, width: float) -> FloatArray:
    """A no-trade band: hold still while within ``width`` of the target, else trade to the edge.

    The width is a constant number of shares. The hedger starts with none.
    """
    out = np.empty_like(target)
    prev = np.zeros(target.shape[0])
    for k in range(target.shape[1]):
        prev = np.clip(prev, target[:, k] - width, target[:, k] + width)
        out[:, k] = prev
    return out


def score(
    cfg: RunConfig, paths: PathSet, deltas: FloatArray, cost_rate: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Loss, cost and turnover per path. Loss is minus terminal wealth with no premium."""
    liability = payoff(paths.prices[:, -1], cfg.strike)
    res = hedge_pnl(paths.prices, deltas, liability, 0.0, cost_rate, maturity=cfg.maturity)
    return -res.pnl, res.cost, res.turnover


@dataclass(frozen=True)
class Band:
    target: str
    width: float


def deep_name(alpha: float) -> str:
    return f"deep_cvar_{round(100 * alpha):d}"


def band_name(alpha: float) -> str:
    return f"band_cvar_{round(100 * alpha):d}"


def choose_bands(
    cfg: RunConfig, design: World, cost_rate: float, val: PathSet
) -> tuple[dict[float, Band], list[Row]]:
    """Pick, on validation paths, the band target and width with the lowest CVaR at each level.

    Width zero is in the grid, so the chosen band is never worse on validation
    paths than the best plain formula hedge.
    """
    best: dict[float, tuple[float, Band]] = {}
    for target, deltas in formula_deltas(cfg, design, cost_rate, val).items():
        for width in cfg.band_widths:
            loss, _, _ = score(cfg, val, band_deltas(deltas, width), cost_rate)
            for alpha in cfg.alphas:
                value = cvar(loss, alpha)
                if alpha not in best or value < best[alpha][0]:
                    best[alpha] = (value, Band(target, width))
    rows: list[Row] = [
        {
            "built_for": design,
            "cost_rate": cost_rate,
            "alpha": alpha,
            "target": best[alpha][1].target,
            "width": best[alpha][1].width,
            "validation_cvar": best[alpha][0],
        }
        for alpha in cfg.alphas
    ]
    return {alpha: best[alpha][1] for alpha in cfg.alphas}, rows


def classical_deltas(
    cfg: RunConfig, design: World, cost_rate: float, paths: PathSet, bands: dict[float, Band]
) -> dict[str, FloatArray]:
    """The formula hedges built for ``design`` plus the band chosen for each CVaR level."""
    out = formula_deltas(cfg, design, cost_rate, paths)
    for alpha, band in bands.items():
        out[band_name(alpha)] = band_deltas(out[band.target], band.width)
    return out


def network_deltas(cfg: RunConfig, hedge: TrainedHedge, paths: PathSet) -> FloatArray:
    """Holdings of a trained network on any paths.

    A network trained with the variance as an input is told sigma^2 on
    Black-Scholes paths. A network trained without it ignores the variance.
    """
    var = variances_or_flat(cfg, paths) if hedge.use_variance else None
    return hedge.deltas(paths.prices, var)


# ------------------------------------------------------------------ training


@dataclass(frozen=True)
class TrainJob:
    cfg: RunConfig
    world: World
    cost_rate: float
    alpha: float
    seed: int
    iterations: int

    @property
    def key(self) -> tuple[str, float, float, int, int]:
        return (self.world, self.cost_rate, self.alpha, self.seed, self.iterations)


def run_job(job: TrainJob) -> TrainedHedge:
    """Train one network. Runs on one thread, so the result does not depend on the pool."""
    set_determinism(1)
    paths = training_paths(job.cfg, job.world)
    return train(
        paths.prices,
        paths.variances,
        contract_for(job.cfg, job.cost_rate),
        job.cfg.train_config(job.alpha, job.seed, job.iterations),
    )


Hedges = dict[tuple[str, float, float, int, int], TrainedHedge]


def train_many(jobs: list[TrainJob], workers: int = 1, log: bool = False) -> Hedges:
    """Train every job, in ``workers`` processes if more than one is asked for."""
    out: Hedges = {}

    def note(job: TrainJob) -> None:
        if log:
            print(
                f"  trained {job.world} cost={job.cost_rate} alpha={job.alpha} "
                f"seed={job.seed} steps={job.iterations} ({len(out)}/{len(jobs)})",
                flush=True,
            )

    if workers <= 1:
        for job in jobs:
            out[job.key] = run_job(job)
            note(job)
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, hedge in zip(jobs, pool.map(run_job, jobs), strict=True):
            out[job.key] = hedge
            note(job)
    return out


# ------------------------------------------------------------------- scoring


def stats_row(loss: FloatArray, cost: FloatArray, turnover: FloatArray) -> Row:
    return {
        "mean_loss": float(loss.mean()),
        "std_loss": float(loss.std(ddof=1)),
        "cvar_50": cvar(loss, 0.5),
        "cvar_99": cvar(loss, 0.99),
        "mean_cost": float(cost.mean()),
        "mean_turnover": float(turnover.mean()),
    }


def judge(diffs: list[float], path_ses: list[float]) -> tuple[float, float, float, str]:
    """Mean difference, spread across seeds, path error, and who is lower.

    ``diffs`` holds one CVaR difference (network minus the other strategy) per
    training seed. The noise in the result of training one network is taken
    as the root sum of squares of the standard deviation across seeds and the
    standard error from the finite test set. The network or the other
    strategy is called lower only when the mean difference is more than twice
    that. Anything else is a tie.
    """
    mean = float(np.mean(diffs))
    seed_sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    path_se = float(np.mean(path_ses))
    noise = float(np.hypot(seed_sd, path_se))
    if abs(mean) <= 2.0 * noise:
        return mean, seed_sd, path_se, "tie"
    return mean, seed_sd, path_se, "network" if mean < 0 else "other"


@dataclass
class Comparison:
    """Output of :func:`run_comparison`.

    ``scores`` has one row per strategy, test world, world and cost it was
    built for, cost it was tested at, and training seed (-1 for a formula
    hedge). ``paired`` has the CVaR difference between each network and the
    strategies it is judged against on the same paths. ``vol_scores`` and
    ``vol_paired`` are the same for Black-Scholes test paths at other
    volatilities.
    """

    scores: list[Row]
    paired: list[Row]
    bands: list[Row]
    lengths: list[Row]
    iterations: dict[float, int]
    vol_scores: list[Row]
    vol_paired: list[Row]
    losses: dict[tuple[str, str], FloatArray]
    hedges: dict[tuple[str, float, float], TrainedHedge]


def study_lengths(
    cfg: RunConfig, hedges: Hedges, val: dict[World, PathSet]
) -> tuple[list[Row], dict[float, int]]:
    """Score every length of the study on validation paths and pick one per CVaR level.

    The pick is the length with the lowest validation CVaR, averaged over the
    study seeds and then over the two worlds after dividing each world's
    numbers by its best. The shorter length wins a tie.
    """
    rows: list[Row] = []
    chosen: dict[float, int] = {}
    for i, alpha in enumerate(cfg.alphas):
        ratio = np.zeros(len(cfg.length_grid[i]))
        for world in WORLDS:
            means = []
            for length in cfg.length_grid[i]:
                values = []
                for seed in cfg.study_seeds:
                    hedge = hedges[(world, cfg.study_cost, alpha, seed, length)]
                    loss, _, _ = score(
                        cfg,
                        val[world],
                        network_deltas(cfg, hedge, val[world]),
                        cfg.study_cost,
                    )
                    values.append(cvar(loss, alpha))
                    rows.append(
                        {
                            "world": world,
                            "cost_rate": cfg.study_cost,
                            "alpha": alpha,
                            "iterations": length,
                            "training_seed": seed,
                            "validation_cvar": values[-1],
                        }
                    )
                means.append(float(np.mean(values)))
            ratio += np.array(means) / min(means)
        chosen[alpha] = cfg.length_grid[i][int(np.argmin(ratio))]
    return rows, chosen


def run_comparison(cfg: RunConfig, workers: int = 1, log: bool = False) -> Comparison:
    """Choose on validation paths, train, then score everything on the test paths."""

    def say(message: str) -> None:
        if log:
            print(message, flush=True)

    val = {w: validation_paths(cfg, w) for w in WORLDS}

    say("  choosing no-trade bands on validation paths")
    bands: dict[tuple[str, float], dict[float, Band]] = {}
    band_rows: list[Row] = []
    for world in WORLDS:
        for cost_rate in cfg.cost_rates:
            bands[(world, cost_rate)], rows = choose_bands(cfg, world, cost_rate, val[world])
            band_rows += rows

    say("  training-length study on validation paths")
    study_jobs = [
        TrainJob(cfg, world, cfg.study_cost, alpha, seed, length)
        for world in WORLDS
        for i, alpha in enumerate(cfg.alphas)
        for length in cfg.length_grid[i]
        for seed in cfg.study_seeds
    ]
    hedges = train_many(study_jobs, workers, log)
    length_rows, iterations = study_lengths(cfg, hedges, val)
    del val

    say(f"  training the comparison networks, steps per level: {iterations}")
    jobs = [
        TrainJob(cfg, world, cost_rate, alpha, seed, iterations[alpha])
        for world in WORLDS
        for cost_rate in cfg.cost_rates
        for alpha in cfg.alphas
        for seed in cfg.seeds
    ]
    hedges.update(train_many([j for j in jobs if j.key not in hedges], workers, log))

    def net(world: str, cost_rate: float, alpha: float, seed: int) -> TrainedHedge:
        return hedges[(world, cost_rate, alpha, seed, iterations[alpha])]

    say("  scoring on test paths")
    scores: list[Row] = []
    paired: list[Row] = []
    kept: dict[tuple[str, str], FloatArray] = {}
    for tw in WORLDS:
        test = holdout_paths(cfg, tw)
        fixed_vol_leland: dict[float, FloatArray] = {}
        for dw in WORLDS:
            for dc in cfg.cost_rates:
                test_costs = cfg.cost_rates if dw == tw else (dc,)
                ref: dict[str, FloatArray] = {}
                strategies = classical_deltas(cfg, dw, dc, test, bands[(dw, dc)])
                if dw == tw:
                    strategies["unhedged"] = np.zeros_like(strategies[leland_name(dw)])
                for name, deltas in strategies.items():
                    for tc in test_costs:
                        loss, cost, turn = score(cfg, test, deltas, tc)
                        scores.append(
                            {
                                "test_world": tw,
                                "built_for": dw,
                                "design_cost": dc,
                                "test_cost": tc,
                                "strategy": name,
                                "training_seed": -1,
                                **stats_row(loss, cost, turn),
                            }
                        )
                        if tc == dc:
                            ref[name] = loss
                            if dw == tw and dc == cfg.study_cost:
                                kept[(tw, name)] = loss
                del strategies
                if dw == "black_scholes":
                    fixed_vol_leland[dc] = ref["leland"]
                for alpha in cfg.alphas:
                    versus = {
                        "band": ref[band_name(alpha)],
                        "leland": ref[leland_name(dw)],
                    }
                    if dw == tw:
                        versus["leland_fixed_vol"] = fixed_vol_leland[dc]
                    for seed in cfg.seeds:
                        deltas = network_deltas(cfg, net(dw, dc, alpha, seed), test)
                        for tc in test_costs:
                            loss, cost, turn = score(cfg, test, deltas, tc)
                            scores.append(
                                {
                                    "test_world": tw,
                                    "built_for": dw,
                                    "design_cost": dc,
                                    "test_cost": tc,
                                    "strategy": deep_name(alpha),
                                    "training_seed": seed,
                                    **stats_row(loss, cost, turn),
                                }
                            )
                            if tc != dc:
                                continue
                            if dw == tw and dc == cfg.study_cost and seed == cfg.seeds[0]:
                                kept[(tw, deep_name(alpha))] = loss
                            for other, other_loss in versus.items():
                                _, se = paired_block_difference(
                                    loss, other_loss, alpha, cfg.n_blocks
                                )
                                paired.append(
                                    {
                                        "test_world": tw,
                                        "built_for": dw,
                                        "cost_rate": dc,
                                        "alpha": alpha,
                                        "training_seed": seed,
                                        "versus": other,
                                        "difference": cvar(loss, alpha) - cvar(other_loss, alpha),
                                        "block_se": se,
                                    }
                                )
        say(f"  scored {tw} test paths")

    vol_scores: list[Row] = []
    vol_paired: list[Row] = []
    dt = cfg.maturity / cfg.n_steps
    for i, vol in enumerate(cfg.vol_shifts):
        prices = gbm_paths(
            cfg.n_test, cfg.n_steps, cfg.maturity, vol, cfg.seed + SEED_VOL_SHIFT + i, cfg.s0
        )
        test = PathSet(prices, None)
        for dc in cfg.cost_rates:
            strategies = classical_deltas(
                cfg, "black_scholes", dc, test, bands[("black_scholes", dc)]
            )
            strategies["bs_delta_true_vol"] = bs_deltas(prices, cfg.strike, cfg.maturity, vol)
            strategies["leland_true_vol"] = bs_deltas(
                prices, cfg.strike, cfg.maturity, leland_vol(vol, dc, dt)
            )
            ref = {}
            for name, deltas in strategies.items():
                loss, cost, turn = score(cfg, test, deltas, dc)
                ref[name] = loss
                vol_scores.append(
                    {
                        "test_vol": vol,
                        "cost_rate": dc,
                        "strategy": name,
                        "training_seed": -1,
                        **stats_row(loss, cost, turn),
                    }
                )
            del strategies
            for alpha in cfg.alphas:
                for seed in cfg.seeds:
                    deltas = network_deltas(cfg, net("black_scholes", dc, alpha, seed), test)
                    loss, cost, turn = score(cfg, test, deltas, dc)
                    vol_scores.append(
                        {
                            "test_vol": vol,
                            "cost_rate": dc,
                            "strategy": deep_name(alpha),
                            "training_seed": seed,
                            **stats_row(loss, cost, turn),
                        }
                    )
                    for other in ("band", "leland", "leland_true_vol"):
                        key = band_name(alpha) if other == "band" else other
                        _, se = paired_block_difference(loss, ref[key], alpha, cfg.n_blocks)
                        vol_paired.append(
                            {
                                "test_vol": vol,
                                "cost_rate": dc,
                                "alpha": alpha,
                                "training_seed": seed,
                                "versus": other,
                                "difference": cvar(loss, alpha) - cvar(ref[key], alpha),
                                "block_se": se,
                            }
                        )
        say(f"  scored Black-Scholes test paths at volatility {vol}")

    first: dict[tuple[str, float, float], TrainedHedge] = {
        (w, c, a): net(w, c, a, cfg.seeds[0])
        for w in WORLDS
        for c in cfg.cost_rates
        for a in cfg.alphas
    }
    return Comparison(
        scores, paired, band_rows, length_rows, iterations, vol_scores, vol_paired, kept, first
    )


# ------------------------------------------------------------ derived tables


def cvar_key(alpha: float) -> str:
    return f"cvar_{round(100 * alpha):d}"


def pick(rows: list[Row], **where: float | int | str) -> list[Row]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def seed_mean(rows: list[Row], key: str) -> tuple[float, float, float, float]:
    """Mean, standard deviation, minimum and maximum of ``key`` across the rows."""
    x = np.array([float(r[key]) for r in rows])
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return float(x.mean()), sd, float(x.min()), float(x.max())


def add_verdicts(
    row: Row, paired: list[Row], versus: tuple[str, ...], **where: float | int | str
) -> None:
    """Add the judged difference against each strategy in ``versus`` to ``row``."""
    for other in versus:
        rows = pick(paired, versus=other, **where)
        d, seed_sd, path_se, who = judge(
            [float(r["difference"]) for r in rows], [float(r["block_se"]) for r in rows]
        )
        row[f"minus_{other}"] = d
        row[f"minus_{other}_seed_sd"] = seed_sd
        row[f"minus_{other}_path_se"] = path_se
        row[f"lower_vs_{other}"] = who
        row[f"seeds_below_{other}"] = sum(1 for r in rows if float(r["difference"]) < 0.0)


def headline_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """In each world, the networks trained there against the hedges built for it.

    Each network is scored on the CVaR level it was trained on. Three
    comparisons are given: against the Leland hedge at 20% volatility (the
    baseline of the first version, which is the wrong volatility on Heston
    paths), against the Leland hedge at the volatility that prices the option,
    and against the no-trade band chosen on validation paths.
    """
    out: list[Row] = []
    for world in WORLDS:
        for cost_rate in cfg.cost_rates:
            here = pick(comp.scores, test_world=world, design_cost=cost_rate, test_cost=cost_rate)
            for alpha in cfg.alphas:
                key = cvar_key(alpha)
                nets = pick(here, built_for=world, strategy=deep_name(alpha))
                mean, sd, lo, hi = seed_mean(nets, key)
                band = pick(comp.bands, built_for=world, cost_rate=cost_rate, alpha=alpha)[0]
                row: Row = {
                    "world": world,
                    "cost_rate": cost_rate,
                    "alpha": alpha,
                    "network_cvar": mean,
                    "network_seed_sd": sd,
                    "network_min": lo,
                    "network_max": hi,
                    "seeds": len(nets),
                    "leland_fixed_vol_cvar": float(
                        pick(here, built_for="black_scholes", strategy="leland")[0][key]
                    ),
                    "leland_cvar": float(
                        pick(here, built_for=world, strategy=leland_name(world))[0][key]
                    ),
                    "band_cvar": float(
                        pick(here, built_for=world, strategy=band_name(alpha))[0][key]
                    ),
                    "band_target": band["target"],
                    "band_width": band["width"],
                }
                add_verdicts(
                    row,
                    comp.paired,
                    ("leland_fixed_vol", "leland", "band"),
                    test_world=world,
                    built_for=world,
                    cost_rate=cost_rate,
                    alpha=alpha,
                )
                out.append(row)
    return out


def count_lower(rows: list[Row], column: str) -> dict[str, int]:
    """How many cells each verdict got."""
    out = {"network": 0, "tie": 0, "other": 0}
    for r in rows:
        out[str(r[column])] += 1
    return out


def cross_world_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """Strategies built for one world and run in the other, beside those built for it."""
    out: list[Row] = []
    for tw in WORLDS:
        dw: World = "heston" if tw == "black_scholes" else "black_scholes"
        for cost_rate in cfg.cost_rates:
            here = pick(comp.scores, test_world=tw, design_cost=cost_rate, test_cost=cost_rate)
            for alpha in cfg.alphas:
                key = cvar_key(alpha)
                w_mean, w_sd, _, _ = seed_mean(
                    pick(here, built_for=dw, strategy=deep_name(alpha)), key
                )
                r_mean, r_sd, _, _ = seed_mean(
                    pick(here, built_for=tw, strategy=deep_name(alpha)), key
                )
                row: Row = {
                    "test_world": tw,
                    "built_for": dw,
                    "cost_rate": cost_rate,
                    "alpha": alpha,
                    "network_cvar": w_mean,
                    "network_seed_sd": w_sd,
                    "leland_cvar": float(
                        pick(here, built_for=dw, strategy=leland_name(dw))[0][key]
                    ),
                    "band_cvar": float(pick(here, built_for=dw, strategy=band_name(alpha))[0][key]),
                    "home_network_cvar": r_mean,
                    "home_network_seed_sd": r_sd,
                    "home_band_cvar": float(
                        pick(here, built_for=tw, strategy=band_name(alpha))[0][key]
                    ),
                }
                add_verdicts(
                    row,
                    comp.paired,
                    ("leland", "band"),
                    test_world=tw,
                    built_for=dw,
                    cost_rate=cost_rate,
                    alpha=alpha,
                )
                out.append(row)
    return out


def vol_shift_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """Strategies built for 20% volatility on Black-Scholes paths at other volatilities."""
    out: list[Row] = []
    for vol in cfg.vol_shifts:
        for cost_rate in cfg.cost_rates:
            here = pick(comp.vol_scores, test_vol=vol, cost_rate=cost_rate)
            for alpha in cfg.alphas:
                key = cvar_key(alpha)
                mean, sd, _, _ = seed_mean(pick(here, strategy=deep_name(alpha)), key)
                row: Row = {
                    "test_vol": vol,
                    "cost_rate": cost_rate,
                    "alpha": alpha,
                    "network_cvar": mean,
                    "network_seed_sd": sd,
                    "bs_delta_cvar": float(pick(here, strategy="bs_delta")[0][key]),
                    "leland_cvar": float(pick(here, strategy="leland")[0][key]),
                    "band_cvar": float(pick(here, strategy=band_name(alpha))[0][key]),
                    "leland_true_vol_cvar": float(pick(here, strategy="leland_true_vol")[0][key]),
                }
                add_verdicts(
                    row,
                    comp.vol_paired,
                    ("leland", "band", "leland_true_vol"),
                    test_vol=vol,
                    cost_rate=cost_rate,
                    alpha=alpha,
                )
                out.append(row)
    return out


def cost_shift_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """Strategies built for one cost rate and charged another, in their own world."""
    out: list[Row] = []
    for world in WORLDS:
        for alpha in cfg.alphas:
            key = cvar_key(alpha)
            for dc in cfg.cost_rates:
                for tc in cfg.cost_rates:
                    here = pick(
                        comp.scores, test_world=world, built_for=world, design_cost=dc, test_cost=tc
                    )
                    mean, sd, _, _ = seed_mean(pick(here, strategy=deep_name(alpha)), key)
                    out.append(
                        {
                            "world": world,
                            "alpha": alpha,
                            "design_cost": dc,
                            "test_cost": tc,
                            "network_cvar": mean,
                            "network_seed_sd": sd,
                            "leland_cvar": float(pick(here, strategy=leland_name(world))[0][key]),
                            "band_cvar": float(pick(here, strategy=band_name(alpha))[0][key]),
                        }
                    )
    return out


def comparison_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """Every strategy at the cost it was built for, networks averaged over seeds."""
    out: list[Row] = []
    stats = ("mean_loss", "std_loss", "cvar_50", "cvar_99", "mean_cost", "mean_turnover")
    for tw in WORLDS:
        for cost_rate in cfg.cost_rates:
            here = pick(comp.scores, test_world=tw, design_cost=cost_rate, test_cost=cost_rate)
            for dw in (tw, *(w for w in WORLDS if w != tw)):
                names = list(dict.fromkeys(str(r["strategy"]) for r in pick(here, built_for=dw)))
                for name in names:
                    rows = pick(here, built_for=dw, strategy=name)
                    row: Row = {
                        "test_world": tw,
                        "cost_rate": cost_rate,
                        "built_for": dw,
                        "strategy": name,
                        "seeds": len(rows) if int(rows[0]["training_seed"]) >= 0 else 0,
                    }
                    for s in stats:
                        row[s] = seed_mean(rows, s)[0]
                    row["cvar_50_seed_sd"] = seed_mean(rows, "cvar_50")[1]
                    row["cvar_99_seed_sd"] = seed_mean(rows, "cvar_99")[1]
                    out.append(row)
    return out


def with_device(cfg: RunConfig, device: str) -> RunConfig:
    return replace(cfg, device=device)
