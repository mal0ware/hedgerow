"""Run every experiment and write the tables and figures.

Tables are written as CSV (full precision) and markdown (rounded). Figures
are PNG with no timestamp or version metadata, so a rerun on the same device
reproduces every file byte for byte.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from hedgerow import black_scholes as bs
from hedgerow.deep import DTYPE
from hedgerow.experiments import (
    WORLDS,
    Comparison,
    Row,
    RunConfig,
    band_name,
    comparison_rows,
    cost_shift_rows,
    count_lower,
    cross_world_rows,
    cvar_key,
    deep_name,
    discretisation_table,
    headline_rows,
    heston_implied_vol,
    leland_table,
    loglog_slope,
    one_interval_sample,
    one_interval_table,
    pick,
    run_comparison,
    seed_mean,
    vol_shift_rows,
)
from hedgerow.hedging import leland_vol
from hedgerow.heston import call_price

FloatArray = NDArray[np.float64]

# Okabe-Ito colours: distinguishable with the common colour-vision deficiencies.
COLOURS = {
    "unhedged": "#999999",
    "bs_delta": "#0072B2",
    "leland": "#E69F00",
    "band": "#009E73",
    "fixed_vol": "#999999",
    "network": "#D55E00",
    "network_alt": "#CC79A7",
    "home": "#56B4E9",
    "theory": "#000000",
}
LABELS = {
    "unhedged": "No hedge",
    "bs_delta": "Black-Scholes delta, 20% vol",
    "leland": "Leland delta, 20% vol",
    "bs_delta_iv": "Black-Scholes delta, implied vol",
    "leland_iv": "Leland delta, implied vol",
    "bs_delta_spot_vol": "Black-Scholes delta, spot variance",
    "heston_delta": "Heston delta",
    "heston_min_var": "Heston minimum-variance delta",
    "band_cvar_50": "No-trade band chosen for 50% CVaR",
    "band_cvar_99": "No-trade band chosen for 99% CVaR",
    "deep_cvar_50": "Network, trained on 50% CVaR",
    "deep_cvar_99": "Network, trained on 99% CVaR",
    "bs_delta_true_vol": "Black-Scholes delta, true vol",
    "leland_true_vol": "Leland delta, true vol",
}
WORLD_LABELS = {"black_scholes": "Black-Scholes paths", "heston": "Heston paths"}
BUILT_LABELS = {"black_scholes": "Black-Scholes", "heston": "Heston"}
VERDICTS = {"network": "network", "tie": "tie", "other": "formula"}


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def _cell(value: float | int | str, digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Row], columns: Sequence[tuple[str, str, int]]) -> str:
    """Render ``rows`` with ``columns`` given as (key, heading, decimals)."""
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    rule = "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(columns))) + "|"
    body = ["| " + " | ".join(_cell(r[k], d) for k, _, d in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body]) + "\n"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _bp(cost_rate: float) -> str:
    return f"{cost_rate * 1e4:g}"


def _pct(alpha: float) -> str:
    return f"{round(100 * alpha)}%"


def _pm(row: Row, mean: str, sd: str) -> str:
    return f"{float(row[mean]):.3f} ± {float(row[sd]):.3f}"


def _diff(row: Row, versus: str) -> str:
    """Mean difference with the noise the verdict is judged against."""
    noise = float(
        np.hypot(float(row[f"minus_{versus}_seed_sd"]), float(row[f"minus_{versus}_path_se"]))
    )
    return f"{float(row[f'minus_{versus}']):+.3f} ± {noise:.3f}"


# -------------------------------------------------------------------- figures


def _cost_axis(ax: Axes, cfg: RunConfig) -> NDArray[np.int_]:
    x = np.arange(len(cfg.cost_rates))
    ax.set_xticks(x, [_bp(c) for c in cfg.cost_rates])
    ax.grid(alpha=0.3)
    return x


def plot_headline(cfg: RunConfig, rows: Sequence[Row], path: Path) -> None:
    fig, axes = plt.subplots(
        len(cfg.alphas), 2, figsize=(9.0, 3.3 * len(cfg.alphas)), squeeze=False
    )
    for i, alpha in enumerate(cfg.alphas):
        for j, world in enumerate(WORLDS):
            ax = axes[i][j]
            x = _cost_axis(ax, cfg)
            cells = [
                pick(list(rows), world=world, cost_rate=c, alpha=alpha)[0] for c in cfg.cost_rates
            ]
            if world == "heston":
                ax.plot(
                    x,
                    [float(r["leland_fixed_vol_cvar"]) for r in cells],
                    marker="v",
                    color=COLOURS["fixed_vol"],
                    linestyle="--",
                    linewidth=1.2,
                    label="Leland delta at 20% vol (wrong vol here)",
                )
            ax.plot(
                x,
                [float(r["leland_cvar"]) for r in cells],
                marker="^",
                color=COLOURS["leland"],
                linewidth=1.6,
                label="Leland delta at the vol that prices the call",
            )
            ax.plot(
                x,
                [float(r["band_cvar"]) for r in cells],
                marker="s",
                color=COLOURS["band"],
                linewidth=1.6,
                label="Formula delta with a no-trade band",
            )
            mean = np.array([float(r["network_cvar"]) for r in cells])
            lo = np.array([float(r["network_min"]) for r in cells])
            hi = np.array([float(r["network_max"]) for r in cells])
            ax.errorbar(
                x,
                mean,
                yerr=np.vstack([mean - lo, hi - mean]),
                marker="o",
                color=COLOURS["network"],
                linewidth=1.6,
                capsize=3,
                label=f"Network, mean of {int(cells[0]['seeds'])} seeds (bar: min to max)",
            )
            ax.set_title(f"{WORLD_LABELS[world]}, {_pct(alpha)} CVaR", fontsize=10)
            if i == len(cfg.alphas) - 1:
                ax.set_xlabel("One-way transaction cost (basis points of traded value)")
            if j == 0:
                ax.set_ylabel("CVaR of hedging loss\n(price units, spot = 100)")
    axes[0][1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Cash needed to make a short at-the-money call acceptable: lower is better", fontsize=11
    )
    fig.tight_layout()
    _save(fig, path)


def plot_misspecification(
    cfg: RunConfig,
    head: Sequence[Row],
    cross: Sequence[Row],
    vol: Sequence[Row],
    path: Path,
) -> None:
    """Strategies run where they were not built: the other world, and another volatility."""
    fig, axes = plt.subplots(
        len(cfg.alphas), 3, figsize=(13.0, 3.3 * len(cfg.alphas)), squeeze=False
    )
    for i, alpha in enumerate(cfg.alphas):
        for j, tw in enumerate(WORLDS):
            ax = axes[i][j]
            x = _cost_axis(ax, cfg)
            cells = [
                pick(list(cross), test_world=tw, cost_rate=c, alpha=alpha)[0]
                for c in cfg.cost_rates
            ]
            built = BUILT_LABELS[str(cells[0]["built_for"])]
            ax.plot(
                x,
                [float(r["leland_cvar"]) for r in cells],
                marker="^",
                color=COLOURS["leland"],
                linewidth=1.6,
                label=f"Leland delta built for {built}",
            )
            ax.plot(
                x,
                [float(r["band_cvar"]) for r in cells],
                marker="s",
                color=COLOURS["band"],
                linewidth=1.6,
                label=f"Band built for {built}",
            )
            ax.errorbar(
                x,
                [float(r["network_cvar"]) for r in cells],
                yerr=[2.0 * float(r["network_seed_sd"]) for r in cells],
                marker="o",
                color=COLOURS["network"],
                linewidth=1.6,
                capsize=3,
                label=f"Network trained on {built}",
            )
            ax.plot(
                x,
                [float(r["home_network_cvar"]) for r in cells],
                marker="o",
                markerfacecolor="none",
                color=COLOURS["home"],
                linestyle=":",
                linewidth=1.4,
                label="Network trained on these paths",
            )
            ax.set_title(f"Tested on {WORLD_LABELS[tw]}, {_pct(alpha)} CVaR", fontsize=10)
            ax.legend(frameon=False, fontsize=7)
            if i == len(cfg.alphas) - 1:
                ax.set_xlabel("One-way transaction cost (basis points)")
            if j == 0:
                ax.set_ylabel("CVaR of hedging loss\n(price units, spot = 100)")
        ax = axes[i][2]
        home = pick(list(head), world="black_scholes", cost_rate=cfg.study_cost, alpha=alpha)[0]
        points: list[tuple[float, Row]] = [
            (float(r["test_vol"]), r)
            for r in pick(list(vol), cost_rate=cfg.study_cost, alpha=alpha)
        ]
        points.append((cfg.vol, {**home, "leland_true_vol_cvar": home["leland_cvar"]}))
        points.sort(key=lambda item: item[0])
        xs = [100.0 * v for v, _ in points]
        for key, colour, marker, label in (
            ("leland_cvar", COLOURS["leland"], "^", "Leland delta at 20% vol"),
            ("band_cvar", COLOURS["band"], "s", "Band built at 20% vol"),
            ("leland_true_vol_cvar", COLOURS["theory"], "x", "Leland delta at the true vol"),
        ):
            ax.plot(
                xs,
                [float(r[key]) for _, r in points],
                marker=marker,
                color=colour,
                linewidth=1.4 if key != "leland_true_vol_cvar" else 1.0,
                linestyle="-" if key != "leland_true_vol_cvar" else ":",
                label=label,
            )
        ax.errorbar(
            xs,
            [float(r["network_cvar"]) for _, r in points],
            yerr=[2.0 * float(r["network_seed_sd"]) for _, r in points],
            marker="o",
            color=COLOURS["network"],
            linewidth=1.6,
            capsize=3,
            label="Network trained at 20% vol",
        )
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.set_title(
            f"Black-Scholes paths, cost {_bp(cfg.study_cost)} bp, {_pct(alpha)} CVaR", fontsize=10
        )
        ax.legend(frameon=False, fontsize=7)
        if i == len(cfg.alphas) - 1:
            ax.set_xlabel("Volatility of the test paths (%)")
    fig.suptitle(
        "Hedges run where they were not built. Network bars are two standard deviations "
        "across seeds",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_discretisation(rows: Sequence[Row], slope: float, path: Path) -> None:
    n = np.array([float(r["rebalances"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(
        n,
        [float(r["predicted_std"]) for r in rows],
        color=COLOURS["theory"],
        linewidth=1.4,
        label="Sum of one-interval variances",
    )
    ax.loglog(
        n,
        [float(r["derman_kamal_std"]) for r in rows],
        color=COLOURS["leland"],
        linestyle="--",
        linewidth=1.4,
        label=r"Derman-Kamal rule $\sqrt{\pi/4}\,\kappa\,\sigma/\sqrt{N}$",
    )
    ax.loglog(
        n,
        [float(r["std_error"]) for r in rows],
        "o",
        color=COLOURS["bs_delta"],
        label=f"Simulated (fitted slope {slope:.3f})",
    )
    ax.set_xlabel("Number of rebalances N over 30 days")
    ax.set_ylabel("Std of hedging error (price units, spot = 100)")
    ax.set_title("Discrete delta hedging: the error shrinks as one over root N", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_one_interval(sample: FloatArray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(-1.0, 7.0, 81).tolist()
    ax.hist(
        sample, bins=bins, density=True, color=COLOURS["bs_delta"], alpha=0.55, label="Simulated"
    )
    x = np.linspace(-0.995, 7.0, 800)
    density = np.exp(-(x + 1.0) / 2.0) / np.sqrt(2.0 * np.pi * (x + 1.0))
    ax.plot(
        x,
        density,
        color=COLOURS["theory"],
        linewidth=1.4,
        label=r"Density of $u^2 - 1$, $u$ standard normal",
    )
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(
        r"One-interval gain of a hedged long call, in units of "
        r"$\lambda = \frac{1}{2}\Gamma S^2 \sigma^2 \Delta t$"
    )
    ax.set_ylabel("Density")
    ax.set_title("One rebalance interval: many small losses, a few large gains", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_leland(cfg: RunConfig, rows: Sequence[Row], path: Path) -> None:
    n = np.array([float(r["rebalances"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, key, title in (
        (axes[0], "mean_error", "Mean hedging error"),
        (axes[1], "std_error", "Std of hedging error"),
    ):
        ax.semilogx(
            n,
            [float(r[f"bs_{key}"]) for r in rows],
            "s-",
            color=COLOURS["bs_delta"],
            label="Black-Scholes price and delta",
        )
        ax.semilogx(
            n,
            [float(r[f"leland_{key}"]) for r in rows],
            "^-",
            color=COLOURS["leland"],
            label="Leland price and delta",
        )
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_xlabel("Number of rebalances N over 30 days")
        ax.set_ylabel(f"{title} (price units)")
        ax.set_title(f"{title}, cost {_bp(cfg.leland_cost)} bp one way", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_learned_delta(cfg: RunConfig, comp: Comparison, path: Path) -> None:
    alpha = cfg.alphas[0]
    free = comp.hedges[("black_scholes", cfg.cost_rates[0], alpha)]
    costly_rate = cfg.study_cost
    costly = comp.hedges[("black_scholes", costly_rate, alpha)]
    dt = cfg.maturity / cfg.n_steps
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))

    ax = axes[0]
    spot = np.linspace(88.0, 112.0, 241)
    steps = (0, cfg.n_steps // 2, cfg.n_steps - 3)
    shades = ("#0072B2", "#009E73", "#CC79A7")
    for step, shade in zip(steps, shades, strict=True):
        tau = cfg.maturity - step * dt
        model = bs.delta(spot, cfg.strike, tau, cfg.vol)
        ax.plot(
            spot,
            model,
            color=shade,
            linewidth=1.2,
            label=f"Formula, {cfg.n_steps - step} days left",
        )
        ax.plot(
            spot,
            free.policy(spot, step, cfg.n_steps, prev=model),
            color=shade,
            linestyle="--",
            linewidth=1.6,
            label=f"Network, {cfg.n_steps - step} days left",
        )
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Shares held")
    ax.set_title("No costs: the network finds the Black-Scholes delta", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    step = cfg.n_steps // 2
    tau = cfg.maturity - step * dt
    prev = np.linspace(0.0, 1.0, 201)
    ax.plot(prev, prev, color="#666666", linewidth=0.9, linestyle=":", label="No trade")
    for s, shade in zip((97.0, 100.0, 103.0), shades, strict=True):
        ax.plot(
            prev,
            costly.policy(np.full_like(prev, s), step, cfg.n_steps, prev=prev),
            color=shade,
            linewidth=1.6,
            label=f"Network, spot {s:g}",
        )
        target = float(bs.delta(s, cfg.strike, tau, leland_vol(cfg.vol, costly_rate, dt))[()])
        ax.plot([target], [target], marker="^", color=shade, markersize=7, linestyle="none")
    ax.plot([], [], marker="^", color="#444444", linestyle="none", label="Leland delta")
    ax.set_xlabel("Shares held before the rebalance")
    ax.set_ylabel("Shares held after")
    ax.set_title(
        f"Cost {_bp(costly_rate)} bp, {cfg.n_steps - step} days left: partial adjustment",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def plot_loss_distribution(cfg: RunConfig, comp: Comparison, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    first = band_name(cfg.alphas[0])
    names = {
        "black_scholes": ["leland", first, *(deep_name(a) for a in cfg.alphas)],
        "heston": ["leland_iv", first, *(deep_name(a) for a in cfg.alphas)],
    }
    colours = [COLOURS["leland"], COLOURS["band"], COLOURS["network"], COLOURS["network_alt"]]
    for ax, world in zip(axes, WORLDS, strict=True):
        pooled = np.concatenate([comp.losses[(world, n)] for n in names[world]])
        lo, hi = np.quantile(pooled, [0.001, 0.999])
        bins = np.linspace(lo, hi, 90).tolist()
        for name, colour in zip(names[world], colours, strict=False):
            ax.hist(
                comp.losses[(world, name)],
                bins=bins,
                histtype="step",
                density=True,
                color=colour,
                linewidth=1.4,
                label=LABELS[name],
            )
        ax.set_xlabel("Hedging loss before any premium (price units)")
        ax.set_ylabel("Density")
        ax.set_title(f"{WORLD_LABELS[world]}, cost {_bp(cfg.study_cost)} bp", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------- tables


def comparison_markdown(cfg: RunConfig, rows: Sequence[Row]) -> str:
    parts = [
        "# Every strategy on the same test paths\n",
        f"Short one at-the-money call, {cfg.n_steps} daily rebalances, "
        f"{cfg.n_test:,} test paths per world. "
        "Loss is minus terminal wealth with no premium received, so the mean loss of a costless "
        "hedge is the option value. Lower is better in every column. "
        f"Network rows are means over {len(cfg.seeds)} training seeds. "
        "'Built for' is the world a strategy was designed or trained in; "
        "when it differs from the test paths the strategy is being run in the wrong model.\n",
    ]
    cols = [
        ("built", "Built for", 0),
        ("label", "Strategy", 0),
        ("mean_loss", "Mean loss", 3),
        ("std_loss", "Std", 3),
        ("cvar_50", "CVaR 50%", 3),
        ("cvar_99", "CVaR 99%", 3),
        ("mean_cost", "Costs paid", 3),
        ("mean_turnover", "Shares traded", 3),
    ]
    for world in WORLDS:
        for cost_rate in cfg.cost_rates:
            subset: list[Row] = [
                {
                    **r,
                    "label": LABELS[str(r["strategy"])],
                    "built": BUILT_LABELS[str(r["built_for"])],
                }
                for r in pick(list(rows), test_world=world, cost_rate=cost_rate)
            ]
            parts.append(f"\n## {WORLD_LABELS[world]}, cost {_bp(cost_rate)} bp one way\n")
            parts.append(markdown_table(subset, cols))
    return "\n".join(parts)


RULE_NOTE = (
    "A ± after a network number is the standard deviation across training seeds. "
    "A ± after a difference is the noise in one trained network: the root sum of squares of "
    "the standard deviation across seeds and the standard error from the finite test set "
    "(paired, over 20 blocks of paths). 'network' or 'formula' is written only when the mean "
    "difference is more than twice that noise. Otherwise the cell is a tie."
)


def counts_sentence(rows: Sequence[Row], column: str) -> str:
    c = count_lower(list(rows), column)
    return f"network lower in {c['network']}, tie in {c['tie']}, formula lower in {c['other']}"


def headline_markdown(cfg: RunConfig, rows: Sequence[Row]) -> str:
    table: list[Row] = [
        {
            **r,
            "world": WORLD_LABELS[str(r["world"])],
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": _pct(float(r["alpha"])),
            "net": _pm(r, "network_cvar", "network_seed_sd"),
            "d_band": _diff(r, "band"),
            "v_band": VERDICTS[str(r["lower_vs_band"])],
            "d_leland": _diff(r, "leland"),
            "v_leland": VERDICTS[str(r["lower_vs_leland"])],
            "v_fixed": VERDICTS[str(r["lower_vs_leland_fixed_vol"])],
        }
        for r in rows
    ]
    cols = [
        ("world", "Paths", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("leland_fixed_vol_cvar", "Leland, 20% vol", 3),
        ("leland_cvar", "Leland", 3),
        ("band_cvar", "Band", 3),
        ("net", "Network", 0),
        ("d_band", "Network minus band", 0),
        ("v_band", "Lower", 0),
        ("d_leland", "Network minus Leland", 0),
        ("v_leland", "Lower", 0),
        ("v_fixed", "Lower against Leland, 20% vol", 0),
    ]
    n = len(rows)
    note = (
        f"\nOf {n} cells, against the band: {counts_sentence(rows, 'lower_vs_band')}. "
        f"Against Leland: {counts_sentence(rows, 'lower_vs_leland')}. "
        "Against Leland at 20% volatility: "
        f"{counts_sentence(rows, 'lower_vs_leland_fixed_vol')}.\n\n"
        "Each network is scored on the CVaR level it was trained on. "
        "'Leland' uses the volatility that prices the call in that world: 20% on Black-Scholes "
        f"paths and the implied volatility of {100 * heston_implied_vol(cfg):.2f}% on Heston "
        "paths. 'Leland, 20% vol' is the wrong volatility on Heston paths. "
        "'Band' is a formula delta with a no-trade band; target and width were chosen on "
        "validation paths for that CVaR level (see bands.md). " + RULE_NOTE + "\n"
    )
    return "# Networks against the formula hedges\n\n" + markdown_table(table, cols) + note


def cross_world_markdown(rows: Sequence[Row]) -> str:
    table: list[Row] = [
        {
            **r,
            "test": WORLD_LABELS[str(r["test_world"])],
            "built": BUILT_LABELS[str(r["built_for"])],
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": _pct(float(r["alpha"])),
            "net": _pm(r, "network_cvar", "network_seed_sd"),
            "home": _pm(r, "home_network_cvar", "home_network_seed_sd"),
            "d_band": _diff(r, "band"),
            "v_band": VERDICTS[str(r["lower_vs_band"])],
            "d_leland": _diff(r, "leland"),
            "v_leland": VERDICTS[str(r["lower_vs_leland"])],
        }
        for r in rows
    ]
    cols = [
        ("test", "Tested on", 0),
        ("built", "Built for", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("leland_cvar", "Leland", 3),
        ("band_cvar", "Band", 3),
        ("net", "Network", 0),
        ("d_band", "Network minus band", 0),
        ("v_band", "Lower", 0),
        ("d_leland", "Network minus Leland", 0),
        ("v_leland", "Lower", 0),
        ("home", "Network trained on the test world", 0),
        ("home_band_cvar", "Band built for the test world", 3),
    ]
    note = (
        f"\nAgainst the band: {counts_sentence(rows, 'lower_vs_band')}. "
        f"Against Leland: {counts_sentence(rows, 'lower_vs_leland')}.\n\n"
        "Every strategy in the first seven columns was built for one world and run unchanged in "
        "the other. A hedge built for Black-Scholes uses 20% volatility and ignores the "
        "variance. A hedge built for Heston is told a variance of 0.04 on Black-Scholes paths. "
        "The last two columns are the strategies built for the test world, for scale. "
        + RULE_NOTE
        + "\n"
    )
    return "# Built for one world, run in the other\n\n" + markdown_table(table, cols) + note


def vol_shift_markdown(cfg: RunConfig, rows: Sequence[Row]) -> str:
    table: list[Row] = [
        {
            **r,
            "vol": f"{100 * float(r['test_vol']):g}%",
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": _pct(float(r["alpha"])),
            "net": _pm(r, "network_cvar", "network_seed_sd"),
            "d_band": _diff(r, "band"),
            "v_band": VERDICTS[str(r["lower_vs_band"])],
            "d_leland": _diff(r, "leland"),
            "v_leland": VERDICTS[str(r["lower_vs_leland"])],
            "v_true": VERDICTS[str(r["lower_vs_leland_true_vol"])],
        }
        for r in rows
    ]
    cols = [
        ("vol", "Test vol", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("leland_cvar", "Leland, 20% vol", 3),
        ("band_cvar", "Band, 20% vol", 3),
        ("net", "Network", 0),
        ("d_band", "Network minus band", 0),
        ("v_band", "Lower", 0),
        ("d_leland", "Network minus Leland", 0),
        ("v_leland", "Lower", 0),
        ("leland_true_vol_cvar", "Leland, true vol", 3),
        ("v_true", "Lower against Leland, true vol", 0),
    ]
    note = (
        f"\nAgainst the band: {counts_sentence(rows, 'lower_vs_band')}. "
        f"Against Leland at 20%: {counts_sentence(rows, 'lower_vs_leland')}. "
        f"Against Leland at the true volatility: "
        f"{counts_sentence(rows, 'lower_vs_leland_true_vol')}.\n\n"
        f"Black-Scholes test paths at a volatility other than the {100 * cfg.vol:g}% everything "
        "was built for. Networks, the band and the first Leland column are unchanged from the "
        "main comparison. 'Leland, true vol' knows the volatility of the test paths. "
        + RULE_NOTE
        + "\n"
    )
    return "# Built for 20% volatility, run at another\n\n" + markdown_table(table, cols) + note


def cost_shift_markdown(cfg: RunConfig, rows: Sequence[Row]) -> str:
    parts = [
        "# Built for one cost rate, charged another\n",
        "Rows are the cost a strategy was built for and columns the cost it was charged, in "
        "basis points one way. Each cell is the CVaR at the level named in the heading. "
        "Networks are means over training seeds. The diagonal is the main comparison.\n",
    ]
    for world in WORLDS:
        for alpha in cfg.alphas:
            parts.append(f"\n## {WORLD_LABELS[world]}, {_pct(alpha)} CVaR\n")
            for key, label in (
                ("network_cvar", "Network"),
                ("band_cvar", "Band"),
                ("leland_cvar", "Leland"),
            ):
                table: list[Row] = []
                for dc in cfg.cost_rates:
                    row: Row = {"built": f"{label}, built for {_bp(dc)} bp"}
                    for tc in cfg.cost_rates:
                        cell = pick(
                            list(rows), world=world, alpha=alpha, design_cost=dc, test_cost=tc
                        )[0]
                        row[_bp(tc)] = float(cell[key])
                    table.append(row)
                cols = [("built", "Strategy", 0)] + [
                    (_bp(tc), f"Charged {_bp(tc)} bp", 3) for tc in cfg.cost_rates
                ]
                parts.append(markdown_table(table, cols))
    return "\n".join(parts)


def length_markdown(cfg: RunConfig, comp: Comparison) -> str:
    table: list[Row] = []
    for world in WORLDS:
        for i, alpha in enumerate(cfg.alphas):
            for length in cfg.length_grid[i]:
                rows = pick(comp.lengths, world=world, alpha=alpha, iterations=length)
                mean, sd, _, _ = seed_mean(rows, "validation_cvar")
                table.append(
                    {
                        "world": WORLD_LABELS[world],
                        "alpha_pct": _pct(alpha),
                        "iterations": length,
                        "cvar": f"{mean:.4f} ± {sd:.4f}",
                        "chosen": "yes" if comp.iterations[alpha] == length else "",
                    }
                )
    cols = [
        ("world", "Paths", 0),
        ("alpha_pct", "CVaR level", 0),
        ("iterations", "Training steps", 0),
        ("cvar", "Validation CVaR", 0),
        ("chosen", "Chosen", 0),
    ]
    note = (
        f"\nCost {_bp(cfg.study_cost)} bp, {len(cfg.study_seeds)} training seeds per row, scored "
        "on validation paths. The ± is the standard deviation across seeds. For each CVaR level "
        "the chosen length has the lowest validation CVaR after averaging over seeds and over "
        "the two worlds (each world's numbers divided by its best). Every network in the "
        "comparison was then trained for the chosen number of steps.\n"
    )
    return "# Training length, chosen on validation paths\n\n" + markdown_table(table, cols) + note


def bands_markdown(cfg: RunConfig, comp: Comparison) -> str:
    table: list[Row] = [
        {
            **r,
            "built": BUILT_LABELS[str(r["built_for"])],
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": _pct(float(r["alpha"])),
            "label": LABELS[str(r["target"])],
        }
        for r in comp.bands
    ]
    cols = [
        ("built", "Built for", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("label", "Target delta", 0),
        ("width", "Half-width (shares)", 3),
        ("validation_cvar", "Validation CVaR", 3),
    ]
    widths = ", ".join(f"{w:g}" for w in cfg.band_widths)
    note = (
        "\nThe band hedger holds still while its holding is within the half-width of the target "
        "delta and otherwise trades to the nearest edge. For each world, cost and CVaR level the "
        "target (any formula delta built for that world) and the half-width (one of "
        f"{widths}) with the lowest CVaR on validation paths were kept. A half-width of zero is "
        "the plain formula hedge.\n"
    )
    return "# No-trade bands, chosen on validation paths\n\n" + markdown_table(table, cols) + note


def seed_rows(cfg: RunConfig, comp: Comparison) -> list[Row]:
    """CVaR of every network in the main comparison, one row per training seed."""
    out: list[Row] = []
    for world in WORLDS:
        for cost_rate in cfg.cost_rates:
            for alpha in cfg.alphas:
                for r in pick(
                    comp.scores,
                    test_world=world,
                    built_for=world,
                    design_cost=cost_rate,
                    test_cost=cost_rate,
                    strategy=deep_name(alpha),
                ):
                    d = pick(
                        comp.paired,
                        test_world=world,
                        built_for=world,
                        cost_rate=cost_rate,
                        alpha=alpha,
                        training_seed=r["training_seed"],
                        versus="band",
                    )[0]
                    out.append(
                        {
                            "world": world,
                            "cost_rate": cost_rate,
                            "alpha": alpha,
                            "training_seed": r["training_seed"],
                            "network_cvar": r[cvar_key(alpha)],
                            "network_std": r["std_loss"],
                            "network_minus_band": d["difference"],
                        }
                    )
    return out


def seeds_markdown(cfg: RunConfig, rows: Sequence[Row]) -> str:
    table: list[Row] = [
        {
            **r,
            "world": WORLD_LABELS[str(r["world"])],
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": _pct(float(r["alpha"])),
        }
        for r in rows
    ]
    cols = [
        ("world", "Paths", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("training_seed", "Training seed", 0),
        ("network_cvar", "Network CVaR", 4),
        ("network_minus_band", "Network minus band", 4),
        ("network_std", "Network std", 4),
    ]
    note = (
        "\nSame training and test paths in every row of a cell. Only the initial weights of the "
        "network and the order of its batches change with the seed.\n"
    )
    return "# Every network, by training seed\n\n" + markdown_table(table, cols) + note


def run_info(cfg: RunConfig, comp: Comparison) -> str:
    price = call_price(cfg.s0, cfg.strike, cfg.maturity, cfg.heston.v0, cfg.heston, cfg.heston_grid)
    bs_price = float(bs.price(cfg.s0, cfg.strike, cfg.maturity, cfg.vol)[()])
    lines = [
        "# Run information",
        "",
        f"- device: {cfg.device}",
        f"- torch: {torch.__version__}",
        f"- numpy: {np.__version__}",
        f"- network dtype: {str(DTYPE).replace('torch.', '')}",
        "- torch threads per network: 1",
        f"- training paths per world: {cfg.n_train}",
        f"- validation paths per world: {cfg.n_val}",
        f"- test paths per world: {cfg.n_test}",
        f"- rebalances: {cfg.n_steps} over {cfg.maturity * 365:g} days",
        f"- cost rates: {', '.join(_bp(c) + ' bp' for c in cfg.cost_rates)}",
        f"- CVaR levels: {', '.join(str(a) for a in cfg.alphas)}",
        f"- training seeds: {', '.join(str(s) for s in cfg.seeds)}",
        "- training steps tried per level: "
        + "; ".join(", ".join(str(n) for n in grid) for grid in cfg.length_grid),
        "- training steps chosen per level: "
        + ", ".join(str(comp.iterations[a]) for a in cfg.alphas),
        f"- batch size per level: {', '.join(str(b) for b in cfg.batch_sizes)}",
        f"- networks trained: {networks_trained(cfg, comp)}",
        f"- Black-Scholes call value: {bs_price:.4f}",
        f"- Heston call value (Fourier inversion): {price:.4f}",
        f"- Heston implied volatility: {100 * heston_implied_vol(cfg):.2f}%",
        f"- base seed: {cfg.seed}",
        "",
    ]
    return "\n".join(lines)


def networks_trained(cfg: RunConfig, comp: Comparison) -> int:
    study = {
        (w, a, s, n)
        for w in WORLDS
        for i, a in enumerate(cfg.alphas)
        for n in cfg.length_grid[i]
        for s in cfg.study_seeds
    }
    main = len(WORLDS) * len(cfg.cost_rates) * len(cfg.alphas) * len(cfg.seeds)
    reused = sum(
        1
        for (_, a, s, n) in study
        if n == comp.iterations[a] and s in cfg.seeds and cfg.study_cost in cfg.cost_rates
    )
    return len(study) + main - reused


# ----------------------------------------------------------------------- main


def run_all(cfg: RunConfig, out: Path, log: bool = False, workers: int = 1) -> None:
    """Run every experiment and write every table and figure into ``out``."""
    out.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if log:
            print(message, flush=True)

    say("1/4 discretisation error")
    disc = discretisation_table(cfg)
    slope = loglog_slope(disc)
    write_csv(out / "discretisation.csv", disc)
    write_text(
        out / "discretisation.md",
        "# Hedging error against the number of rebalances\n\n"
        + markdown_table(
            disc,
            [
                ("rebalances", "Rebalances N", 0),
                ("mean_error", "Mean error", 4),
                ("std_error", "Std of error", 4),
                ("predicted_std", "Sum of interval variances", 4),
                ("derman_kamal_std", "Derman-Kamal rule", 4),
                ("std_times_root_n", "Std x root N", 3),
                ("std_pct_of_premium", "Std, % of premium", 1),
            ],
        )
        + f"\nFitted slope of log std on log N: {slope:.4f} (theory: -0.5).\n",
    )
    plot_discretisation(disc, slope, out / "fig_discretisation.png")

    say("2/4 one-interval error")
    sample, _ = one_interval_sample(cfg)
    one = one_interval_table(sample)
    write_csv(out / "one_interval.csv", one)
    write_text(
        out / "one_interval.md",
        "# One rebalance interval against the chi-square prediction\n\n"
        + markdown_table(
            one,
            [
                ("statistic", "Statistic", 0),
                ("simulated", "Simulated", 4),
                ("chi_square", "u^2 - 1", 4),
            ],
        ),
    )
    plot_one_interval(sample, out / "fig_one_interval.png")

    say("3/4 Leland under costs")
    leland = leland_table(cfg)
    write_csv(out / "leland.csv", leland)
    write_text(
        out / "leland.md",
        f"# Black-Scholes and Leland hedges at {_bp(cfg.leland_cost)} bp one-way cost\n\n"
        + markdown_table(
            leland,
            [
                ("rebalances", "Rebalances N", 0),
                ("leland_number", "Leland number", 3),
                ("leland_vol", "Adjusted vol", 4),
                ("bs_mean_error", "BS mean error", 3),
                ("leland_mean_error", "Leland mean error", 3),
                ("bs_std_error", "BS std", 3),
                ("leland_std_error", "Leland std", 3),
            ],
        )
        + "\nEach hedger charges the option price at its own volatility. "
        "The entry trade is excluded.\n",
    )
    plot_leland(cfg, leland, out / "fig_leland.png")

    say("4/4 strategy comparison (trains the networks)")
    comp = run_comparison(cfg, workers=workers, log=log)
    write_csv(out / "scores.csv", comp.scores)
    write_csv(out / "paired.csv", comp.paired)
    write_csv(out / "vol_shift_scores.csv", comp.vol_scores)
    write_csv(out / "training_length.csv", comp.lengths)
    write_text(out / "training_length.md", length_markdown(cfg, comp))
    write_csv(out / "bands.csv", comp.bands)
    write_text(out / "bands.md", bands_markdown(cfg, comp))

    rows = comparison_rows(cfg, comp)
    write_csv(out / "comparison.csv", rows)
    write_text(out / "comparison.md", comparison_markdown(cfg, rows))
    head = headline_rows(cfg, comp)
    write_csv(out / "headline.csv", head)
    write_text(out / "headline.md", headline_markdown(cfg, head))
    seeds = seed_rows(cfg, comp)
    write_csv(out / "seeds.csv", seeds)
    write_text(out / "seeds.md", seeds_markdown(cfg, seeds))
    cross = cross_world_rows(cfg, comp)
    write_csv(out / "cross_world.csv", cross)
    write_text(out / "cross_world.md", cross_world_markdown(cross))
    vol = vol_shift_rows(cfg, comp)
    write_csv(out / "vol_shift.csv", vol)
    write_text(out / "vol_shift.md", vol_shift_markdown(cfg, vol))
    cost = cost_shift_rows(cfg, comp)
    write_csv(out / "cost_shift.csv", cost)
    write_text(out / "cost_shift.md", cost_shift_markdown(cfg, cost))

    plot_headline(cfg, head, out / "fig_headline.png")
    plot_misspecification(cfg, head, cross, vol, out / "fig_misspecification.png")
    plot_learned_delta(cfg, comp, out / "fig_learned_delta.png")
    plot_loss_distribution(cfg, comp, out / "fig_loss_distribution.png")
    write_text(out / "run_info.md", run_info(cfg, comp))
    say(f"done: results in {out}")
