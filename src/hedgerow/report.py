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
from matplotlib.figure import Figure
from numpy.typing import NDArray

from hedgerow import black_scholes as bs
from hedgerow.deep import DTYPE
from hedgerow.experiments import (
    Comparison,
    Row,
    RunConfig,
    deep_name,
    discretisation_table,
    leland_table,
    loglog_slope,
    one_interval_sample,
    one_interval_table,
    run_comparison,
)
from hedgerow.hedging import leland_vol

FloatArray = NDArray[np.float64]

# Okabe-Ito colours: distinguishable with the common colour-vision deficiencies.
COLOURS = {
    "unhedged": "#999999",
    "bs_delta": "#0072B2",
    "leland": "#E69F00",
    "bs_delta_spot_vol": "#56B4E9",
    "network": "#D55E00",
    "network_alt": "#009E73",
    "theory": "#000000",
}
LABELS = {
    "unhedged": "No hedge",
    "bs_delta": "Black-Scholes delta",
    "leland": "Leland delta",
    "bs_delta_spot_vol": "Black-Scholes delta, spot variance",
    "deep_cvar_50": "Network, trained on 50% CVaR",
    "deep_cvar_99": "Network, trained on 99% CVaR",
}
WORLD_LABELS = {"black_scholes": "Black-Scholes paths", "heston": "Heston paths"}


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


# -------------------------------------------------------------------- figures


def plot_headline(cfg: RunConfig, comp: Comparison, path: Path) -> None:
    by_key = {(str(r["world"]), float(r["cost_rate"]), float(r["alpha"])): r for r in comp.headline}
    worlds = ["black_scholes", "heston"]
    fig, axes = plt.subplots(
        len(cfg.alphas), 2, figsize=(9.0, 3.3 * len(cfg.alphas)), squeeze=False
    )
    x = np.arange(len(cfg.cost_rates))
    for i, alpha in enumerate(cfg.alphas):
        for j, world in enumerate(worlds):
            ax = axes[i][j]
            for key, colour, label, marker in (
                ("bs_delta_cvar", COLOURS["bs_delta"], "Black-Scholes delta", "s"),
                ("leland_cvar", COLOURS["leland"], "Leland delta", "^"),
                ("network_cvar", COLOURS["network"], "Network trained on this CVaR", "o"),
            ):
                y = [float(by_key[(world, c, alpha)][key]) for c in cfg.cost_rates]
                ax.plot(x, y, marker=marker, color=colour, label=label, linewidth=1.6)
            ax.set_xticks(x, [_bp(c) for c in cfg.cost_rates])
            ax.set_title(f"{WORLD_LABELS[world]}, {round(100 * alpha)}% CVaR", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(cfg.alphas) - 1:
                ax.set_xlabel("One-way transaction cost (basis points of traded value)")
            if j == 0:
                ax.set_ylabel("CVaR of hedging loss\n(price units, spot = 100)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Cash needed to make a short at-the-money call acceptable: lower is better", fontsize=11
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
    costly_rate = cfg.seed_study_cost
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
    cost_rate = cfg.seed_study_cost
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    names = ["bs_delta", "leland", *(deep_name(a) for a in cfg.alphas)]
    colours = [COLOURS["bs_delta"], COLOURS["leland"], COLOURS["network"], COLOURS["network_alt"]]
    for ax, world in zip(axes, ("black_scholes", "heston"), strict=True):
        pooled = np.concatenate([comp.losses[(world, cost_rate, n)] for n in names])
        lo, hi = np.quantile(pooled, [0.001, 0.999])
        bins = np.linspace(lo, hi, 90).tolist()
        for name, colour in zip(names, colours, strict=False):
            ax.hist(
                comp.losses[(world, cost_rate, name)],
                bins=bins,
                histtype="step",
                density=True,
                color=colour,
                linewidth=1.4,
                label=LABELS[name],
            )
        ax.set_xlabel("Hedging loss before any premium (price units)")
        ax.set_ylabel("Density")
        ax.set_title(f"{WORLD_LABELS[world]}, cost {_bp(cost_rate)} bp", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------- tables


def comparison_markdown(cfg: RunConfig, comp: Comparison) -> str:
    parts = [
        "# Every strategy on the same test paths\n",
        f"Short one at-the-money call, {cfg.n_steps} daily rebalances, "
        f"{cfg.n_test:,} test paths per world. "
        "Loss is minus terminal wealth with no premium received, so the mean loss of a costless "
        "hedge is the option value. Lower is better in every column.\n",
    ]
    cols = [
        ("strategy", "Strategy", 0),
        ("mean_loss", "Mean loss", 3),
        ("std_loss", "Std", 3),
        ("cvar_50", "CVaR 50%", 3),
        ("cvar_99", "CVaR 99%", 3),
        ("mean_cost", "Costs paid", 3),
        ("mean_turnover", "Shares traded", 3),
    ]
    for world in ("black_scholes", "heston"):
        for cost_rate in cfg.cost_rates:
            subset = [
                {**r, "strategy": LABELS[str(r["strategy"])]}
                for r in comp.rows
                if r["world"] == world and r["cost_rate"] == cost_rate
            ]
            parts.append(f"\n## {WORLD_LABELS[world]}, cost {_bp(cost_rate)} bp one way\n")
            parts.append(markdown_table(subset, cols))
    return "\n".join(parts)


def headline_markdown(comp: Comparison) -> str:
    rows = [
        {
            **r,
            "world": WORLD_LABELS[str(r["world"])],
            "cost_bp": _bp(float(r["cost_rate"])),
            "alpha_pct": f"{round(100 * float(r['alpha']))}%",
            "diff": f"{float(r['network_minus_leland']):+.3f} ± {float(r['block_se']):.3f}",
        }
        for r in comp.headline
    ]
    cols = [
        ("world", "Paths", 0),
        ("cost_bp", "Cost (bp)", 0),
        ("alpha_pct", "CVaR level", 0),
        ("bs_delta_cvar", "Black-Scholes delta", 3),
        ("leland_cvar", "Leland", 3),
        ("network_cvar", "Network", 3),
        ("diff", "Network minus Leland", 0),
        ("lower_cvar", "Lower", 0),
    ]
    note = (
        "\nEach network is scored on the CVaR level it was trained on. "
        "The ± is one standard error of "
        "the paired difference across 20 equal blocks of test paths; 'tie' means the difference is "
        "inside two standard errors. At zero cost the Leland delta is the Black-Scholes delta.\n"
    )
    return "# Network against Leland\n\n" + markdown_table(rows, cols) + note


def run_info(cfg: RunConfig) -> str:
    lines = [
        "# Run information",
        "",
        f"- device: {cfg.device}",
        f"- torch: {torch.__version__}",
        f"- numpy: {np.__version__}",
        f"- network dtype: {str(DTYPE).replace('torch.', '')}",
        f"- torch threads: {torch.get_num_threads()}",
        f"- training paths per world: {cfg.n_train}",
        f"- test paths per world: {cfg.n_test}",
        f"- rebalances: {cfg.n_steps} over {cfg.maturity * 365:g} days",
        f"- cost rates: {', '.join(_bp(c) + ' bp' for c in cfg.cost_rates)}",
        f"- CVaR levels: {', '.join(str(a) for a in cfg.alphas)}",
        f"- iterations per level: {', '.join(str(i) for i in cfg.iterations)}",
        f"- batch size per level: {', '.join(str(b) for b in cfg.batch_sizes)}",
        f"- base seed: {cfg.seed}",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------- main


def run_all(cfg: RunConfig, out: Path, log: bool = False) -> None:
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
    comp = run_comparison(cfg, log=log)
    write_csv(out / "comparison.csv", comp.rows)
    write_text(out / "comparison.md", comparison_markdown(cfg, comp))
    write_csv(out / "headline.csv", comp.headline)
    write_text(out / "headline.md", headline_markdown(comp))
    write_csv(out / "seeds.csv", comp.seeds)
    write_text(
        out / "seeds.md",
        "# Spread across training seeds\n\n"
        + markdown_table(
            [{**r, "world": WORLD_LABELS[str(r["world"])]} for r in comp.seeds],
            [
                ("world", "Paths", 0),
                ("training_seed", "Training seed", 0),
                ("network_cvar", "Network CVaR 50%", 4),
                ("leland_cvar", "Leland CVaR 50%", 4),
                ("network_std", "Network std", 4),
            ],
        )
        + f"\nCost {_bp(cfg.seed_study_cost)} bp. Same training and test paths; only the network's "
        "initial weights and batch order change.\n",
    )
    plot_headline(cfg, comp, out / "fig_headline.png")
    plot_learned_delta(cfg, comp, out / "fig_learned_delta.png")
    plot_loss_distribution(cfg, comp, out / "fig_loss_distribution.png")
    write_text(out / "run_info.md", run_info(cfg))
    say(f"done: results in {out}")
