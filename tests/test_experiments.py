from pathlib import Path

import numpy as np
import pytest

from hedgerow.deep import set_determinism
from hedgerow.experiments import (
    WORLDS,
    RunConfig,
    TrainJob,
    band_deltas,
    band_name,
    choose_bands,
    classical_deltas,
    count_lower,
    cross_world_rows,
    deep_name,
    discretisation_table,
    formula_deltas,
    headline_rows,
    holdout_paths,
    judge,
    leland_table,
    loglog_slope,
    network_deltas,
    one_interval_sample,
    one_interval_table,
    pick,
    run_comparison,
    run_job,
    score,
    training_paths,
    validation_paths,
    vol_shift_rows,
)
from hedgerow.hedging import payoff
from hedgerow.heston import GridSpec
from hedgerow.report import markdown_table, run_all
from hedgerow.risk import cvar

set_determinism(1)

TINY = RunConfig(
    heston_grid=GridSpec(log2_n=11, n_money=101, n_var=12),
    cost_rates=(0.0, 0.005),
    n_train=1500,
    n_val=1500,
    n_test=1500,
    n_blocks=5,
    length_grid=((6, 8), (6, 8)),
    batch_sizes=(256, 256),
    seeds=(0, 1),
    study_seeds=(0,),
    band_widths=(0.0, 0.05, 0.2),
    vol_shifts=(0.25,),
    n_classical=20_000,
    rebalance_grid=(8, 32, 128),
)


def test_path_sets_are_functions_of_the_seed_and_do_not_overlap() -> None:
    for world in WORLDS:
        sets = [f(TINY, world) for f in (training_paths, validation_paths, holdout_paths)]
        again = [f(TINY, world) for f in (training_paths, validation_paths, holdout_paths)]
        for a, b in zip(sets, again, strict=True):
            np.testing.assert_array_equal(a.prices, b.prices)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.array_equal(sets[i].prices, sets[j].prices)
    assert holdout_paths(TINY, "black_scholes").variances is None
    assert holdout_paths(TINY, "heston").variances is not None


def test_every_strategy_is_scored_on_the_same_paths() -> None:
    # With the hedge gains and costs added back, every strategy owes the same
    # payoff on every path: they were all run on one set of prices.
    data = holdout_paths(TINY, "heston")
    liability = payoff(data.prices[:, -1], TINY.strike)
    increments = np.diff(data.prices, axis=1)
    for name, deltas in formula_deltas(TINY, "heston", 0.005, data).items():
        loss, cost, _ = score(TINY, data, deltas, 0.005)
        rebuilt = loss + (deltas * increments).sum(axis=1) - cost
        np.testing.assert_allclose(rebuilt, liability, atol=1e-9, err_msg=name)
    unhedged, _, turnover = score(TINY, data, np.zeros_like(increments), 0.005)
    np.testing.assert_array_equal(unhedged, liability)
    assert np.all(turnover == 0.0)


def test_formula_hedges_by_world() -> None:
    data = holdout_paths(TINY, "black_scholes")
    d = formula_deltas(TINY, "black_scholes", 0.0, data)
    assert set(d) == {"bs_delta", "leland"}
    np.testing.assert_array_equal(d["leland"], d["bs_delta"])
    # A hedge built for Heston still runs on Black-Scholes paths: it is told sigma^2.
    h = formula_deltas(TINY, "heston", 0.005, data)
    assert set(h) == {
        "bs_delta_iv",
        "leland_iv",
        "bs_delta_spot_vol",
        "heston_delta",
        "heston_min_var",
    }
    np.testing.assert_allclose(h["bs_delta_spot_vol"], d["bs_delta"], atol=1e-12)
    # The implied-vol hedges use a lower volatility than 20%, so they differ.
    assert not np.allclose(h["bs_delta_iv"], d["bs_delta"])


def test_band_holds_still_inside_and_trades_to_the_edge_outside() -> None:
    target = np.array([[0.50, 0.52, 0.70, 0.40, 0.41]])
    np.testing.assert_array_equal(band_deltas(target, 0.0), target)
    got = band_deltas(target, 0.05)
    np.testing.assert_allclose(got, [[0.45, 0.47, 0.65, 0.45, 0.45]])
    assert np.all(np.abs(got - target) <= 0.05 + 1e-12)
    data = holdout_paths(TINY, "black_scholes")
    plain = formula_deltas(TINY, "black_scholes", 0.005, data)["leland"]
    _, _, turn_plain = score(TINY, data, plain, 0.005)
    _, _, turn_band = score(TINY, data, band_deltas(plain, 0.05), 0.005)
    assert turn_band.mean() < turn_plain.mean()


def test_band_choice_is_the_validation_minimum() -> None:
    val = validation_paths(TINY, "black_scholes")
    bands, rows = choose_bands(TINY, "black_scholes", 0.005, val)
    for row in rows:
        alpha = float(row["alpha"])
        band = bands[alpha]
        assert band.width in TINY.band_widths
        for target, deltas in formula_deltas(TINY, "black_scholes", 0.005, val).items():
            for width in TINY.band_widths:
                loss, _, _ = score(TINY, val, band_deltas(deltas, width), 0.005)
                assert float(row["validation_cvar"]) <= cvar(loss, alpha) + 1e-12, target
    held = classical_deltas(TINY, "black_scholes", 0.005, val, bands)
    np.testing.assert_array_equal(
        held[band_name(0.5)], band_deltas(held[bands[0.5].target], bands[0.5].width)
    )


def test_networks_run_in_the_other_world() -> None:
    bs_net = run_job(TrainJob(TINY, "black_scholes", 0.005, 0.5, 0, 4))
    heston_net = run_job(TrainJob(TINY, "heston", 0.005, 0.5, 0, 4))
    assert not bs_net.use_variance
    assert heston_net.use_variance
    for world in WORLDS:
        data = holdout_paths(TINY, world)
        for net in (bs_net, heston_net):
            deltas = network_deltas(TINY, net, data)
            assert deltas.shape == (TINY.n_test, TINY.n_steps)
            assert np.all(np.isfinite(deltas))
    # Training is a function of the job.
    again = run_job(TrainJob(TINY, "black_scholes", 0.005, 0.5, 0, 4))
    data = holdout_paths(TINY, "black_scholes")
    np.testing.assert_array_equal(
        network_deltas(TINY, bs_net, data), network_deltas(TINY, again, data)
    )


def test_discretisation_table_has_slope_minus_one_half() -> None:
    rows = discretisation_table(TINY)
    assert loglog_slope(rows) == pytest.approx(-0.5, abs=0.03)
    for r in rows:
        assert abs(float(r["mean_error"])) < 4 * float(r["mean_se"])
        assert float(r["std_error"]) == pytest.approx(float(r["predicted_std"]), rel=0.06)


def test_one_interval_table_matches_chi_square() -> None:
    sample, lam = one_interval_sample(TINY)
    assert lam > 0
    rows = {str(r["statistic"]): r for r in one_interval_table(sample)}
    assert float(rows["std / lambda"]["simulated"]) == pytest.approx(np.sqrt(2), rel=0.05)
    assert float(rows["share of intervals with a loss"]["simulated"]) == pytest.approx(
        0.683, abs=0.015
    )
    assert float(rows["worst outcome / lambda"]["simulated"]) == pytest.approx(-1.0, abs=0.02)


def test_leland_table() -> None:
    rows = leland_table(TINY)
    for r in rows:
        assert float(r["leland_vol"]) > TINY.vol
        assert float(r["bs_mean_error"]) < -0.2
        assert abs(float(r["leland_mean_error"])) < 0.06
    # more rebalances cost the Black-Scholes hedger more
    assert float(rows[-1]["bs_mean_error"]) < float(rows[0]["bs_mean_error"])
    assert float(rows[-1]["leland_number"]) == pytest.approx(4.0 * float(rows[0]["leland_number"]))


def test_verdict_needs_to_clear_seed_noise() -> None:
    # Five seeds, all lower, but spread as wide as the gap: a tie.
    assert judge([-0.02, -0.01, -0.03, -0.001, -0.04], [0.001] * 5)[3] == "tie"
    # The same mean with a tight spread: a win.
    assert judge([-0.020, -0.021, -0.019, -0.020, -0.020], [0.001] * 5)[3] == "network"
    assert judge([0.020, 0.021, 0.019, 0.020, 0.020], [0.001] * 5)[3] == "other"
    # A tight spread across seeds does not help if the test set is too small to tell.
    assert judge([-0.020, -0.021, -0.019], [0.05] * 3)[3] == "tie"
    mean, seed_sd, path_se, _ = judge([1.0, 3.0], [0.1, 0.3])
    assert mean == pytest.approx(2.0)
    assert seed_sd == pytest.approx(np.sqrt(2.0))
    assert path_se == pytest.approx(0.2)
    # One seed has no spread to measure; only the path error counts.
    assert judge([-0.5], [0.1])[1] == 0.0


def test_comparison_tables() -> None:
    comp = run_comparison(TINY)
    cells = len(WORLDS) * len(TINY.cost_rates) * len(TINY.alphas)
    head = headline_rows(TINY, comp)
    assert len(head) == cells
    for column in ("lower_vs_band", "lower_vs_leland", "lower_vs_leland_fixed_vol"):
        assert sum(count_lower(head, column).values()) == cells
    for row in head:
        assert int(row["seeds"]) == len(TINY.seeds)
        assert float(row["network_min"]) <= float(row["network_cvar"]) <= float(row["network_max"])
        if row["world"] == "black_scholes":
            # On Black-Scholes paths 20% is the right volatility: one Leland hedge, not two.
            assert row["leland_cvar"] == row["leland_fixed_vol_cvar"]
            assert row["minus_leland"] == row["minus_leland_fixed_vol"]
        else:
            assert row["leland_cvar"] != row["leland_fixed_vol_cvar"]
    # Chosen training lengths come from the grid.
    for i, alpha in enumerate(TINY.alphas):
        assert comp.iterations[alpha] in TINY.length_grid[i]
    assert len(comp.lengths) == len(WORLDS) * sum(len(g) for g in TINY.length_grid)
    # Cross-world rows compare strategies built for the other world only.
    cross = cross_world_rows(TINY, comp)
    assert len(cross) == cells
    assert all(r["test_world"] != r["built_for"] for r in cross)
    # A strategy is charged other costs only in its own world.
    off_diagonal = [r for r in comp.scores if r["design_cost"] != r["test_cost"]]
    assert off_diagonal
    assert all(r["test_world"] == r["built_for"] for r in off_diagonal)
    # Every network row has a seed, every formula row does not.
    for r in comp.scores:
        assert (int(r["training_seed"]) >= 0) == str(r["strategy"]).startswith("deep_")
    vol = vol_shift_rows(TINY, comp)
    assert len(vol) == len(TINY.vol_shifts) * len(TINY.cost_rates) * len(TINY.alphas)
    # At the wrong volatility the hedge that knows the true one differs from the 20% hedge.
    assert all(r["leland_cvar"] != r["leland_true_vol_cvar"] for r in vol)
    assert pick(comp.scores, strategy=deep_name(0.5), training_seed=1)


def test_worker_processes_do_not_change_the_numbers() -> None:
    small = RunConfig(
        heston_grid=GridSpec(log2_n=11, n_money=101, n_var=12),
        cost_rates=(0.005,),
        alphas=(0.5,),
        n_train=1000,
        n_val=1000,
        n_test=1000,
        n_blocks=5,
        length_grid=((4, 6),),
        batch_sizes=(256,),
        seeds=(0, 1),
        study_seeds=(0,),
        band_widths=(0.0, 0.05),
        vol_shifts=(0.25,),
    )
    one = run_comparison(small, workers=1)
    two = run_comparison(small, workers=2)
    assert one.scores == two.scores
    assert one.paired == two.paired
    assert one.iterations == two.iterations


def test_markdown_table() -> None:
    text = markdown_table(
        [{"a": "x", "b": 1.23456, "c": 7}], [("a", "A", 0), ("b", "B", 2), ("c", "C", 0)]
    )
    assert text.splitlines() == ["| A | B | C |", "| --- | ---: | ---: |", "| x | 1.23 | 7 |"]


def test_fast_config_is_small() -> None:
    fast = RunConfig.fast()
    assert fast.n_train <= 20_000
    assert max(max(g) for g in fast.length_grid) <= 200
    assert fast.train_config(0.99, 0, 10).batch_size == fast.batch_sizes[1]
    full = RunConfig.full()
    assert len(full.seeds) >= 5
    assert set(full.study_seeds) <= set(full.seeds)


def test_reproduce_twice_gives_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    small = RunConfig(
        heston_grid=GridSpec(log2_n=11, n_money=101, n_var=12),
        cost_rates=(0.0, 0.005),
        n_train=1500,
        n_val=1500,
        n_test=1500,
        n_blocks=5,
        length_grid=((6, 8), (6, 8)),
        batch_sizes=(256, 256),
        seeds=(0, 1),
        study_seeds=(0,),
        band_widths=(0.0, 0.05),
        vol_shifts=(0.25,),
        n_classical=2000,
        rebalance_grid=(5, 10),
    )
    run_all(small, first)
    run_all(small, second)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert {
        "headline.csv",
        "comparison.csv",
        "cross_world.csv",
        "vol_shift.csv",
        "cost_shift.csv",
        "training_length.csv",
        "bands.csv",
        "fig_headline.png",
        "fig_misspecification.png",
        "run_info.md",
    } <= set(names)
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
