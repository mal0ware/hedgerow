from pathlib import Path

import numpy as np
import pytest

from hedgerow.deep import set_determinism
from hedgerow.experiments import (
    RunConfig,
    classical_deltas,
    discretisation_table,
    headline_row,
    leland_table,
    loglog_slope,
    make_world,
    one_interval_sample,
    one_interval_table,
    strategy_losses,
)
from hedgerow.hedging import payoff
from hedgerow.report import markdown_table, run_all

set_determinism(1)

TINY = RunConfig(
    cost_rates=(0.0, 0.005),
    n_train=1500,
    n_test=1500,
    n_blocks=5,
    iterations=(8, 8),
    batch_sizes=(256, 256),
    extra_seeds=(1,),
    n_classical=20_000,
    rebalance_grid=(8, 32, 128),
)


def test_worlds_are_functions_of_the_seed() -> None:
    for world in ("black_scholes", "heston"):
        a = make_world(TINY, world)  # type: ignore[arg-type]
        b = make_world(TINY, world)  # type: ignore[arg-type]
        np.testing.assert_array_equal(a.test, b.test)
        np.testing.assert_array_equal(a.train, b.train)
        assert not np.array_equal(a.train, a.test)
    assert make_world(TINY, "black_scholes").test_var is None
    assert make_world(TINY, "heston").test_var is not None


def test_every_strategy_is_scored_on_the_same_paths() -> None:
    # With the hedge gains and costs added back, every strategy owes the same
    # payoff on every path: they were all run on one set of prices.
    data = make_world(TINY, "heston")
    liability = payoff(data.test[:, -1], TINY.strike)
    increments = np.diff(data.test, axis=1)
    for name, deltas in classical_deltas(TINY, data, 0.005).items():
        loss, cost, _ = strategy_losses(TINY, data, 0.005, deltas)
        rebuilt = loss + (deltas * increments).sum(axis=1) - cost
        np.testing.assert_allclose(rebuilt, liability, atol=1e-9, err_msg=name)
    unhedged, _, turnover = strategy_losses(TINY, data, 0.005, np.zeros_like(increments))
    np.testing.assert_array_equal(unhedged, liability)
    assert np.all(turnover == 0.0)


def test_leland_equals_black_scholes_at_zero_cost() -> None:
    data = make_world(TINY, "black_scholes")
    d = classical_deltas(TINY, data, 0.0)
    np.testing.assert_array_equal(d["leland"], d["bs_delta"])
    assert "bs_delta_spot_vol" not in d


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


def test_headline_verdict() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(4000)
    noise = 0.01 * rng.standard_normal(4000)
    cfg = TINY
    for shift, expected in ((-0.5, "network"), (0.5, "leland"), (0.0, "tie")):
        losses = {
            ("w", 0.0, "bs_delta"): base,
            ("w", 0.0, "leland"): base,
            ("w", 0.0, "deep_cvar_50"): base + shift + (noise if shift == 0.0 else 0.0),
        }
        row = headline_row(cfg, "w", 0.0, 0.5, losses)
        assert row["lower_cvar"] == expected
        assert float(row["network_minus_leland"]) == pytest.approx(shift, abs=0.01)


def test_markdown_table() -> None:
    text = markdown_table(
        [{"a": "x", "b": 1.23456, "c": 7}], [("a", "A", 0), ("b", "B", 2), ("c", "C", 0)]
    )
    assert text.splitlines() == ["| A | B | C |", "| --- | ---: | ---: |", "| x | 1.23 | 7 |"]


def test_fast_config_is_small() -> None:
    fast = RunConfig.fast()
    assert fast.n_train <= 20_000
    assert max(fast.iterations) <= 200
    assert fast.train_config(0.99).batch_size == fast.batch_sizes[1]


def test_reproduce_twice_gives_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    small = RunConfig(
        cost_rates=(0.0, 0.005),
        n_train=1500,
        n_test=1500,
        n_blocks=5,
        iterations=(8, 8),
        batch_sizes=(256, 256),
        extra_seeds=(1,),
        n_classical=2000,
        rebalance_grid=(5, 10),
    )
    run_all(small, first)
    run_all(small, second)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert {"headline.csv", "comparison.csv", "fig_headline.png", "run_info.md"} <= set(names)
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
