import numpy as np
import pytest
import torch

from hedgerow.deep import (
    DTYPE,
    Contract,
    HedgeNet,
    TrainConfig,
    cvar_loss,
    n_features,
    rollout,
    set_determinism,
    train,
)
from hedgerow.hedging import bs_deltas, hedge_pnl, payoff
from hedgerow.paths import HestonParams, gbm_paths, heston_paths
from hedgerow.risk import cvar

T = 30 / 365
K = 100.0

set_determinism(1)


def _constant_net(value: float, width: int) -> HedgeNet:
    net = HedgeNet(width, hidden=4, layers=1).to(DTYPE)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.body[-1].bias.fill_(value)  # type: ignore[operator, union-attr]
    return net


def test_feature_count() -> None:
    assert n_features(False) == 3
    assert n_features(True) == 4
    assert n_features(True, recurrent=False) == 3


def test_constant_holding_has_the_closed_form_loss() -> None:
    paths = gbm_paths(64, 10, T, 0.2, seed=0)
    contract = Contract(cost_rate=0.01)
    loss, deltas = rollout(
        _constant_net(0.4, 3), torch.as_tensor(paths, dtype=DTYPE), None, contract
    )
    assert torch.allclose(deltas, torch.full_like(deltas, 0.4))
    # buy 0.4 shares once at S0, hold to expiry
    expected = payoff(paths[:, -1], K) - 0.4 * (paths[:, -1] - 100.0) + 0.01 * 0.4 * 100.0
    np.testing.assert_allclose(loss.detach().numpy(), expected, atol=1e-4)


@pytest.mark.parametrize("cost", [0.0, 0.004])
def test_rollout_loss_equals_numpy_accounting(cost: float) -> None:
    torch.manual_seed(0)
    prices, variances = heston_paths(200, 12, T, HestonParams(), seed=1)
    net = HedgeNet(4, 8, 2).to(DTYPE)
    contract = Contract(cost_rate=cost)
    loss, deltas = rollout(
        net,
        torch.as_tensor(prices, dtype=DTYPE),
        torch.as_tensor(variances, dtype=DTYPE),
        contract,
    )
    d = deltas.detach().numpy().astype(np.float64)
    res = hedge_pnl(prices, d, payoff(prices[:, -1], K), 0.0, cost, maturity=T)
    np.testing.assert_allclose(loss.detach().numpy(), -res.pnl, atol=2e-4)


def test_costs_only_ever_add_to_the_loss() -> None:
    torch.manual_seed(1)
    p = torch.as_tensor(gbm_paths(100, 8, T, 0.2, seed=2), dtype=DTYPE)
    net = HedgeNet(3, 8, 2).to(DTYPE)
    free, _ = rollout(net, p, None, Contract(cost_rate=0.0))
    costly, _ = rollout(net, p, None, Contract(cost_rate=0.01))
    assert torch.all(costly >= free)


def test_cvar_loss_matches_numpy_cvar() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000)
    for alpha in (0.5, 0.9, 0.99):
        got = float(cvar_loss(torch.as_tensor(x, dtype=torch.float64), alpha))
        assert got == pytest.approx(cvar(x, alpha), abs=1e-9)
    assert float(cvar_loss(torch.as_tensor(x), 0.0)) == pytest.approx(x.mean())


def test_cvar_loss_gradient_flows_only_through_the_tail() -> None:
    x = torch.arange(10.0, dtype=torch.float64, requires_grad=True)
    cvar_loss(x, 0.8).backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), [0] * 8 + [0.5, 0.5], atol=1e-12)


def test_training_is_deterministic() -> None:
    paths = gbm_paths(2000, 10, T, 0.2, seed=4)
    cfg = TrainConfig(iterations=20, batch_size=256, hidden=8, layers=2, seed=5)
    contract = Contract(cost_rate=0.002)
    a = train(paths, None, contract, cfg)
    b = train(paths, None, contract, cfg)
    c = train(
        paths,
        None,
        contract,
        TrainConfig(iterations=20, batch_size=256, hidden=8, layers=2, seed=6),
    )
    np.testing.assert_array_equal(a.history, b.history)
    np.testing.assert_array_equal(a.deltas(paths), b.deltas(paths))
    assert not np.array_equal(a.history, c.history)


def test_training_lowers_the_objective() -> None:
    paths = gbm_paths(5000, 10, T, 0.2, seed=6)
    hedge = train(paths, None, Contract(), TrainConfig(iterations=150, batch_size=512, seed=0))
    assert hedge.history[-10:].mean() < 0.75 * hedge.history[:3].mean()


def test_policy_agrees_with_rollout() -> None:
    prices, variances = heston_paths(50, 6, T, HestonParams(), seed=7)
    hedge = train(
        prices, variances, Contract(cost_rate=0.001), TrainConfig(iterations=3, batch_size=50)
    )
    d = hedge.deltas(prices, variances)
    first = hedge.policy(prices[:, 0], 0, 6, prev=np.zeros(50), variance=variances[:, 0])
    np.testing.assert_allclose(first, d[:, 0], atol=1e-6)
    third = hedge.policy(prices[:, 2], 2, 6, prev=d[:, 1], variance=variances[:, 2])
    np.testing.assert_allclose(third, d[:, 2], atol=1e-6)
    with pytest.raises(ValueError):
        hedge.deltas(prices)


def test_network_recovers_black_scholes_delta_at_zero_cost() -> None:
    # Buehler et al. (2019), Section 5.2: without costs and with the 50% CVaR
    # the network hedge is close to the model hedge.
    train_paths = gbm_paths(40_000, 30, T, 0.2, seed=8)
    test_paths = gbm_paths(20_000, 30, T, 0.2, seed=9)
    hedge = train(
        train_paths, None, Contract(), TrainConfig(iterations=600, batch_size=1024, seed=0)
    )
    net_delta = hedge.deltas(test_paths)
    model_delta = bs_deltas(test_paths, K, T, 0.2)
    assert np.abs(net_delta - model_delta).mean() < 0.03
    z = payoff(test_paths[:, -1], K)
    net_loss = -hedge_pnl(test_paths, net_delta, z, maturity=T).pnl
    model_loss = -hedge_pnl(test_paths, model_delta, z, maturity=T).pnl
    assert cvar(net_loss, 0.5) == pytest.approx(cvar(model_loss, 0.5), rel=0.02)
    assert net_loss.std() < 1.25 * model_loss.std()
    # and it is far better than not hedging
    assert net_loss.std() < 0.3 * z.std()
