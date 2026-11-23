"""A deep hedge: a small recurrent network trained to minimise the CVaR of the loss.

The setup follows Buehler, Gonon, Teichmann and Wood (2019). The hedger is
short one option with payoff Z and holds delta_k shares over [t_k, t_{k+1}].
The loss on a path is

    L = Z - sum_k delta_k (S_{k+1} - S_k) + sum_k eps S_k |delta_k - delta_{k-1}|,

which is minus the terminal wealth of their Eq. (2.1) with no premium. The
network maps the information at t_k and the previous holding to delta_k
(their Eq. (4.1)) and is trained on the CVaR of L (their Eq. (4.6) with the
loss function of Example 3.9).

Differences from the paper are listed in PAPER.md. The two that matter: one
network is shared across all dates with time to expiry as an input, where the
paper uses a separate network per date; and the inner minimisation over w in
Eq. (4.6) is solved exactly on each batch instead of by gradient steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

FloatArray = NDArray[np.float64]

DTYPE = torch.float32


@dataclass(frozen=True)
class Contract:
    """A short European call and the market it is hedged in."""

    strike: float = 100.0
    maturity: float = 30.0 / 365.0
    cost_rate: float = 0.0
    vol_scale: float = 0.2
    """Used only to scale network inputs: log-moneyness is divided by vol_scale * sqrt(T)."""
    var_scale: float = 0.04
    """Used only to scale the variance input, when there is one."""


@dataclass(frozen=True)
class TrainConfig:
    """Training settings. Every source of randomness is derived from ``seed``."""

    alpha: float = 0.5
    hidden: int = 32
    layers: int = 3
    iterations: int = 2000
    batch_size: int = 4096
    lr: float = 2e-3
    lr_final: float = 1e-4
    seed: int = 0
    device: str = "cpu"


class HedgeNet(nn.Module):
    """delta_k = F(log-moneyness, time to expiry, [variance], delta_{k-1}).

    A feed-forward network with ReLU activations and a linear output. The
    holding is unconstrained: it may be negative or above one.
    """

    def __init__(self, n_features: int, hidden: int = 32, layers: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = n_features
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*blocks)
        self.n_features = n_features

    def forward(self, features: Tensor) -> Tensor:
        out: Tensor = self.body(features)
        return out.squeeze(-1)


def n_features(use_variance: bool, recurrent: bool = True) -> int:
    """Input width: moneyness and time, plus variance, plus the previous holding."""
    return 2 + int(use_variance) + int(recurrent)


def features(
    spot: Tensor,
    step: int,
    n_steps: int,
    prev: Tensor | None,
    variance: Tensor | None,
    contract: Contract,
) -> Tensor:
    """Network inputs at rebalance date ``step``.

    Columns: log(S / K) / (vol_scale sqrt(T)); sqrt of the share of time left;
    V / var_scale if a variance is given; the previous holding if given.
    """
    money_scale = contract.vol_scale * float(np.sqrt(contract.maturity))
    cols = [
        torch.log(spot / contract.strike) / money_scale,
        torch.full_like(spot, float(np.sqrt(1.0 - step / n_steps))),
    ]
    if variance is not None:
        cols.append(variance / contract.var_scale)
    if prev is not None:
        cols.append(prev)
    return torch.stack(cols, dim=1)


def rollout(
    net: HedgeNet,
    prices: Tensor,
    variances: Tensor | None,
    contract: Contract,
    recurrent: bool = True,
) -> tuple[Tensor, Tensor]:
    """Run the network along each path.

    Returns ``(loss, deltas)``: the per-path loss L defined in the module
    docstring, shape ``(n_paths,)``, and the holdings, shape ``(n_paths, n_steps)``.
    """
    n_paths, n_cols = prices.shape
    n_steps = n_cols - 1
    prev = torch.zeros(n_paths, dtype=prices.dtype, device=prices.device)
    gains = torch.zeros_like(prev)
    costs = torch.zeros_like(prev)
    holdings: list[Tensor] = []
    for k in range(n_steps):
        var_k = None if variances is None else variances[:, k]
        x = features(prices[:, k], k, n_steps, prev if recurrent else None, var_k, contract)
        delta = net(x)
        costs = costs + contract.cost_rate * prices[:, k] * torch.abs(delta - prev)
        gains = gains + delta * (prices[:, k + 1] - prices[:, k])
        holdings.append(delta)
        prev = delta
    liability = torch.clamp(prices[:, -1] - contract.strike, min=0.0)
    return liability - gains + costs, torch.stack(holdings, dim=1)


def cvar_loss(losses: Tensor, alpha: float) -> Tensor:
    """Batch CVaR: min over w of w + mean(max(L - w, 0)) / (1 - alpha).

    The minimising w is an alpha-quantile of the batch. It is treated as a
    constant in the backward pass, which gives the right gradient because w is
    at its optimum (envelope theorem).
    """
    if alpha <= 0.0:
        return losses.mean()
    w = torch.quantile(losses.detach(), alpha, interpolation="lower")
    return w + torch.clamp(losses - w, min=0.0).mean() / (1.0 - alpha)


@dataclass
class TrainedHedge:
    """A trained network with the curve of its training objective."""

    net: HedgeNet
    contract: Contract
    config: TrainConfig
    use_variance: bool
    recurrent: bool
    history: FloatArray

    def deltas(self, prices: FloatArray, variances: FloatArray | None = None, chunk: int = 50_000) -> FloatArray:
        """Holdings on new paths, as a numpy array of shape ``(n_paths, n_steps)``."""
        if self.use_variance and variances is None:
            raise ValueError("this network was trained with the variance as an input")
        device = torch.device(self.config.device)
        out: list[FloatArray] = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, prices.shape[0], chunk):
                p = torch.as_tensor(prices[start : start + chunk], dtype=DTYPE, device=device)
                v = None
                if self.use_variance and variances is not None:
                    v = torch.as_tensor(variances[start : start + chunk], dtype=DTYPE, device=device)
                _, d = rollout(self.net, p, v, self.contract, self.recurrent)
                out.append(np.asarray(d.cpu().numpy(), dtype=np.float64))
        return np.concatenate(out, axis=0)

    def policy(
        self,
        spot: FloatArray,
        step: int,
        n_steps: int,
        prev: FloatArray | None = None,
        variance: FloatArray | None = None,
    ) -> FloatArray:
        """The holding the network chooses in one given state. Arrays broadcast."""
        device = torch.device(self.config.device)
        arrays = [np.asarray(spot, dtype=np.float64)]
        if self.recurrent:
            if prev is None:
                raise ValueError("a recurrent network needs the previous holding")
            arrays.append(np.asarray(prev, dtype=np.float64))
        if self.use_variance:
            if variance is None:
                raise ValueError("this network was trained with the variance as an input")
            arrays.append(np.asarray(variance, dtype=np.float64))
        flat = [np.ravel(a) for a in np.broadcast_arrays(*arrays)]
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        tensors = [torch.as_tensor(a, dtype=DTYPE, device=device) for a in flat]
        s_t = tensors[0]
        p_t = tensors[1] if self.recurrent else None
        v_t = tensors[-1] if self.use_variance else None
        self.net.eval()
        with torch.no_grad():
            out = self.net(features(s_t, step, n_steps, p_t, v_t, self.contract))
        return np.asarray(out.cpu().numpy(), dtype=np.float64).reshape(shape)


def train(
    prices: FloatArray,
    variances: FloatArray | None,
    contract: Contract,
    config: TrainConfig,
    recurrent: bool = True,
) -> TrainedHedge:
    """Fit a :class:`HedgeNet` to a fixed set of training paths.

    Minibatches are drawn without replacement inside each pass over the data,
    by a generator seeded from ``config.seed``. The learning rate decays
    geometrically from ``lr`` to ``lr_final``. On one device, the same inputs
    give the same network.
    """
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    use_variance = variances is not None
    net = HedgeNet(n_features(use_variance, recurrent), config.hidden, config.layers)
    net = net.to(device=device, dtype=DTYPE)
    p_all = torch.as_tensor(prices, dtype=DTYPE, device=device)
    v_all = None if variances is None else torch.as_tensor(variances, dtype=DTYPE, device=device)
    n_paths = p_all.shape[0]
    batch = min(config.batch_size, n_paths)

    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
    decay = (config.lr_final / config.lr) ** (1.0 / max(config.iterations - 1, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=decay)
    sampler = torch.Generator(device="cpu").manual_seed(config.seed + 1)

    history = np.empty(config.iterations)
    order = torch.randperm(n_paths, generator=sampler)
    cursor = 0
    net.train()
    for it in range(config.iterations):
        if cursor + batch > n_paths:
            order = torch.randperm(n_paths, generator=sampler)
            cursor = 0
        idx = order[cursor : cursor + batch].to(device)
        cursor += batch
        v_batch = None if v_all is None else v_all[idx]
        losses, _ = rollout(net, p_all[idx], v_batch, contract, recurrent)
        objective = cvar_loss(losses, config.alpha)
        optimiser.zero_grad(set_to_none=True)
        objective.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        scheduler.step()
        history[it] = float(objective.detach())
    return TrainedHedge(net, contract, config, use_variance, recurrent, history)


def set_determinism(threads: int = 1) -> None:
    """Pin the thread count and ask PyTorch for deterministic kernels."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
