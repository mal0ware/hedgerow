# Hedgerow

**What a hedge costs when you can only trade now and then, and whether a neural network does it better.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

The textbook hedge assumes you trade continuously and for free. You do neither. How large is the error, and can a network trained on the real costs beat the classical fix?

## What this will contain

- Delta hedging at discrete intervals, with the hedging error measured against theory
- Proportional transaction costs and Leland's adjusted volatility
- A deep-hedging network trained under the same costs with a CVaR objective
- All strategies compared on identical paths: first Black-Scholes, then Heston paths the classical hedge was not built for

## The result it is built to show

One table. Does the network beat Leland's hedge when the world is Black-Scholes? And when it is not?

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Boyle and Emanuel, "Discretely adjusted option hedges", Journal of Financial Economics 8(3), 1980
- Leland, "Option Pricing and Replication with Transactions Costs", Journal of Finance 40(5), 1985
- Buehler, Gonon, Teichmann and Wood, "Deep hedging", Quantitative Finance 19(8), 2019

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Simulated paths only.

## How it is tested

The maths is tested before any result is trusted.

- Hedging error shrinks as one over the square root of the number of rebalances
- With zero costs the network recovers the Black-Scholes delta
- The same seed gives the same paths to every strategy

## Layout

```
src/hedgerow/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Delta hedging at discrete intervals, with the hedging error measured against theory
- [ ] Proportional transaction costs and Leland's adjusted volatility
- [ ] A deep-hedging network trained under the same costs with a CVaR objective
- [ ] All strategies compared on identical paths: first Black-Scholes, then Heston paths the classical hedge was not built for
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- A network that wins on simulated paths has learned the simulator. Real markets are not the simulator.
- If the network does not beat Leland, the README will say so. That is a result too.
- This is research code. It is not investment advice and it does not trade.

## The name

A hedgerow is a hedge that took years to grow and still has gaps in it.

## Licence

MIT. See `LICENSE`.
