# No-trade bands, chosen on validation paths

| Built for | Cost (bp) | CVaR level | Target delta | Half-width (shares) | Validation CVaR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | 0 | 50% | Black-Scholes delta, 20% vol | 0.000 | 2.555 |
| Black-Scholes | 0 | 99% | Black-Scholes delta, 20% vol | 0.020 | 3.495 |
| Black-Scholes | 10 | 50% | Black-Scholes delta, 20% vol | 0.030 | 2.757 |
| Black-Scholes | 10 | 99% | Leland delta, 20% vol | 0.030 | 3.664 |
| Black-Scholes | 50 | 50% | Black-Scholes delta, 20% vol | 0.100 | 3.324 |
| Black-Scholes | 50 | 99% | Leland delta, 20% vol | 0.030 | 4.292 |
| Black-Scholes | 100 | 50% | Black-Scholes delta, 20% vol | 0.150 | 3.852 |
| Black-Scholes | 100 | 99% | Leland delta, 20% vol | 0.030 | 5.028 |
| Heston | 0 | 50% | Heston minimum-variance delta | 0.000 | 2.569 |
| Heston | 0 | 99% | Heston minimum-variance delta | 0.020 | 6.158 |
| Heston | 10 | 50% | Heston minimum-variance delta | 0.050 | 2.710 |
| Heston | 10 | 99% | Heston minimum-variance delta | 0.030 | 6.360 |
| Heston | 50 | 50% | Heston delta | 0.500 | 2.994 |
| Heston | 50 | 99% | Heston minimum-variance delta | 0.075 | 6.992 |
| Heston | 100 | 50% | Heston delta | 0.600 | 3.194 |
| Heston | 100 | 99% | Heston minimum-variance delta | 0.100 | 7.636 |

The band hedger holds still while its holding is within the half-width of the target delta and otherwise trades to the nearest edge. For each world, cost and CVaR level the target (any formula delta built for that world) and the half-width (one of 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1) with the lowest CVaR on validation paths were kept. A half-width of zero is the plain formula hedge.
