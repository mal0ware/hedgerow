# Training length, chosen on validation paths

| Paths | CVaR level | Training steps | Validation CVaR | Chosen |
| --- | ---: | ---: | ---: | ---: |
| Black-Scholes paths | 50% | 1500 | 3.3374 ± 0.0070 |  |
| Black-Scholes paths | 50% | 3000 | 3.3281 ± 0.0110 |  |
| Black-Scholes paths | 50% | 6000 | 3.3150 ± 0.0140 | yes |
| Black-Scholes paths | 99% | 1000 | 4.2724 ± 0.0051 |  |
| Black-Scholes paths | 99% | 2000 | 4.2586 ± 0.0046 |  |
| Black-Scholes paths | 99% | 4000 | 4.2468 ± 0.0024 | yes |
| Heston paths | 50% | 1500 | 2.8923 ± 0.0017 |  |
| Heston paths | 50% | 3000 | 2.8829 ± 0.0018 |  |
| Heston paths | 50% | 6000 | 2.8793 ± 0.0013 | yes |
| Heston paths | 99% | 1000 | 6.7721 ± 0.0353 |  |
| Heston paths | 99% | 2000 | 6.7317 ± 0.0173 |  |
| Heston paths | 99% | 4000 | 6.7085 ± 0.0083 | yes |

Cost 50 bp, 3 training seeds per row, scored on validation paths. The ± is the standard deviation across seeds. For each CVaR level the chosen length has the lowest validation CVaR after averaging over seeds and over the two worlds (each world's numbers divided by its best). Every network in the comparison was then trained for the chosen number of steps.
