# Networks against the formula hedges

| Paths | Cost (bp) | CVaR level | Leland, 20% vol | Leland | Band | Network | Network minus band | Lower | Network minus Leland | Lower | Lower against Leland, 20% vol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes paths | 0 | 50% | 2.556 | 2.556 | 2.556 | 2.556 ± 0.000 | -0.000 ± 0.000 | tie | -0.000 ± 0.000 | tie | tie |
| Black-Scholes paths | 0 | 99% | 3.520 | 3.520 | 3.503 | 3.269 ± 0.002 | -0.234 ± 0.008 | network | -0.251 ± 0.007 | network | network |
| Black-Scholes paths | 10 | 50% | 2.782 | 2.782 | 2.758 | 2.759 ± 0.001 | +0.001 ± 0.001 | tie | -0.023 ± 0.001 | network | network |
| Black-Scholes paths | 10 | 99% | 3.742 | 3.742 | 3.676 | 3.487 ± 0.002 | -0.188 ± 0.007 | network | -0.255 ± 0.007 | network | network |
| Black-Scholes paths | 50 | 50% | 3.602 | 3.602 | 3.327 | 3.313 ± 0.015 | -0.014 ± 0.015 | tie | -0.289 ± 0.015 | network | network |
| Black-Scholes paths | 50 | 99% | 4.586 | 4.586 | 4.305 | 4.246 ± 0.008 | -0.059 ± 0.010 | network | -0.340 ± 0.010 | network | network |
| Black-Scholes paths | 100 | 50% | 4.511 | 4.511 | 3.856 | 3.809 ± 0.012 | -0.047 ± 0.012 | network | -0.702 ± 0.012 | network | network |
| Black-Scholes paths | 100 | 99% | 5.557 | 5.557 | 5.039 | 5.044 ± 0.008 | +0.004 ± 0.010 | tie | -0.513 ± 0.011 | network | network |
| Heston paths | 0 | 50% | 2.674 | 2.647 | 2.567 | 2.537 ± 0.000 | -0.030 ± 0.003 | network | -0.110 ± 0.003 | network | network |
| Heston paths | 0 | 99% | 7.979 | 8.555 | 6.134 | 5.853 ± 0.003 | -0.280 ± 0.016 | network | -2.702 ± 0.049 | network | network |
| Heston paths | 10 | 50% | 2.879 | 2.872 | 2.710 | 2.668 ± 0.001 | -0.042 ± 0.002 | network | -0.204 ± 0.004 | network | network |
| Heston paths | 10 | 99% | 8.265 | 8.858 | 6.342 | 6.050 ± 0.003 | -0.292 ± 0.016 | network | -2.808 ± 0.048 | network | network |
| Heston paths | 50 | 50% | 3.629 | 3.683 | 2.994 | 2.882 ± 0.002 | -0.112 ± 0.004 | network | -0.801 ± 0.006 | network | network |
| Heston paths | 50 | 99% | 9.363 | 10.031 | 6.993 | 6.687 ± 0.007 | -0.306 ± 0.020 | network | -3.344 ± 0.048 | network | network |
| Heston paths | 100 | 50% | 4.467 | 4.578 | 3.196 | 3.002 ± 0.002 | -0.194 ± 0.003 | network | -1.576 ± 0.008 | network | network |
| Heston paths | 100 | 99% | 10.645 | 11.416 | 7.644 | 7.328 ± 0.013 | -0.316 ± 0.023 | network | -4.088 ± 0.052 | network | network |

Of 16 cells, against the band: network lower in 12, tie in 4, formula lower in 0. Against Leland: network lower in 15, tie in 1, formula lower in 0. Against Leland at 20% volatility: network lower in 15, tie in 1, formula lower in 0.

Each network is scored on the CVaR level it was trained on. 'Leland' uses the volatility that prices the call in that world: 20% on Black-Scholes paths and the implied volatility of 14.79% on Heston paths. 'Leland, 20% vol' is the wrong volatility on Heston paths. 'Band' is a formula delta with a no-trade band; target and width were chosen on validation paths for that CVaR level (see bands.md). A ± after a network number is the standard deviation across training seeds. A ± after a difference is the noise in one trained network: the root sum of squares of the standard deviation across seeds and the standard error from the finite test set (paired, over 20 blocks of paths). 'network' or 'formula' is written only when the mean difference is more than twice that noise. Otherwise the cell is a tie.
