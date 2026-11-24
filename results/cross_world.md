# Built for one world, run in the other

| Tested on | Built for | Cost (bp) | CVaR level | Leland | Band | Network | Network minus band | Lower | Network minus Leland | Lower | Network trained on the test world | Band built for the test world |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes paths | Heston | 0 | 50% | 2.618 | 3.134 | 2.992 ± 0.016 | -0.142 ± 0.016 | network | +0.374 ± 0.016 | formula | 2.556 ± 0.000 | 2.556 |
| Black-Scholes paths | Heston | 0 | 99% | 3.988 | 4.584 | 6.116 ± 0.061 | +1.532 ± 0.061 | formula | +2.128 ± 0.062 | formula | 3.269 ± 0.002 | 3.503 |
| Black-Scholes paths | Heston | 10 | 50% | 2.861 | 3.320 | 3.374 ± 0.017 | +0.054 ± 0.018 | formula | +0.513 ± 0.018 | formula | 2.759 ± 0.001 | 2.758 |
| Black-Scholes paths | Heston | 10 | 99% | 4.219 | 4.829 | 6.375 ± 0.077 | +1.546 ± 0.077 | formula | +2.156 ± 0.078 | formula | 3.487 ± 0.002 | 3.676 |
| Black-Scholes paths | Heston | 50 | 50% | 3.737 | 3.834 | 4.012 ± 0.012 | +0.177 ± 0.013 | formula | +0.275 ± 0.016 | formula | 3.313 ± 0.015 | 3.327 |
| Black-Scholes paths | Heston | 50 | 99% | 5.102 | 5.808 | 7.396 ± 0.073 | +1.588 ± 0.074 | formula | +2.294 ± 0.075 | formula | 4.246 ± 0.008 | 4.305 |
| Black-Scholes paths | Heston | 100 | 50% | 4.699 | 4.227 | 4.224 ± 0.012 | -0.002 ± 0.012 | tie | -0.475 ± 0.016 | network | 3.809 ± 0.012 | 3.856 |
| Black-Scholes paths | Heston | 100 | 99% | 6.122 | 6.661 | 8.471 ± 0.087 | +1.810 ± 0.087 | formula | +2.349 ± 0.088 | formula | 5.044 ± 0.008 | 5.039 |
| Heston paths | Black-Scholes | 0 | 50% | 2.674 | 2.674 | 2.672 ± 0.002 | -0.002 ± 0.002 | tie | -0.002 ± 0.002 | tie | 2.537 ± 0.000 | 2.567 |
| Heston paths | Black-Scholes | 0 | 99% | 7.979 | 7.853 | 7.430 ± 0.012 | -0.424 ± 0.020 | network | -0.549 ± 0.021 | network | 5.853 ± 0.003 | 6.134 |
| Heston paths | Black-Scholes | 10 | 50% | 2.879 | 2.835 | 2.845 ± 0.003 | +0.010 ± 0.003 | formula | -0.034 ± 0.003 | network | 2.668 ± 0.001 | 2.710 |
| Heston paths | Black-Scholes | 10 | 99% | 8.265 | 8.004 | 7.673 ± 0.018 | -0.330 ± 0.023 | network | -0.592 ± 0.025 | network | 6.050 ± 0.003 | 6.342 |
| Heston paths | Black-Scholes | 50 | 50% | 3.629 | 3.234 | 3.257 ± 0.011 | +0.023 ± 0.011 | formula | -0.372 ± 0.012 | network | 2.882 ± 0.002 | 2.994 |
| Heston paths | Black-Scholes | 50 | 99% | 9.363 | 8.837 | 8.527 ± 0.028 | -0.311 ± 0.031 | network | -0.836 ± 0.034 | network | 6.687 ± 0.007 | 6.993 |
| Heston paths | Black-Scholes | 100 | 50% | 4.467 | 3.607 | 3.533 ± 0.009 | -0.073 ± 0.009 | network | -0.934 ± 0.010 | network | 3.002 ± 0.002 | 3.196 |
| Heston paths | Black-Scholes | 100 | 99% | 10.645 | 9.789 | 9.376 ± 0.075 | -0.413 ± 0.077 | network | -1.269 ± 0.079 | network | 7.328 ± 0.013 | 7.644 |

Against the band: network lower in 6, tie in 2, formula lower in 8. Against Leland: network lower in 8, tie in 1, formula lower in 7.

Every strategy in the first seven columns was built for one world and run unchanged in the other. A hedge built for Black-Scholes uses 20% volatility and ignores the variance. A hedge built for Heston is told a variance of 0.04 on Black-Scholes paths. The last two columns are the strategies built for the test world, for scale. A ± after a network number is the standard deviation across training seeds. A ± after a difference is the noise in one trained network: the root sum of squares of the standard deviation across seeds and the standard error from the finite test set (paired, over 20 blocks of paths). 'network' or 'formula' is written only when the mean difference is more than twice that noise. Otherwise the cell is a tie.
