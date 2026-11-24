# Built for 20% volatility, run at another

| Test vol | Cost (bp) | CVaR level | Leland, 20% vol | Band, 20% vol | Network | Network minus band | Lower | Network minus Leland | Lower | Leland, true vol | Lower against Leland, true vol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15% | 0 | 50% | 1.974 | 1.974 | 1.969 ± 0.001 | -0.005 ± 0.001 | network | -0.005 ± 0.001 | network | 1.917 | formula |
| 15% | 0 | 99% | 2.460 | 2.530 | 2.672 ± 0.004 | +0.142 ± 0.005 | formula | +0.212 ± 0.005 | formula | 2.643 | formula |
| 15% | 10 | 50% | 2.177 | 2.149 | 2.157 ± 0.002 | +0.008 ± 0.002 | formula | -0.021 ± 0.002 | network | 2.141 | formula |
| 15% | 10 | 99% | 2.673 | 2.781 | 2.885 ± 0.004 | +0.104 ± 0.005 | formula | +0.212 ± 0.006 | formula | 2.862 | formula |
| 15% | 50 | 50% | 2.913 | 2.651 | 2.665 ± 0.004 | +0.014 ± 0.004 | formula | -0.248 ± 0.004 | network | 2.935 | network |
| 15% | 50 | 99% | 3.467 | 3.459 | 3.584 ± 0.004 | +0.126 ± 0.005 | formula | +0.117 ± 0.006 | formula | 3.684 | network |
| 15% | 100 | 50% | 3.726 | 3.127 | 3.078 ± 0.008 | -0.049 ± 0.008 | network | -0.647 ± 0.008 | network | 3.799 | network |
| 15% | 100 | 99% | 4.362 | 4.211 | 4.312 ± 0.009 | +0.101 ± 0.009 | formula | -0.050 ± 0.010 | network | 4.615 | network |
| 25% | 0 | 50% | 3.238 | 3.238 | 3.253 ± 0.002 | +0.015 ± 0.002 | formula | +0.015 ± 0.002 | formula | 3.196 | formula |
| 25% | 0 | 99% | 4.824 | 4.767 | 4.270 ± 0.005 | -0.497 ± 0.010 | network | -0.554 ± 0.009 | network | 4.394 | network |
| 25% | 10 | 50% | 3.479 | 3.451 | 3.457 ± 0.003 | +0.005 ± 0.003 | tie | -0.022 ± 0.003 | network | 3.423 | formula |
| 25% | 10 | 99% | 5.053 | 4.913 | 4.488 ± 0.006 | -0.425 ± 0.010 | network | -0.565 ± 0.009 | network | 4.616 | network |
| 25% | 50 | 50% | 4.359 | 4.049 | 4.044 ± 0.006 | -0.005 ± 0.006 | tie | -0.315 ± 0.006 | network | 4.261 | network |
| 25% | 50 | 99% | 5.930 | 5.540 | 5.277 ± 0.007 | -0.263 ± 0.009 | network | -0.653 ± 0.010 | network | 5.466 | network |
| 25% | 100 | 50% | 5.343 | 4.613 | 4.581 ± 0.012 | -0.032 ± 0.012 | network | -0.762 ± 0.012 | network | 5.203 | network |
| 25% | 100 | 99% | 6.956 | 6.274 | 6.114 ± 0.018 | -0.160 ± 0.019 | network | -0.842 ± 0.019 | network | 6.457 | network |

Against the band: network lower in 7, tie in 2, formula lower in 7. Against Leland at 20%: network lower in 12, tie in 0, formula lower in 4. Against Leland at the true volatility: network lower in 10, tie in 0, formula lower in 6.

Black-Scholes test paths at a volatility other than the 20% everything was built for. Networks, the band and the first Leland column are unchanged from the main comparison. 'Leland, true vol' knows the volatility of the test paths. A ± after a network number is the standard deviation across training seeds. A ± after a difference is the noise in one trained network: the root sum of squares of the standard deviation across seeds and the standard error from the finite test set (paired, over 20 blocks of paths). 'network' or 'formula' is written only when the mean difference is more than twice that noise. Otherwise the cell is a tie.
