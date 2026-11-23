# Network against Leland

| Paths | Cost (bp) | CVaR level | Black-Scholes delta | Leland | Network | Network minus Leland | Lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes paths | 0 | 50% | 2.555 | 2.555 | 2.556 | +0.001 ± 0.000 | leland |
| Black-Scholes paths | 0 | 99% | 3.515 | 3.515 | 3.266 | -0.249 ± 0.009 | network |
| Black-Scholes paths | 10 | 50% | 2.787 | 2.781 | 2.762 | -0.019 ± 0.000 | network |
| Black-Scholes paths | 10 | 99% | 3.847 | 3.737 | 3.484 | -0.253 ± 0.009 | network |
| Black-Scholes paths | 50 | 50% | 3.787 | 3.601 | 3.316 | -0.285 ± 0.001 | network |
| Black-Scholes paths | 50 | 99% | 5.229 | 4.576 | 4.254 | -0.322 ± 0.009 | network |
| Black-Scholes paths | 100 | 50% | 5.109 | 4.510 | 3.821 | -0.689 ± 0.003 | network |
| Black-Scholes paths | 100 | 99% | 7.017 | 5.545 | 5.058 | -0.487 ± 0.010 | network |
| Heston paths | 0 | 50% | 2.673 | 2.673 | 2.542 | -0.131 ± 0.004 | network |
| Heston paths | 0 | 99% | 7.931 | 7.931 | 5.905 | -2.026 ± 0.041 | network |
| Heston paths | 10 | 50% | 2.876 | 2.878 | 2.672 | -0.206 ± 0.004 | network |
| Heston paths | 10 | 99% | 8.363 | 8.214 | 6.100 | -2.115 ± 0.040 | network |
| Heston paths | 50 | 50% | 3.695 | 3.627 | 2.885 | -0.742 ± 0.006 | network |
| Heston paths | 50 | 99% | 10.114 | 9.302 | 6.751 | -2.552 ± 0.044 | network |
| Heston paths | 100 | 50% | 4.727 | 4.464 | 3.008 | -1.456 ± 0.008 | network |
| Heston paths | 100 | 99% | 12.342 | 10.581 | 7.383 | -3.198 ± 0.047 | network |

Each network is scored on the CVaR level it was trained on. The ± is one standard error of the paired difference across 20 equal blocks of test paths; 'tie' means the difference is inside two standard errors. At zero cost the Leland delta is the Black-Scholes delta.
