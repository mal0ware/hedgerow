# Spread across training seeds

| Paths | Training seed | Network CVaR 50% | Leland CVaR 50% | Network std |
| --- | ---: | ---: | ---: | ---: |
| Black-Scholes paths | 0 | 3.3162 | 3.6009 | 0.6282 |
| Black-Scholes paths | 1 | 3.3302 | 3.6009 | 0.6412 |
| Black-Scholes paths | 2 | 3.3379 | 3.6009 | 0.6280 |
| Heston paths | 0 | 2.8848 | 3.6270 | 1.6207 |
| Heston paths | 1 | 2.8826 | 3.6270 | 1.6138 |
| Heston paths | 2 | 2.8812 | 3.6270 | 1.6173 |

Cost 50 bp. Same training and test paths; only the network's initial weights and batch order change.
