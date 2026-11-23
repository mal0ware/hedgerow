# Every strategy on the same test paths

Short one at-the-money call, 30 daily rebalances, 200,000 test paths per world. Loss is minus terminal wealth with no premium received, so the mean loss of a costless hedge is the option value. Lower is better in every column.


## Black-Scholes paths, cost 0 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 2.277 | 3.454 | 4.555 | 16.275 | 0.000 | 0.000 |
| Black-Scholes delta | 2.286 | 0.358 | 2.555 | 3.515 | 0.000 | 2.224 |
| Leland delta | 2.286 | 0.358 | 2.555 | 3.515 | 0.000 | 2.224 |
| Network, trained on 50% CVaR | 2.286 | 0.361 | 2.556 | 3.557 | 0.000 | 2.252 |
| Network, trained on 99% CVaR | 2.286 | 0.451 | 2.644 | 3.266 | 0.000 | 2.010 |


## Black-Scholes paths, cost 10 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 2.277 | 3.454 | 4.555 | 16.275 | 0.000 | 0.000 |
| Black-Scholes delta | 2.508 | 0.373 | 2.787 | 3.847 | 0.222 | 2.224 |
| Leland delta | 2.502 | 0.369 | 2.781 | 3.737 | 0.217 | 2.166 |
| Network, trained on 50% CVaR | 2.465 | 0.395 | 2.762 | 3.815 | 0.179 | 1.792 |
| Network, trained on 99% CVaR | 2.474 | 0.467 | 2.846 | 3.484 | 0.188 | 1.885 |


## Black-Scholes paths, cost 50 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 2.277 | 3.454 | 4.555 | 16.275 | 0.000 | 0.000 |
| Black-Scholes delta | 3.398 | 0.507 | 3.787 | 5.229 | 1.112 | 2.224 |
| Leland delta | 3.282 | 0.418 | 3.601 | 4.576 | 0.997 | 1.994 |
| Network, trained on 50% CVaR | 2.842 | 0.628 | 3.316 | 4.928 | 0.556 | 1.112 |
| Network, trained on 99% CVaR | 3.075 | 0.544 | 3.511 | 4.254 | 0.789 | 1.581 |


## Black-Scholes paths, cost 100 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 2.277 | 3.454 | 4.555 | 16.275 | 0.000 | 0.000 |
| Black-Scholes delta | 4.510 | 0.755 | 5.109 | 7.017 | 2.224 | 2.224 |
| Leland delta | 4.138 | 0.485 | 4.510 | 5.545 | 1.852 | 1.853 |
| Network, trained on 50% CVaR | 3.086 | 1.008 | 3.821 | 7.131 | 0.802 | 0.803 |
| Network, trained on 99% CVaR | 3.632 | 0.649 | 4.159 | 5.058 | 1.347 | 1.348 |


## Heston paths, cost 0 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 1.704 | 2.190 | 3.205 | 12.982 | 0.000 | 0.000 |
| Black-Scholes delta | 1.706 | 1.349 | 2.673 | 7.931 | 0.000 | 1.572 |
| Leland delta | 1.706 | 1.349 | 2.673 | 7.931 | 0.000 | 1.572 |
| Black-Scholes delta, spot variance | 1.706 | 1.355 | 2.689 | 6.765 | 0.000 | 1.769 |
| Network, trained on 50% CVaR | 1.706 | 1.314 | 2.542 | 8.704 | 0.000 | 1.914 |
| Network, trained on 99% CVaR | 1.707 | 1.317 | 2.773 | 5.905 | 0.000 | 1.419 |


## Heston paths, cost 10 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 1.704 | 2.190 | 3.205 | 12.982 | 0.000 | 0.000 |
| Black-Scholes delta | 1.863 | 1.415 | 2.876 | 8.363 | 0.157 | 1.572 |
| Leland delta | 1.859 | 1.409 | 2.878 | 8.214 | 0.153 | 1.537 |
| Black-Scholes delta, spot variance | 1.883 | 1.377 | 2.885 | 7.001 | 0.177 | 1.769 |
| Network, trained on 50% CVaR | 1.789 | 1.398 | 2.672 | 9.606 | 0.084 | 0.847 |
| Network, trained on 99% CVaR | 1.812 | 1.314 | 2.869 | 6.100 | 0.106 | 1.059 |


## Heston paths, cost 50 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 1.704 | 2.190 | 3.205 | 12.982 | 0.000 | 0.000 |
| Black-Scholes delta | 2.490 | 1.688 | 3.695 | 10.114 | 0.784 | 1.572 |
| Leland delta | 2.421 | 1.634 | 3.627 | 9.302 | 0.715 | 1.435 |
| Black-Scholes delta, spot variance | 2.590 | 1.493 | 3.697 | 8.016 | 0.884 | 1.769 |
| Network, trained on 50% CVaR | 1.863 | 1.621 | 2.885 | 10.758 | 0.159 | 0.321 |
| Network, trained on 99% CVaR | 2.081 | 1.393 | 3.193 | 6.751 | 0.374 | 0.750 |


## Heston paths, cost 100 bp one way

| Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No hedge | 1.704 | 2.190 | 3.205 | 12.982 | 0.000 | 0.000 |
| Black-Scholes delta | 3.274 | 2.046 | 4.727 | 12.342 | 1.568 | 1.572 |
| Leland delta | 3.051 | 1.890 | 4.464 | 10.581 | 1.346 | 1.351 |
| Black-Scholes delta, spot variance | 3.474 | 1.691 | 4.767 | 9.420 | 1.768 | 1.769 |
| Network, trained on 50% CVaR | 1.893 | 1.754 | 3.008 | 11.378 | 0.189 | 0.191 |
| Network, trained on 99% CVaR | 2.384 | 1.465 | 3.541 | 7.383 | 0.678 | 0.679 |
