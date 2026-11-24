# Every strategy on the same test paths

Short one at-the-money call, 30 daily rebalances, 200,000 test paths per world. Loss is minus terminal wealth with no premium received, so the mean loss of a costless hedge is the option value. Lower is better in every column. Network rows are means over 5 training seeds. 'Built for' is the world a strategy was designed or trained in; when it differs from the test paths the strategy is being run in the wrong model.


## Black-Scholes paths, cost 0 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | Black-Scholes delta, 20% vol | 2.286 | 0.358 | 2.556 | 3.520 | 0.000 | 2.222 |
| Black-Scholes | Leland delta, 20% vol | 2.286 | 0.358 | 2.556 | 3.520 | 0.000 | 2.222 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 2.286 | 0.358 | 2.556 | 3.520 | 0.000 | 2.222 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 2.286 | 0.374 | 2.572 | 3.503 | 0.000 | 1.799 |
| Black-Scholes | No hedge | 2.290 | 3.468 | 4.581 | 16.345 | 0.000 | 0.000 |
| Black-Scholes | Network, trained on 50% CVaR | 2.286 | 0.361 | 2.556 | 3.560 | 0.000 | 2.243 |
| Black-Scholes | Network, trained on 99% CVaR | 2.287 | 0.444 | 2.639 | 3.269 | 0.000 | 2.013 |
| Heston | Black-Scholes delta, implied vol | 2.286 | 0.441 | 2.618 | 3.988 | 0.000 | 2.469 |
| Heston | Leland delta, implied vol | 2.286 | 0.441 | 2.618 | 3.988 | 0.000 | 2.469 |
| Heston | Black-Scholes delta, spot variance | 2.286 | 0.358 | 2.556 | 3.520 | 0.000 | 2.222 |
| Heston | Heston delta | 2.286 | 0.697 | 2.846 | 4.412 | 0.000 | 2.756 |
| Heston | Heston minimum-variance delta | 2.286 | 1.000 | 3.134 | 4.456 | 0.000 | 2.221 |
| Heston | No-trade band chosen for 50% CVaR | 2.286 | 1.000 | 3.134 | 4.456 | 0.000 | 2.221 |
| Heston | No-trade band chosen for 99% CVaR | 2.286 | 1.006 | 3.139 | 4.584 | 0.000 | 1.820 |
| Heston | Network, trained on 50% CVaR | 2.286 | 0.868 | 2.992 | 4.606 | 0.000 | 2.944 |
| Heston | Network, trained on 99% CVaR | 2.285 | 1.695 | 3.703 | 6.116 | 0.000 | 1.809 |


## Black-Scholes paths, cost 10 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | Black-Scholes delta, 20% vol | 2.508 | 0.373 | 2.788 | 3.852 | 0.222 | 2.222 |
| Black-Scholes | Leland delta, 20% vol | 2.503 | 0.369 | 2.782 | 3.742 | 0.216 | 2.164 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 2.451 | 0.400 | 2.758 | 3.763 | 0.165 | 1.648 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 2.446 | 0.408 | 2.763 | 3.676 | 0.159 | 1.593 |
| Black-Scholes | No hedge | 2.290 | 3.468 | 4.581 | 16.345 | 0.000 | 0.000 |
| Black-Scholes | Network, trained on 50% CVaR | 2.463 | 0.394 | 2.759 | 3.810 | 0.176 | 1.764 |
| Black-Scholes | Network, trained on 99% CVaR | 2.475 | 0.463 | 2.843 | 3.487 | 0.188 | 1.879 |
| Heston | Black-Scholes delta, implied vol | 2.533 | 0.499 | 2.914 | 4.400 | 0.247 | 2.469 |
| Heston | Leland delta, implied vol | 2.525 | 0.445 | 2.861 | 4.219 | 0.239 | 2.393 |
| Heston | Black-Scholes delta, spot variance | 2.508 | 0.373 | 2.788 | 3.852 | 0.222 | 2.222 |
| Heston | Heston delta | 2.560 | 0.773 | 3.182 | 4.898 | 0.274 | 2.756 |
| Heston | Heston minimum-variance delta | 2.509 | 1.042 | 3.388 | 4.688 | 0.223 | 2.221 |
| Heston | No-trade band chosen for 50% CVaR | 2.429 | 1.058 | 3.320 | 5.029 | 0.143 | 1.425 |
| Heston | No-trade band chosen for 99% CVaR | 2.454 | 1.043 | 3.335 | 4.829 | 0.168 | 1.669 |
| Heston | Network, trained on 50% CVaR | 2.411 | 1.401 | 3.374 | 8.597 | 0.123 | 1.232 |
| Heston | Network, trained on 99% CVaR | 2.427 | 1.623 | 3.766 | 6.375 | 0.141 | 1.400 |


## Black-Scholes paths, cost 50 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | Black-Scholes delta, 20% vol | 3.397 | 0.507 | 3.786 | 5.237 | 1.111 | 2.222 |
| Black-Scholes | Leland delta, 20% vol | 3.283 | 0.419 | 3.602 | 4.586 | 0.996 | 1.992 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 2.829 | 0.628 | 3.327 | 4.655 | 0.541 | 1.082 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 3.005 | 0.482 | 3.383 | 4.305 | 0.717 | 1.435 |
| Black-Scholes | No hedge | 2.290 | 3.468 | 4.581 | 16.345 | 0.000 | 0.000 |
| Black-Scholes | Network, trained on 50% CVaR | 2.844 | 0.655 | 3.313 | 5.581 | 0.557 | 1.114 |
| Black-Scholes | Network, trained on 99% CVaR | 3.075 | 0.536 | 3.504 | 4.246 | 0.787 | 1.575 |
| Heston | Black-Scholes delta, implied vol | 3.520 | 0.773 | 4.136 | 6.101 | 1.234 | 2.469 |
| Heston | Leland delta, implied vol | 3.376 | 0.475 | 3.737 | 5.102 | 1.089 | 2.179 |
| Heston | Black-Scholes delta, spot variance | 3.397 | 0.507 | 3.786 | 5.237 | 1.111 | 2.222 |
| Heston | Heston delta | 3.657 | 1.110 | 4.560 | 6.901 | 1.371 | 2.756 |
| Heston | Heston minimum-variance delta | 3.401 | 1.252 | 4.421 | 6.010 | 1.115 | 2.221 |
| Heston | No-trade band chosen for 50% CVaR | 2.481 | 1.871 | 3.834 | 9.851 | 0.190 | 0.376 |
| Heston | No-trade band chosen for 99% CVaR | 2.887 | 1.199 | 3.893 | 5.808 | 0.601 | 1.194 |
| Heston | Network, trained on 50% CVaR | 2.491 | 2.418 | 4.012 | 12.874 | 0.201 | 0.407 |
| Heston | Network, trained on 99% CVaR | 2.759 | 1.614 | 4.028 | 7.396 | 0.472 | 0.935 |


## Black-Scholes paths, cost 100 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | Black-Scholes delta, 20% vol | 4.508 | 0.756 | 5.107 | 7.029 | 2.221 | 2.222 |
| Black-Scholes | Leland delta, 20% vol | 4.138 | 0.486 | 4.511 | 5.557 | 1.851 | 1.852 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 3.172 | 0.860 | 3.856 | 5.689 | 0.883 | 0.880 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 3.601 | 0.577 | 4.057 | 5.039 | 1.313 | 1.314 |
| Black-Scholes | No hedge | 2.290 | 3.468 | 4.581 | 16.345 | 0.000 | 0.000 |
| Black-Scholes | Network, trained on 50% CVaR | 3.092 | 0.973 | 3.809 | 6.880 | 0.804 | 0.804 |
| Black-Scholes | Network, trained on 99% CVaR | 3.639 | 0.633 | 4.151 | 5.044 | 1.351 | 1.352 |
| Heston | Black-Scholes delta, implied vol | 4.754 | 1.156 | 5.695 | 8.288 | 2.469 | 2.469 |
| Heston | Leland delta, implied vol | 4.299 | 0.524 | 4.699 | 6.122 | 2.012 | 2.012 |
| Heston | Black-Scholes delta, spot variance | 4.508 | 0.756 | 5.107 | 7.029 | 2.221 | 2.222 |
| Heston | Heston delta | 5.029 | 1.564 | 6.313 | 9.466 | 2.743 | 2.756 |
| Heston | Heston minimum-variance delta | 4.516 | 1.577 | 5.755 | 7.999 | 2.230 | 2.221 |
| Heston | No-trade band chosen for 50% CVaR | 2.571 | 2.334 | 4.227 | 11.641 | 0.281 | 0.276 |
| Heston | No-trade band chosen for 99% CVaR | 3.312 | 1.376 | 4.464 | 6.661 | 1.025 | 1.016 |
| Heston | Network, trained on 50% CVaR | 2.512 | 2.719 | 4.224 | 14.047 | 0.222 | 0.225 |
| Heston | Network, trained on 99% CVaR | 3.071 | 1.724 | 4.380 | 8.471 | 0.783 | 0.777 |


## Heston paths, cost 0 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heston | Black-Scholes delta, implied vol | 1.708 | 1.378 | 2.647 | 8.555 | 0.000 | 1.721 |
| Heston | Leland delta, implied vol | 1.708 | 1.378 | 2.647 | 8.555 | 0.000 | 1.721 |
| Heston | Black-Scholes delta, spot variance | 1.707 | 1.350 | 2.688 | 6.720 | 0.000 | 1.769 |
| Heston | Heston delta | 1.707 | 1.570 | 2.840 | 7.419 | 0.000 | 1.990 |
| Heston | Heston minimum-variance delta | 1.707 | 1.147 | 2.567 | 6.144 | 0.000 | 1.581 |
| Heston | No-trade band chosen for 50% CVaR | 1.707 | 1.147 | 2.567 | 6.144 | 0.000 | 1.581 |
| Heston | No-trade band chosen for 99% CVaR | 1.707 | 1.151 | 2.574 | 6.134 | 0.000 | 1.330 |
| Heston | No hedge | 1.705 | 2.181 | 3.209 | 12.799 | 0.000 | 0.000 |
| Heston | Network, trained on 50% CVaR | 1.706 | 1.282 | 2.537 | 8.182 | 0.000 | 2.180 |
| Heston | Network, trained on 99% CVaR | 1.707 | 1.358 | 2.810 | 5.853 | 0.000 | 1.752 |
| Black-Scholes | Black-Scholes delta, 20% vol | 1.708 | 1.349 | 2.674 | 7.979 | 0.000 | 1.572 |
| Black-Scholes | Leland delta, 20% vol | 1.708 | 1.349 | 2.674 | 7.979 | 0.000 | 1.572 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 1.708 | 1.349 | 2.674 | 7.979 | 0.000 | 1.572 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 1.708 | 1.342 | 2.676 | 7.853 | 0.000 | 1.333 |
| Black-Scholes | Network, trained on 50% CVaR | 1.708 | 1.352 | 2.672 | 8.036 | 0.000 | 1.586 |
| Black-Scholes | Network, trained on 99% CVaR | 1.708 | 1.347 | 2.701 | 7.430 | 0.000 | 1.422 |


## Heston paths, cost 10 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heston | Black-Scholes delta, implied vol | 1.880 | 1.456 | 2.871 | 9.054 | 0.172 | 1.721 |
| Heston | Leland delta, implied vol | 1.875 | 1.440 | 2.872 | 8.858 | 0.167 | 1.675 |
| Heston | Black-Scholes delta, spot variance | 1.884 | 1.372 | 2.884 | 6.957 | 0.177 | 1.769 |
| Heston | Heston delta | 1.905 | 1.622 | 3.090 | 7.715 | 0.198 | 1.990 |
| Heston | Heston minimum-variance delta | 1.865 | 1.174 | 2.744 | 6.414 | 0.158 | 1.581 |
| Heston | No-trade band chosen for 50% CVaR | 1.817 | 1.182 | 2.710 | 6.361 | 0.110 | 1.094 |
| Heston | No-trade band chosen for 99% CVaR | 1.831 | 1.172 | 2.714 | 6.342 | 0.124 | 1.237 |
| Heston | No hedge | 1.705 | 2.181 | 3.209 | 12.799 | 0.000 | 0.000 |
| Heston | Network, trained on 50% CVaR | 1.790 | 1.340 | 2.668 | 8.857 | 0.085 | 0.854 |
| Heston | Network, trained on 99% CVaR | 1.824 | 1.326 | 2.897 | 6.050 | 0.117 | 1.175 |
| Black-Scholes | Black-Scholes delta, 20% vol | 1.865 | 1.415 | 2.877 | 8.415 | 0.157 | 1.572 |
| Black-Scholes | Leland delta, 20% vol | 1.862 | 1.409 | 2.879 | 8.265 | 0.153 | 1.538 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 1.832 | 1.390 | 2.835 | 8.143 | 0.125 | 1.250 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 1.829 | 1.387 | 2.839 | 8.004 | 0.122 | 1.218 |
| Black-Scholes | Network, trained on 50% CVaR | 1.839 | 1.394 | 2.845 | 8.167 | 0.132 | 1.317 |
| Black-Scholes | Network, trained on 99% CVaR | 1.842 | 1.403 | 2.883 | 7.673 | 0.134 | 1.346 |


## Heston paths, cost 50 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heston | Black-Scholes delta, implied vol | 2.567 | 1.778 | 3.775 | 11.082 | 0.859 | 1.721 |
| Heston | Leland delta, implied vol | 2.480 | 1.676 | 3.683 | 10.031 | 0.771 | 1.547 |
| Heston | Black-Scholes delta, spot variance | 2.591 | 1.489 | 3.695 | 7.977 | 0.884 | 1.769 |
| Heston | Heston delta | 2.697 | 1.852 | 4.107 | 8.968 | 0.990 | 1.990 |
| Heston | Heston minimum-variance delta | 2.498 | 1.305 | 3.466 | 7.532 | 0.792 | 1.581 |
| Heston | No-trade band chosen for 50% CVaR | 1.917 | 1.566 | 2.994 | 9.467 | 0.213 | 0.421 |
| Heston | No-trade band chosen for 99% CVaR | 2.191 | 1.248 | 3.136 | 6.993 | 0.485 | 0.963 |
| Heston | No hedge | 1.705 | 2.181 | 3.209 | 12.799 | 0.000 | 0.000 |
| Heston | Network, trained on 50% CVaR | 1.863 | 1.626 | 2.882 | 10.707 | 0.158 | 0.320 |
| Heston | Network, trained on 99% CVaR | 2.059 | 1.396 | 3.173 | 6.687 | 0.352 | 0.705 |
| Black-Scholes | Black-Scholes delta, 20% vol | 2.493 | 1.689 | 3.696 | 10.187 | 0.784 | 1.572 |
| Black-Scholes | Leland delta, 20% vol | 2.424 | 1.635 | 3.629 | 9.363 | 0.716 | 1.436 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 2.164 | 1.471 | 3.234 | 8.505 | 0.457 | 0.913 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 2.270 | 1.556 | 3.417 | 8.837 | 0.562 | 1.127 |
| Black-Scholes | Network, trained on 50% CVaR | 2.172 | 1.513 | 3.257 | 9.320 | 0.465 | 0.931 |
| Black-Scholes | Network, trained on 99% CVaR | 2.291 | 1.580 | 3.474 | 8.527 | 0.584 | 1.172 |


## Heston paths, cost 100 bp one way

| Built for | Strategy | Mean loss | Std | CVaR 50% | CVaR 99% | Costs paid | Shares traded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heston | Black-Scholes delta, implied vol | 3.426 | 2.198 | 4.917 | 13.674 | 1.718 | 1.721 |
| Heston | Leland delta, implied vol | 3.152 | 1.946 | 4.578 | 11.416 | 1.443 | 1.448 |
| Heston | Black-Scholes delta, spot variance | 3.475 | 1.687 | 4.764 | 9.398 | 1.768 | 1.769 |
| Heston | Heston delta | 3.687 | 2.176 | 5.396 | 10.663 | 1.980 | 1.990 |
| Heston | Heston minimum-variance delta | 3.290 | 1.504 | 4.402 | 8.998 | 1.583 | 1.581 |
| Heston | No-trade band chosen for 50% CVaR | 2.030 | 1.666 | 3.196 | 10.079 | 0.325 | 0.321 |
| Heston | No-trade band chosen for 99% CVaR | 2.579 | 1.333 | 3.592 | 7.644 | 0.872 | 0.866 |
| Heston | No hedge | 1.705 | 2.181 | 3.209 | 12.799 | 0.000 | 0.000 |
| Heston | Network, trained on 50% CVaR | 1.892 | 1.750 | 3.002 | 11.277 | 0.187 | 0.189 |
| Heston | Network, trained on 99% CVaR | 2.313 | 1.486 | 3.490 | 7.328 | 0.606 | 0.608 |
| Black-Scholes | Black-Scholes delta, 20% vol | 3.277 | 2.046 | 4.728 | 12.446 | 1.569 | 1.572 |
| Black-Scholes | Leland delta, 20% vol | 3.055 | 1.890 | 4.467 | 10.645 | 1.347 | 1.352 |
| Black-Scholes | No-trade band chosen for 50% CVaR | 2.486 | 1.533 | 3.607 | 8.939 | 0.780 | 0.777 |
| Black-Scholes | No-trade band chosen for 99% CVaR | 2.760 | 1.737 | 4.051 | 9.789 | 1.052 | 1.055 |
| Black-Scholes | Network, trained on 50% CVaR | 2.446 | 1.505 | 3.533 | 9.476 | 0.739 | 0.739 |
| Black-Scholes | Network, trained on 99% CVaR | 2.749 | 1.732 | 4.051 | 9.376 | 1.041 | 1.045 |
