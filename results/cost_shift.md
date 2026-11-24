# Built for one cost rate, charged another

Rows are the cost a strategy was built for and columns the cost it was charged, in basis points one way. Each cell is the CVaR at the level named in the heading. Networks are means over training seeds. The diagonal is the main comparison.


## Black-Scholes paths, 50% CVaR

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Network, built for 0 bp | 2.556 | 2.793 | 3.810 | 5.144 |
| Network, built for 10 bp | 2.580 | 2.759 | 3.514 | 4.519 |
| Network, built for 50 bp | 2.738 | 2.852 | 3.313 | 3.899 |
| Network, built for 100 bp | 2.985 | 3.067 | 3.396 | 3.809 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Band, built for 0 bp | 2.556 | 2.788 | 3.786 | 5.107 |
| Band, built for 10 bp | 2.589 | 2.758 | 3.476 | 4.438 |
| Band, built for 50 bp | 2.763 | 2.874 | 3.327 | 3.912 |
| Band, built for 100 bp | 2.913 | 3.005 | 3.379 | 3.856 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Leland, built for 0 bp | 2.556 | 2.788 | 3.786 | 5.107 |
| Leland, built for 10 bp | 2.564 | 2.782 | 3.721 | 4.982 |
| Leland, built for 50 bp | 2.632 | 2.818 | 3.602 | 4.668 |
| Leland, built for 100 bp | 2.718 | 2.889 | 3.590 | 4.511 |


## Black-Scholes paths, 99% CVaR

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Network, built for 0 bp | 3.269 | 3.511 | 4.586 | 6.014 |
| Network, built for 10 bp | 3.284 | 3.487 | 4.437 | 5.737 |
| Network, built for 50 bp | 3.492 | 3.624 | 4.246 | 5.216 |
| Network, built for 100 bp | 3.762 | 3.875 | 4.351 | 5.044 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Band, built for 0 bp | 3.503 | 3.780 | 4.940 | 6.451 |
| Band, built for 10 bp | 3.443 | 3.676 | 4.674 | 5.990 |
| Band, built for 50 bp | 3.450 | 3.593 | 4.305 | 5.376 |
| Band, built for 100 bp | 3.700 | 3.808 | 4.282 | 5.039 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Leland, built for 0 bp | 3.520 | 3.852 | 5.237 | 7.029 |
| Leland, built for 10 bp | 3.429 | 3.742 | 5.056 | 6.762 |
| Leland, built for 50 bp | 3.278 | 3.506 | 4.586 | 6.044 |
| Leland, built for 100 bp | 3.435 | 3.581 | 4.329 | 5.557 |


## Heston paths, 50% CVaR

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Network, built for 0 bp | 2.537 | 2.804 | 3.905 | 5.331 |
| Network, built for 10 bp | 2.575 | 2.668 | 3.048 | 3.534 |
| Network, built for 50 bp | 2.727 | 2.758 | 2.882 | 3.040 |
| Network, built for 100 bp | 2.823 | 2.841 | 2.912 | 3.002 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Band, built for 0 bp | 2.567 | 2.744 | 3.466 | 4.402 |
| Band, built for 10 bp | 2.592 | 2.710 | 3.189 | 3.802 |
| Band, built for 50 bp | 2.766 | 2.812 | 2.994 | 3.223 |
| Band, built for 100 bp | 2.818 | 2.855 | 3.006 | 3.196 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Leland, built for 0 bp | 2.647 | 2.871 | 3.775 | 4.917 |
| Leland, built for 10 bp | 2.655 | 2.872 | 3.749 | 4.857 |
| Leland, built for 50 bp | 2.679 | 2.879 | 3.683 | 4.696 |
| Leland, built for 100 bp | 2.699 | 2.885 | 3.635 | 4.578 |


## Heston paths, 99% CVaR

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Network, built for 0 bp | 5.853 | 6.066 | 6.960 | 8.164 |
| Network, built for 10 bp | 5.869 | 6.050 | 6.802 | 7.796 |
| Network, built for 50 bp | 5.998 | 6.133 | 6.687 | 7.412 |
| Network, built for 100 bp | 6.180 | 6.291 | 6.743 | 7.328 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Band, built for 0 bp | 6.134 | 6.352 | 7.258 | 8.451 |
| Band, built for 10 bp | 6.144 | 6.342 | 7.162 | 8.246 |
| Band, built for 50 bp | 6.305 | 6.439 | 6.993 | 7.727 |
| Band, built for 100 bp | 6.458 | 6.571 | 7.035 | 7.644 |

| Strategy | Charged 0 bp | Charged 10 bp | Charged 50 bp | Charged 100 bp |
| --- | ---: | ---: | ---: | ---: |
| Leland, built for 0 bp | 8.555 | 9.054 | 11.082 | 13.674 |
| Leland, built for 10 bp | 8.378 | 8.858 | 10.807 | 13.293 |
| Leland, built for 50 bp | 7.878 | 8.304 | 10.031 | 12.233 |
| Leland, built for 100 bp | 7.492 | 7.876 | 9.435 | 11.416 |
