# Sources, formulas and comparison

Every formula the code uses is listed here with where it was read. Two of the three named sources could not be reached and are marked.

## 1. Sources and what was actually read

| Source | Read? | Where |
| --- | --- | --- |
| Buehler, Gonon, Teichmann and Wood, "Deep hedging", Quantitative Finance 19(8), 2019 | Yes, the arXiv version 1802.03042 | arxiv.org/abs/1802.03042 |
| Leland, "Option Pricing and Replication with Transactions Costs", Journal of Finance 40(5), 1985 | **No.** Wiley returned 403 and no open copy was found | Formula taken from Kabanov and Safarian (below) |
| Boyle and Emanuel, "Discretely adjusted option hedges", Journal of Financial Economics 8(3), 1980 | **No.** ScienceDirect returned 403 and no open copy was found | Result derived in section 2 and checked by simulation |
| Kabanov and Safarian, "On Leland's strategy of option pricing with transactions costs", Finance and Stochastics 1, 1997 | Yes | link.springer.com/content/pdf/10.1007/s007800050023.pdf |
| Kamal and Derman, "When You Cannot Hedge Continuously: The Corrections to Black-Scholes", Goldman Sachs Quantitative Strategies Research Notes, 1998 (Risk, January 1999) | Yes | emanuelderman.com |

Kabanov and Safarian restate Leland's strategy in order to prove a theorem about it, so their statement of the formula is reliable. For Boyle and Emanuel only the abstract-level description was available: the hedging error over a short interval is proportional to gamma and follows a chi-square law. The exact form used here is derived below from the Black-Scholes equation. Equation numbers from the 1980 paper cannot be given.

## 2. Formulas

Notation: spot S, strike K, volatility σ, time to expiry τ, N rebalances over the life T, interval Δt = T/N. Rates are zero except where stated.

### 2.1 Black-Scholes

Kabanov and Safarian Eq. (1), with zero rates: C = S Φ(d) − K Φ(d − σ√τ), d = ln(S/K)/(σ√τ) + σ√τ/2. Delta is Φ(d), gamma is φ(d)/(Sσ√τ), vega is Sφ(d)√τ. The code also carries a rate r so that the Derman and Kamal example (r = 5%) can be rerun. Code: `black_scholes.py`.

### 2.2 The hedging error over one interval (Boyle and Emanuel; derived here)

Hold one call and short Φ(d) shares over an interval Δt. To second order the gain is

    Θ Δt + ½ Γ (ΔS)².

With zero rates the Black-Scholes equation gives Θ = −½ σ² S² Γ, and ΔS ≈ σ S √Δt u with u standard normal, so

    gain ≈ λ (u² − 1),   λ = ½ Γ S² σ² Δt.          (BE)

This is a chi-square variable with one degree of freedom, shifted to mean zero. Its standard deviation is √2 λ, its skewness is √8, it is negative with probability P(|u| < 1) = 68.3%, and it cannot be worse than −λ. The seller of the option has the mirror image: many small gains and a few large losses. Code: `hedging.boyle_emanuel_scale`. Checked in `tests/test_hedging.py::test_one_interval_error_is_a_shifted_chi_square`.

### 2.3 The hedging error over the whole life

Errors in different intervals are uncorrelated to leading order, so the variances add:

    Var ≈ Σ_k ½ (σ² Δt)² E[(Γ_k S_k²)²].               (SUM)

There are N terms each of order Δt² = T²/N², so the standard deviation falls as 1/√N. The expectation is taken by Gauss-Hermite quadrature over the lognormal law of S_k. Code: `hedging.discrete_hedge_std`.

Derman and Kamal Eq. (1) is a closed-form approximation to the same quantity:

    σ_P&L ≈ √(π/4) κ σ / √N,   κ = vega at inception.    (DK)

The equations in that note are images and did not survive text extraction. (DK) was confirmed against the numbers the note prints from it: 0.443 for N = 21 and 0.222 for N = 84 with κ = 11.5. The code gives 0.4432 and 0.2216. Code: `hedging.derman_kamal_std`.

### 2.4 Leland's adjusted volatility

Kabanov and Safarian Eq. (2), with T = 1 and n revisions:

    σ̂² = σ² (1 + γ/σ),   γ = 2 √(2/π) k √n.

Here k is the one-way cost as a fraction of traded value, which they note is half of Leland's round-trip coefficient. For a general interval, √n becomes 1/√Δt:

    σ̂² = σ² (1 + 2 √(2/π) k / (σ √Δt)).               (L)

The hedger of a short option uses the Black-Scholes delta at σ̂ and charges the Black-Scholes price at σ̂. The ratio 2√(2/π) k/(σ√Δt) is called the Leland number below. Code: `hedging.leland_vol`.

Their value process, Eq. (3), charges k S |Δ holding| at each revision and leaves out the trade that opens the position and the trade at expiry (their Remark 2).

### 2.5 The deep hedge

All from the arXiv version of Buehler et al.

- Terminal wealth, Eq. (2.1): PL_T(Z, p0, δ) = −Z + p0 + (δ·S)_T − C_T(δ), with (δ·S)_T = Σ_k δ_k (S_{k+1} − S_k) and C_T(δ) = Σ_k c_k(δ_k − δ_{k−1}), δ_{−1} = 0. Code: `hedging.hedge_pnl`.
- Proportional cost, Eq. (5.7): c_k(n) = ε |n| S_k. Section 2 says the numerical examples charge no cost at maturity; the code does the same.
- Optimised certainty equivalent, Eq. (3.5): ρ(X) = inf_w { w + E[ℓ(−X − w)] }.
- Example 3.9: ℓ(x) = max(x, 0)/(1 − α) gives the average value at risk, also written in Eq. (5.1) as the average of VaR_γ over γ in (0, 1 − α). Code: `risk.cvar`, `deep.cvar_loss`.
- Training objective, Eq. (4.6): J(θ) = w + E[ℓ(Z − (δ^θ·S)_T + C_T(δ^θ) − w)].
- Strategy, Eq. (4.1): δ_k = F_k(I_0, …, I_k, δ_{k−1}). Section 5.1 uses δ_k = F_k(I_k, δ_{k−1}).
- Heston model, Eq. (5.2): dS = √V S dB, dV = α(b − V) dt + σ√V dW, correlation ρ, with α = 1, b = 0.04, ρ = −0.7, σ = 2, v0 = 0.04, s0 = 100. Simulated as in Section 5.2 and footnote 5: exact draws from the CIR transition law, and the price step with the variance held at its start-of-step value. Code: `paths.heston_paths`.
- Expected integrated variance, Eq. (5.4): L(t, v) = (v − b)/α (1 − e^{−α(T−t)}) + b(T − t). Used only as a test of the simulator.
- Setting, Section 5.1: T = 30/365, n = 30 daily rebalances, at-the-money call.

The volatility of variance σ = 2 looks like a misprint for 0.2, since it puts the Feller ratio 2αb/σ² at 0.02. It is not: the simulator with σ = 2 gives an option price of 1.702 (standard error 0.002) on a million paths, and the paper prints q = 1.69. With σ = 0.2 the same simulator gives 2.279, next to the Black-Scholes value of 2.287. With σ = 2 the variance is at or near zero at expiry on 69% of paths (below 0.0001), and very large on a few. This is the world the paper trained in, and it is an extreme one.

## 3. Claim by claim

All numbers below come from `results/`, produced by `scripts/reproduce.py` on an Intel Core i9-13900K, CPU only, one thread. The contract is a short at-the-money call, S = K = 100, T = 30/365, zero rates, σ = 0.2 in the Black-Scholes world.

### 3.1 Discrete hedging without costs

| Claim | Source | Here | Match |
| --- | --- | --- | --- |
| Mean hedging error is zero | Derman and Kamal, Table 1 | Between −0.003 and +0.003 for N from 5 to 500, 100,000 paths each, all inside 2.3 standard errors | Yes |
| Std falls as 1/√N | Derman and Kamal, Eq. (1); (SUM) | Fitted log-log slope −0.485 over N = 5 to 500. Std × √N rises from 1.87 at N = 5 to 2.01 at N = 500 | Roughly. The error falls a little more slowly than 1/√N over this range |
| Std ≈ √(π/4) κ σ/√N | Derman and Kamal, Eq. (1) | Rule is 8% above the simulation at N = 5, 3% at N = 30, 1% at N = 500 | Yes, as an approximation |
| Std from summed interval variances | (SUM), derived here | 5% below the simulation at N = 5, 2.4% below at N = 30, 2.6% below at N = 500 | Close. The gap stops closing after N = 30 |
| One-interval error is λ(u² − 1) | (BE) | Std/λ 1.423 against 1.414, skewness 2.77 against 2.83, loss in 68.1% of intervals against 68.3%, worst outcome −1.003λ against −λ | Yes |
| Simulated std 0.41 at N = 21 and 0.20 at N = 84 (one month, r = 5%) | Derman and Kamal, Table 1 | 0.428 and 0.218 on 200,000 paths (`tests/test_hedging.py`) | **No.** 4% and 9% above the printed values |

On the (SUM) row: the likely reason is the last few intervals. Gamma at the money grows as 1/√τ, so the terms of (SUM) grow as 1/τ and the final intervals carry a share of the total that shrinks only as 1/log N. In those intervals the second-order expansion behind (BE) is poor. This would also explain why std × √N keeps drifting up. I did not prove it.

On the last row: the note used 50,000 paths, so the sampling error in its standard deviations is about 0.3%, too small to explain the gap. The numbers here agree with the note's own rule of thumb (0.443 and 0.222) better than the note's simulation does. Counting 23 intervals in place of 21 would give 0.409, but no shift of that kind also gives 0.20 at N = 84 (88 intervals gives 0.213). The cause is not established.

### 3.2 Leland under costs

At 50 bp one way, each hedger charging the price at its own volatility, entry trade excluded as in Kabanov and Safarian:

| Claim | Source | Here | Match |
| --- | --- | --- | --- |
| Expected cost of one rebalance equals the extra time decay at σ̂ | Leland's argument, which fixes the constant in Kabanov and Safarian Eq. (2) | Simulated mean cost within 3% of ½ΓS²(σ̂² − σ²)Δt (`test_expected_cost_of_one_rebalance_is_the_extra_theta`) | Yes |
| The Leland hedge is self-financing on average | Leland (1985), via Kabanov and Safarian | Mean error +0.058 at N = 5, +0.010 at N = 30, −0.009 at N = 500, against a premium between 2.6 and 4.6 | Yes, to within 0.06 |
| The Black-Scholes hedge loses its costs, and more as N grows | Standard | Mean error −0.30 at N = 5, −0.86 at N = 30, −3.56 at N = 500 | Yes |
| With constant k the error does not vanish as N grows | Kabanov and Safarian, Theorem 2 | Std 0.188 at N = 250 and 0.150 at N = 500, still falling | **Not tested.** The theorem is about the limit. At N = 500 the Leland number is 3.1 and this run cannot tell the two claims apart |

### 3.3 Deep hedging

| Claim | Source | Here | Match |
| --- | --- | --- | --- |
| Heston call price q = 1.69 | Section 5.2 | 1.702 ± 0.002 on a million paths; 1.706 as the mean loss of the costless hedges on the test set | Close. 0.012 above, about five standard errors |
| With no costs and 50% CVaR the network is close to the model hedge | Section 5.2, Figure 2 | Black-Scholes paths: network CVaR 2.556, Black-Scholes delta 2.555; std 0.361 against 0.358 | Yes, in the Black-Scholes world. The paper's test was in Heston with a variance swap, which is not built here |
| A 99% CVaR hedge has a thinner tail and a wider body than a 50% one | Section 5.2, Figures 6 and 7 | In all eight cells the 99% network has the lower 99% CVaR and the higher 50% CVaR | Yes |
| The recurrent input is needed under costs | Section 5.2, footnote 7, Figure 5 | Not run without the recurrent input | Not tested |
| Price rises as ε^(2/3) for small costs | Section 5.3, Eq. (5.8) | Not run. The paper's test uses the entropic risk measure, which is not built here | Not tested |

### 3.4 The question in the README

Does the network beat Leland? Each network is scored on the CVaR level it was trained on, on 200,000 test paths the training never saw. Differences are network minus Leland, with the standard error of the paired difference over 20 blocks of paths.

| Paths | Cost | 50% CVaR | 99% CVaR |
| --- | ---: | ---: | ---: |
| Black-Scholes | 0 bp | +0.001 ± 0.000 | −0.249 ± 0.009 |
| Black-Scholes | 10 bp | −0.019 ± 0.000 | −0.253 ± 0.009 |
| Black-Scholes | 50 bp | −0.285 ± 0.001 | −0.322 ± 0.009 |
| Black-Scholes | 100 bp | −0.689 ± 0.003 | −0.487 ± 0.010 |
| Heston | 0 bp | −0.131 ± 0.004 | −2.026 ± 0.041 |
| Heston | 10 bp | −0.206 ± 0.004 | −2.115 ± 0.040 |
| Heston | 50 bp | −0.742 ± 0.006 | −2.552 ± 0.044 |
| Heston | 100 bp | −1.456 ± 0.008 | −3.198 ± 0.047 |

The network is lower in 15 of 16 cells. The exception is the costless Black-Scholes case at 50%, where it lands 0.001 above the formula it is supposed to recover.

What that table leaves out, from `results/comparison.md`:

- Leland has the smaller standard deviation on Black-Scholes paths at every positive cost. At 50 bp it is 0.418, against 0.628 for the 50% network and 0.544 for the 99% network.
- A network is only good on the measure it was trained on. At 100 bp on Black-Scholes paths the 50% network has a 99% CVaR of 7.131, worse than Leland (5.545) and worse than the plain Black-Scholes delta (7.017). On Heston paths the 50% network has a worse 99% CVaR than Leland at every cost level.
- Most of the 50% network's advantage is cost it does not pay. At 50 bp on Black-Scholes paths it trades 1.11 shares over the life against Leland's 1.99 and pays 0.556 in costs against 0.997. The gap in mean loss (0.440) is larger than the gap in 50% CVaR (0.285).
- Three training seeds at 50 bp give a 50% CVaR between 3.316 and 3.338 on Black-Scholes paths and between 2.881 and 2.885 on Heston paths. The spread across seeds is under a tenth of the gap to Leland.

## 4. Differences from the sources

Deep hedging, against Sections 5.1 and 5.2 of the paper:

1. Hedging instruments. The paper hedges with the stock and a variance swap in the Heston model. Here the hedger trades the stock only, in both worlds. The Heston results are therefore not comparable with the paper's figures, and the benchmark is the Black-Scholes delta, since the Heston model hedge needs the swap.
2. Network. One network is shared across all 30 dates, with the square root of the share of time left as an input. The paper trains a separate network for each date. There are three hidden layers of 32 units here and two hidden layers of d + 15 units there. No batch normalisation here.
3. Inputs. Log-moneyness scaled by σ√T, time, the previous holding, and in the Heston world the current variance divided by 0.04. The paper uses log S and V.
4. Inner minimisation. Eq. (4.6) optimises w by gradient steps together with the weights. Here w is set to the batch quantile at every step, which is the exact minimiser for the batch.
5. Optimiser settings. Adam with the learning rate decaying from 0.002 to 0.0001, batch 1,024 for 50% CVaR and 4,096 for 99%, 3,000 and 2,000 steps. The paper uses a constant 0.005 and batch 256.
6. Sample sizes. 200,000 training and 200,000 test paths per world. The paper tests on a million.
7. Entry cost. The first trade from zero shares pays costs, as Eq. (2.1) says. This applies to every strategy equally.
8. Precision. Network arithmetic is 32-bit. Paths and all reported losses are 64-bit: the network supplies holdings and the same numpy accounting scores every strategy.

Leland:

9. The comparison table charges every strategy the entry cost and reports loss before premium, so Leland is judged on CVaR, a measure it was not designed for. Section 3.2 uses Kabanov and Safarian's convention.
10. On Heston paths the Leland and Black-Scholes deltas use σ = 0.2, the square root of the starting and long-run variance. A third formula hedge, the Black-Scholes delta with the current variance plugged in, is reported in `results/comparison.md` but is not part of the question.

## 5. Variants tried

Everything that was run before the committed full run, on exploratory paths with other seeds:

1. Network arithmetic in 64-bit, then 32-bit. At 50 bp on Black-Scholes paths with 2,000 steps the 50% CVaR was 3.3348 and 3.3340. Kept 32-bit because it halves the run time.
2. Training length at the same setting: 2,000 steps gave 3.334 and 6,000 gave 3.314. Chose 3,000. Longer training would lower the network's numbers by something like half a percent.
3. Batch 4,096 with 1,000 steps at zero cost for the first trial, then batch 1,024 for the 50% objective. Chosen for speed.
4. Thread counts of 1, 4 and 8 were timed. One thread was kept so the run is reproducible and light on a shared machine.
5. A 99% objective was added after the first trial with costs showed the 50% network losing to Leland in the far tail. That is a change to the design made after seeing a result. Both objectives are reported in full.

Not varied: the architecture (width, depth, activation), the cost grid, the Heston parameters and the contract. One full run was made, then repeated from a clean clone.

## 6. Not verified

- Leland (1985) and Boyle and Emanuel (1980) were not read. Their equation numbers are not given anywhere in this repository.
- The Derman and Kamal equations are images. Eq. (1) was confirmed from the numbers printed beside it. Eq. (2) to (7) were not checked.
- Whether a GPU run gives the same tables. The code accepts `--device cuda`, but that path was not run.
- CI has not run on GitHub.
