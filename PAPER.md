# Sources, formulas and comparison

Every formula the code uses is listed here with where it was read. Sources that could not be reached are marked.

## 1. Sources and what was actually read

| Source | Read? | Where |
| --- | --- | --- |
| Buehler, Gonon, Teichmann and Wood, "Deep hedging", Quantitative Finance 19(8), 2019 | Yes, the arXiv version 1802.03042 | arxiv.org/abs/1802.03042 |
| Leland, "Option Pricing and Replication with Transactions Costs", Journal of Finance 40(5), 1985 | **No.** Wiley returned 403 and no open copy was found | Formula taken from Kabanov and Safarian (below) |
| Boyle and Emanuel, "Discretely adjusted option hedges", Journal of Financial Economics 8(3), 1980 | **No.** ScienceDirect returned 403 and no open copy was found | Result derived in section 2 and checked by simulation |
| Kabanov and Safarian, "On Leland's strategy of option pricing with transactions costs", Finance and Stochastics 1, 1997 | Yes | link.springer.com/content/pdf/10.1007/s007800050023.pdf |
| Kamal and Derman, "When You Cannot Hedge Continuously: The Corrections to Black-Scholes", Goldman Sachs Quantitative Strategies Research Notes, 1998 (Risk, January 1999) | Yes | emanuelderman.com |
| Heston, "A Closed-Form Solution for Options with Stochastic Volatility", Review of Financial Studies 6(2), 1993, and Albrecher, Mayer, Schoutens and Tistaert, "The Little Heston Trap", Wilmott Magazine, 2007 | **No.** Neither was opened for this work | Characteristic function written from memory of the standard form and checked numerically (section 2.6) |

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

### 2.6 Heston price and deltas

Used for the formula hedges on Heston paths. Code: `heston.py`.

Let X = ln(S_T/S_t). Under the martingale measure E[exp(iuX)] = exp(A(u) + B(u)V), with

    b = κ − ρξiu,   d = √(b² + ξ²(iu + u²)),   g = (b − d)/(b + d),
    B = (b − d)/ξ² · (1 − e^{−dτ})/(1 − g e^{−dτ}),
    A = κθ/ξ² · ((b − d)τ − 2 ln((1 − g e^{−dτ})/(1 − g))).

This is the form usually credited to Albrecher et al., which keeps the logarithm on its principal branch. I did not read that paper or Heston's. The formula was accepted on four checks: A and B vanish at u = 0 and at u = −i; the call price is 1.6919 against the 1.69 printed in the deep hedging paper; with ξ near zero the price and delta match Black-Scholes to 0.0002; and the two deltas below match finite differences of the price. All four are in `tests/test_heston.py`.

Under the measure that uses the stock as numeraire X has characteristic function exp(A(u − i) + B(u − i)V). Its density f is recovered by one FFT. With k = ln(K/S),

    ∂C/∂S = P(X > k),    C/S = ∫ over x > k of (1 − e^{k−x}) f(x) dx,

and ∂C/∂V comes from the same integrals applied to the inverse transform of B(u − i) times the characteristic function. A hedger who trades only the stock and wants the smallest variance of the next change in the hedged position holds

    δ_mv = ∂C/∂S + ρξ (∂C/∂V)/S,

because d⟨S, V⟩/d⟨S⟩ = ρξ/S. With ρ = −0.7 this is well below ∂C/∂S.

Two numerical choices. The characteristic function is multiplied by exp(−ε²u²/2) with ε = 0.0002, which blurs X by 0.02% of spot, because with V near zero the density is close to a point mass. The deltas are tabulated on 1,201 moneyness points and 96 variance points per date and read by bilinear interpolation.

The implied volatility of the Heston call is 14.79%. The Leland and Black-Scholes hedges built for the Heston world use it.

### 2.7 The no-trade band

Not from a source. The hedger has a target delta δ*_k and a half-width w, and holds δ_k = min(max(δ_{k−1}, δ*_k − w), δ*_k + w), starting from zero shares. The target is one of the formula deltas for that world and w is one of 15 values from 0 to 1 share. For each world, cost and CVaR level the pair with the lowest CVaR on validation paths is kept. Width zero is the plain formula hedge, so on validation paths the band is never worse than the best formula. The literature on hedging with proportional costs has bands whose width depends on gamma (Whalley and Wilmott 1997 is the usual reference, not read). The constant width here is cruder. Code: `experiments.band_deltas`, `experiments.choose_bands`.

### 2.8 When a difference counts

For each cell and each of five training seeds, d_s is the network's CVaR minus the other strategy's on the same test paths. Let s be the standard deviation of d_s across seeds and e the mean standard error of d_s from the finite test set (paired, over 20 blocks of 10,000 paths). The noise in one trained network is taken as √(s² + e²). The cell goes to the network or to the formula when the mean of d_s is more than twice that, and is a tie otherwise. This is stricter than testing the mean of five seeds. It asks whether a sixth network would be expected to land on the same side. Code: `experiments.judge`.

The volatility of variance σ = 2 looks like a misprint for 0.2, since it puts the Feller ratio 2αb/σ² at 0.02. It is not: the simulator with σ = 2 gives an option price of 1.702 (standard error 0.002) on a million paths, and the paper prints q = 1.69. With σ = 0.2 the same simulator gives 2.279, next to the Black-Scholes value of 2.287. With σ = 2 the variance is at or near zero at expiry on 69% of paths (below 0.0001), and very large on a few. This is the world the paper trained in, and it is an extreme one.

## 3. Claim by claim

Numbers below come from `results/`, produced by `scripts/reproduce.py` on an Intel Core i9-13900K, CPU only, one thread per network. The few that come from a test instead say so. The contract is a short at-the-money call, S = K = 100, T = 30/365, zero rates, σ = 0.2 in the Black-Scholes world.

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
| Heston call price q = 1.69 | Section 5.2 | 1.6919 by Fourier inversion. The simulator gives a mean payoff of 1.705 on the 200,000 test paths | Yes for the continuous model. The simulator, which holds the variance fixed within each day as the paper's footnote 5 describes, is about 0.01 higher |
| With no costs and 50% CVaR the network is close to the model hedge | Section 5.2, Figure 2 | Black-Scholes paths: network CVaR 2.556, Black-Scholes delta 2.556; std 0.361 against 0.358 | Yes, in the Black-Scholes world. The paper's test was in Heston with a variance swap, which is not built here |
| A 99% CVaR hedge has a thinner tail and a wider body than a 50% one | Section 5.2, Figures 6 and 7 | In all eight cells the 99% network has the lower 99% CVaR and the higher 50% CVaR (means of five seeds) | Yes |
| The recurrent input is needed under costs | Section 5.2, footnote 7, Figure 5 | Not run without the recurrent input | Not tested |
| Price rises as ε^(2/3) for small costs | Section 5.3, Eq. (5.8) | Not run. The paper's test uses the entropic risk measure, which is not built here | Not tested |

### 3.4 Does the network beat the formulas?

Each network is scored on the CVaR level it was trained on, on 200,000 test paths. Entries are the mean over five seeds of network minus the other strategy, with the noise of section 2.8. A star marks a difference that clears twice the noise.

| Paths | Cost | Level | Minus Leland at 20% vol | Minus Leland | Minus band |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | 0 bp | 50% | −0.000 ± 0.000 | same hedge | −0.000 ± 0.000 |
| Black-Scholes | 10 bp | 50% | −0.023 ± 0.001 * | same hedge | +0.001 ± 0.001 |
| Black-Scholes | 50 bp | 50% | −0.289 ± 0.015 * | same hedge | −0.014 ± 0.015 |
| Black-Scholes | 100 bp | 50% | −0.702 ± 0.012 * | same hedge | −0.047 ± 0.012 * |
| Black-Scholes | 0 bp | 99% | −0.251 ± 0.007 * | same hedge | −0.234 ± 0.008 * |
| Black-Scholes | 10 bp | 99% | −0.255 ± 0.007 * | same hedge | −0.188 ± 0.007 * |
| Black-Scholes | 50 bp | 99% | −0.340 ± 0.010 * | same hedge | −0.059 ± 0.010 * |
| Black-Scholes | 100 bp | 99% | −0.513 ± 0.011 * | same hedge | +0.004 ± 0.010 |
| Heston | 0 bp | 50% | −0.137 ± 0.003 * | −0.110 ± 0.003 * | −0.030 ± 0.003 * |
| Heston | 10 bp | 50% | −0.211 ± 0.004 * | −0.204 ± 0.004 * | −0.042 ± 0.002 * |
| Heston | 50 bp | 50% | −0.747 ± 0.006 * | −0.801 ± 0.006 * | −0.112 ± 0.004 * |
| Heston | 100 bp | 50% | −1.465 ± 0.007 * | −1.576 ± 0.008 * | −0.194 ± 0.003 * |
| Heston | 0 bp | 99% | −2.125 ± 0.040 * | −2.702 ± 0.049 * | −0.280 ± 0.016 * |
| Heston | 10 bp | 99% | −2.215 ± 0.040 * | −2.808 ± 0.048 * | −0.292 ± 0.016 * |
| Heston | 50 bp | 99% | −2.676 ± 0.042 * | −3.344 ± 0.048 * | −0.306 ± 0.020 * |
| Heston | 100 bp | 99% | −3.317 ± 0.048 * | −4.088 ± 0.052 * | −0.316 ± 0.023 * |

"Leland" is at the volatility that prices the call: 20% on Black-Scholes paths, 14.79% on Heston paths. The numbers are in `results/headline.csv`.

Counts over the 16 cells. Against Leland at 20%: network lower in 15, tie in 1. Against Leland at the right volatility: the same. Against the band: network lower in 12, tie in 4. No cell goes to a formula.

What the counts hide:

- The baseline matters more than the network. On Black-Scholes paths at 50% the gap to Leland at 50 bp is 0.289 and the gap to the band is 0.014. The band and the network do the same thing there: trade about half as much as Leland (1.08 and 1.11 shares against 1.99) and accept a wider spread of outcomes (std 0.63 and 0.65 against 0.42).
- Correcting the Heston volatility did not help Leland. At 14.79% its 99% CVaR is higher than at 20% in every cost row (10.03 against 9.36 at 50 bp). The first version's Heston wins were against a mis-specified baseline, but the better-specified Leland is no harder to beat.
- The strongest Heston formula is the minimum-variance delta. With no costs it has a 99% CVaR of 6.14, against 8.56 for the Black-Scholes delta at implied volatility and 7.42 for the Heston delta ∂C/∂S. The 99% network is at 5.85.
- The network's clearest edge is the tail. At 99% it beats the band by 0.19 to 0.23 on Black-Scholes paths at 0 and 10 bp, and by about 0.3 on Heston paths at every cost.
- On Heston paths at 50 bp and 100 bp, at the 50% level, the unhedged position (3.21) beats every formula delta that trades daily (3.47 to 5.40). The chosen bands are 0.5 and 0.6 shares wide, close to not trading. The network trades 0.32 and 0.19 shares over the life.
- A network is only good on the measure it was trained on. At 100 bp on Black-Scholes paths the 50% network has a 99% CVaR of 6.88, against 5.56 for Leland and 7.03 for the plain delta.
- Seed spread is uneven. It is 0.015 at 50 bp and 50% on Black-Scholes paths (five seeds between 3.294 and 3.333) and 0.002 in the matching Heston cell. The first version quoted ± 0.001 for the first of these, which was the test-set error only.

### 3.5 Run in the wrong model

Nothing is refitted in this section. `results/cross_world.md`, `results/vol_shift.md` and `results/cost_shift.md` have every cell.

Built for Black-Scholes, run on Heston paths. The formulas use 20% volatility. The network has no variance input. Against the band built for Black-Scholes the network is lower in 5 of 8 cells (all four at 99%, by 0.31 to 0.42, and 100 bp at 50%), ties one, and is higher in two by 0.010 and 0.023. Against Leland it is lower in 7 and ties one. Both are far behind what was built for Heston: at 50 bp and 99% the Black-Scholes network has 8.53, the Black-Scholes band 8.84, the Heston network 6.69.

Built for Heston, run on Black-Scholes paths. The formulas and the network are told V = 0.04 at every date. The 99% network has a CVaR of 6.12 to 8.47 across the four costs. The band built for Heston has 4.58 to 6.66, the Leland delta at the Heston implied volatility 3.99 to 6.12, and the network trained on Black-Scholes paths 3.27 to 5.04. Against the band the Heston network is lower in one cell, ties one and is higher in six. In training its variance input ranged from near zero to several times its mean. A constant 0.04 on every date of every path is far from anything it saw.

Built at 20% volatility, run at 15% and 25%. Against the band built at 20%: network lower in 7 of 16, tie in 2, higher in 7. The 50% networks stay within 0.05 of the band. The 99% networks are higher than the band by 0.10 to 0.14 at 15% and lower by 0.16 to 0.50 at 25%. Against a Leland hedge that knows the true volatility the network is lower in 10 of 16.

Built for one cost, charged another. On Black-Scholes paths the network and the band lose about the same amount. Built for 50 bp and charged 100 bp, the 50% CVaR is 3.90 for the network and 3.91 for the band, against 3.81 and 3.86 when built for 100 bp. The worst case is built for no cost and charged 100 bp: 5.14 for the network and 5.11 for the band on Black-Scholes paths, 5.33 and 4.40 on Heston paths. The costless Heston network trades 2.18 shares over the life, more than any formula.

## 4. Differences from the sources

Deep hedging, against Sections 5.1 and 5.2 of the paper:

1. Hedging instruments. The paper hedges with the stock and a variance swap in the Heston model. Here the hedger trades the stock only, in both worlds. The Heston results are therefore not comparable with the paper's figures. The paper's model hedge needs the swap. The stock-only formula hedges here are the Heston delta and the minimum-variance delta of section 2.6.
2. Network. One network is shared across all 30 dates, with the square root of the share of time left as an input. The paper trains a separate network for each date. There are three hidden layers of 32 units here and two hidden layers of d + 15 units there. No batch normalisation here.
3. Inputs. Log-moneyness scaled by σ√T, time, the previous holding, and in the Heston world the current variance divided by 0.04. The paper uses log S and V.
4. Inner minimisation. Eq. (4.6) optimises w by gradient steps together with the weights. Here w is set to the batch quantile at every step, which is the exact minimiser for the batch.
5. Optimiser settings. Adam with the learning rate decaying from 0.002 to 0.0001, batch 1,024 for 50% CVaR and 4,096 for 99%, 6,000 and 4,000 steps. The paper uses a constant 0.005 and batch 256.
6. Sample sizes. 200,000 training, 200,000 validation and 200,000 test paths per world. The paper tests on a million.
7. Entry cost. The first trade from zero shares pays costs, as Eq. (2.1) says. This applies to every strategy equally.
8. Precision. Network arithmetic is 32-bit. Paths and all reported losses are 64-bit: the network supplies holdings and the same numpy accounting scores every strategy.

Leland:

9. The comparison table charges every strategy the entry cost and reports loss before premium, so Leland is judged on CVaR, a measure it was not designed for. Section 3.2 uses Kabanov and Safarian's convention.
10. On Heston paths the Leland and Black-Scholes deltas use the implied volatility of the call at inception, 14.79%, and keep it for the life of the hedge. A desk would re-mark the implied volatility each day. The version with 20%, which the first version of this repository used, is kept in the tables as a hedge built for the wrong model.

## 5. What was chosen, and when

Chosen on validation paths, inside `scripts/reproduce.py`:

1. Training length. Three lengths per CVaR level were trained at 50 bp in both worlds with three seeds each and scored on validation paths (`results/training_length.md`). The longest won at both levels: 6,000 steps for 50% and 4,000 for 99%. Validation CVaR was still falling, by 0.4% from 3,000 to 6,000 steps on Black-Scholes paths, so the networks are probably a little undertrained.
2. The target and half-width of every no-trade band (`results/bands.md`).

Chosen before the first version, in trials that were not recorded and did not use a validation set. These are post hoc choices and nothing in the repository can show what they were tuned on:

3. Network arithmetic in 32-bit after a 64-bit trial gave nearly the same CVaR.
4. Batch sizes of 1,024 and 4,096, and the learning-rate schedule.
5. Three hidden layers of 32 units. No other size was tried.
6. The 99% objective. It was added after the first trial with costs showed the 50% network losing to Leland in the far tail. Both objectives are reported in full.

Chosen after seeing results of the current design:

7. The band grid first stopped at a half-width of 0.3 shares. In the first full run two Heston cells (50% CVaR at 50 and 100 bp) picked 0.3, the edge of the grid. The grid was extended to 1.0 and everything rerun. This lowered the band's test CVaR in those cells from 3.057 to 2.994 and from 3.372 to 3.196, and cut the network's lead there from 0.175 and 0.370 to 0.112 and 0.194. The earlier numbers are not in `results/`. They are recorded here only.
8. The no-trade band itself was added in this revision, after the first version's results were known. So were the rule of section 2.8 and the wrong-model runs of section 3.5.

The test paths of this version are a new draw. The first version's test paths are now the validation set.

Not varied: the cost grid, the Heston parameters, the contract, the number of seeds. One full run was made with the final design, then repeated, and the two sets of files were identical.

## 6. Not verified

- Leland (1985) and Boyle and Emanuel (1980) were not read. Their equation numbers are not given anywhere in this repository.
- The Derman and Kamal equations are images. Eq. (1) was confirmed from the numbers printed beside it. Eq. (2) to (7) were not checked.
- Heston (1993) and Albrecher et al. (2007) were not read. The characteristic function rests on the numerical checks in section 2.6.
- Whalley and Wilmott (1997) was not read. Their band is not implemented.
- Whether a GPU run gives the same tables. The code accepts `--device cuda`, but that path was not run.
- Whether the results survive other Heston parameters, other maturities or strikes, or a larger network.
