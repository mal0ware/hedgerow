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
