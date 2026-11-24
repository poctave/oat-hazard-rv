# Method

## 1. Spreads and the coupon problem

| Measure | Reference | Limit |
|---|---|---|
| G-spread | interpolated Bund yield | one point, mixes German technicals |
| I-spread | swap rate at the same maturity | one point |
| Asset-swap | floating leg after swapping the bond | depends on price vs par |
| Z-spread | constant shift of the €STR OIS zero curve | assumes every cash flow is paid |
| CDS | none (pure credit) | USD quotes, liquid only at some tenors |

Z-spread puts all credit into the discount rate and gives nothing back in default. A bond at 50 with 40% recovery
loses about 10 in default; a bond at par loses about 60. Low coupons therefore hold up better under stress and look
rich on Z-spread: a Z-spread screen recommends selling them for no credit reason.

## 2. Reduced-form pricing

Default arrives with intensity λ(t); survival `Q(t) = exp(−∫₀ᵗ λ)`. With default independent of rates,

```
P = Σ_i CF_i D(t_i) Q(t_i)                       coupons and principal, paid if alive
  + 100 R Σ_k D(m_k) [Q(m_{k−1}) − Q(m_k)]       recovery of face R, monthly grid
```

`D` is the €STR OIS discount factor (EONIA OIS − 8.5bp before Oct 2019 when needed). With constant λ the promised
flows are discounted at r + λ and the par spread is ≈ λ(1 − R) (credit triangle).

## 3. Calibration

- λ is piecewise constant on maturity buckets (main spec: 0-3, 3-7, 7-15, 15-30, 30Y+).
- Objective: Σ [(P_model − P_mkt) / (P_mkt · ModDur)]², i.e. squared yield errors, so long bonds do not dominate.
- `least_squares` with λ ∈ [−5%, 100%]; negative λ is allowed on purpose and flagged (`breakdown`).
- Optional: recovery fitted jointly (`--recovery fit`, R ∈ [0, 60%]).
- Residual = market yield − model yield (bp, > 0 cheap). Exact yields recomputed after the fit.
- Far fewer parameters than bonds (5 vs ~45-55), so residuals carry information.

## 4. Universe and conventions

- FRTR fixed-coupon nominal annual OATs, 1 to 50 years to maturity, at least 30 days after issue.
- Settlement T+2 business days; accrued ACT/ACT ICMA; yields annual compounding on ICMA periods.
- Coupon dates = maturity anniversaries (irregular first coupons ignored).

## 5. Specifications (`scripts/run_all.py`)

| Tag | Recovery | Buckets | Risk-free / funding |
|---|---|---|---|
| R25_k5 (main) | 25% | 0,3,7,15,30 | €STR OIS |
| R25_k5_fundbtf | 25% | 0,3,7,15,30 | €STR OIS + (3M BTF − €STR) |
| R25_k5_funding | 25% | 0,3,7,15,30 | €STR OIS + (GC Pooling ON − €STR) |
| R40 | 40% | 0,3,7 | €STR OIS (benchmark) |
| R40_k5 | 40% | 0,3,7,15,30 | €STR OIS |
| R10_k5 | 10% | 0,3,7,15,30 | €STR OIS |
| Rfit_k5 | fitted | 0,3,7,15,30 | €STR OIS |
| R25_k5_bund | 25% | 0,3,7,15,30 | Bund curve (France-vs-Germany credit) |

## 6. Signal tests (`scripts/analyze.py`)

For the hazard residual and the Z-spread residual (Z-spread minus a cubic in maturity), by regime
(pre-QE, QE = Mar 2015 to Jun 2022, post-QE):

1. Coupon bias: mean cross-sectional correlation with the coupon.
2. Mean reversion: pooled AR(1) of each bond's residual, half-life in fit periods.
3. Rank IC: correlation between the residual and its next-period compression.
4. Long/short: top quintile (cheap) vs bottom quintile (rich), DV01-neutral, residual compression in bp.
