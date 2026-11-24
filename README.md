# oat-hazard-rv

Relative value on French government bonds (OATs) with a reduced-form hazard-rate model instead of Z-spread.
Master 203 thesis project, Université Paris Dauphine-PSL.

**Research question.** Does coupon-adjusted RV (residuals from one default curve fitted across all OATs) give
cleaner rich/cheap signals than Z-spread RV, and when does the method break?

- **Result sheet:** [`reports/results/results.pdf`](reports/results/results.pdf)
- **Data recap:** [`reports/data/data_recap.pdf`](reports/data/data_recap.pdf)
- **Method:** [`docs/method.md`](docs/method.md)

## Idea

Z-spread discounts every promised cash flow at €STR + z: it assumes 100% survival and no recovery. In default the
claim is on face value, so a low-coupon bond (price 70) loses far less than a par bond. Under credit stress, Z-spread
therefore makes low coupons look rich and high coupons cheap even with identical credit. A reduced-form model prices
that effect:

```
P = Σ CF_i · D(t_i) · Q(t_i)  +  100 · R · Σ D(m_k) · [Q(m_{k-1}) − Q(m_k)]
Q(t) = exp(−∫ λ),   D = €STR OIS discount factor,   spread ≈ λ (1 − R)
```

On each date one piecewise-constant λ curve is fitted across all OATs by minimising duration-scaled price errors
(≈ yield errors). Residuals (market yield − model yield, > 0 = cheap) are the trade signals; the curve gives a
bond-implied CDS curve. λ is allowed to go negative so that the QE breakdown (bonds richer than risk-free
discounting allows, i.e. survival above 100%) is visible rather than hidden by a bound.

## Results so far (real Bloomberg data, 117 OATs, weekly, 2010 to Nov 2026)

| | Finding |
|---|---|
| QE breakdown | The 0-3Y λ is negative on 77% of QE dates (Mar 2015 to Jun 2022): short OATs traded richer than €STR. |
| Funding | Discounting at €STR + (3M BTF − €STR), a collateral-scarcity proxy, cuts that to 40% and lifts the QE 0-3Y λ from −12bp to +3bp. GC Pooling (ECB basket) sits within a few bp of €STR and changes nothing. |
| Recovery | R = 40% cannot price the deep-discount ultra-longs (0.5% 2072 at 23, 0.75% 2053 at 37): RMSE 10.5bp post-QE. Main spec R = 25%, five buckets: median RMSE 3.0bp. Fitting R per date gives ~25% since 2024 but is weakly identified (5-42%). |
| Signals | Hazard residual vs Z-spread residual: lower coupon loading (−0.07 vs −0.13; QE −0.11 vs −0.25), slightly higher rank IC (0.112 vs 0.095), smaller residuals (2.7 vs 3.2bp); the naive long/short is not better yet (0.37 vs 0.43bp/week, no carry, roll or costs). |
| CDS basis | Bond-implied 5Y CDS is far below traded USD CDS before 2015 (40 vs 92bp), negative under QE, above it since 2025. |

## Layout

```
oathazard/              library
  curves.py             €STR / EONIA OIS bootstrap, log-linear discount curve, term shifts
  bonds.py              OAT schedule, accrued (ACT/ACT ICMA), yield, duration
  hazard.py             hazard pricing, Z-spread, global fit (fixed or fitted recovery), implied CDS
  data.py               raw Bloomberg pull -> model inputs (curves, funding proxies, cross-sections)
scripts/
  bloomberg/            run on a machine with a Bloomberg Terminal (blpapi)
    fetch_bloomberg.py  OAT universe, static, prices, €STR/EONIA OIS, France CDS, Bund/OAT generics
    fetch_extra.py      bid/ask, Bunds/BTPs/Bonos, ECB, swaps, CDS, iTraxx, vol, futures, GC/BTF discovery
    probe_bloomberg.py  ticker checks and searches
    fetch_repo.py       French GC repo candidates (not entitled on the Dauphine terminal)
  run_history.py        fit every date -> results/<tag>/fits.parquet, signals.parquet
  analyze.py            coupon bias, mean reversion, rank IC, long/short, by regime
  make_report.py        LaTeX result sheet
  make_data_sheet.py    LaTeX data recap
  make_synthetic.py     fake pull with a known truth (pipeline test without Bloomberg)
  run_all.py            reproduces everything in reports/
reports/
  results/              result sheet (pdf, tex, figures)
  data/                 data recap (pdf, tex, figures)
docs/method.md          model, calibration, conventions, specifications
tests/                  unit tests
config.yaml             tickers (all checked on a Terminal) and model settings
```

## Reproduce

```bash
pip install -r requirements.txt
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

# on the Bloomberg machine (Terminal open and logged in)
python scripts/bloomberg/fetch_bloomberg.py --check   # every ticker OK?
python scripts/bloomberg/fetch_bloomberg.py           # ~5 min  -> data/raw/
python scripts/bloomberg/fetch_extra.py               # ~10 min -> data/raw/extra/

# anywhere
python scripts/run_all.py                             # 8 specifications + both PDFs, ~7 min
```

Without a Terminal: `python -m pytest -q` and `python scripts/run_all.py --synthetic`.

## Data

Daily, 1 Jan 2010 to 24 Nov 2026, pulled through the Bloomberg Desktop API. Full inventory with coverage charts in
[`reports/data/data_recap.pdf`](reports/data/data_recap.pdf).

| Block | Content |
|---|---|
| OATs | 366 FRTR lines (live + matured); 117 fixed nominal priced: clean price, yield, bid/ask, Bloomberg Z-spread and ASW, amount outstanding history |
| Other govies | Bunds/Bobls/Schatz (218), BTPs (235, semi-annual), Bonos (127): price, bid/ask, yield |
| Rates | €STR and EONIA fixings, €STR OIS 1M-50Y, EONIA OIS (to 2021), ECB DFR/MRO, Euribor, EUR 6M swaps 2-30Y |
| Funding | STOXX GC Pooling ON to 12M, French BTF bills 1M-1Y, Bubill indices, €STR compounded |
| Credit | France CDS USD 1-30Y and EUR; Germany, Italy, Spain, Belgium, Netherlands CDS; iTraxx Main / Crossover |
| Other | Generic govt yields FR/DE/IT/ES, futures (OAT, Bund, Bobl, Buxl, BTP), V2X, VIX, MOVE, SX5E, CAC |

**The data is not in this repository**: Bloomberg's terms do not allow republishing it (`data/` is git-ignored).
Re-pull it with `scripts/bloomberg/`. Not available on the terminal used: French GC repo history, per-bond repo
rates, ECB holdings per ISIN.

## Simplifications and limits

- Fixed-coupon nominal annual OATs only. Coupon dates are maturity anniversaries; irregular first coupons, holidays
  and business-day rolls are ignored. BTPs (semi-annual) need the pricer extended before the cross-country step.
- ACT/365 discounting times, ACT/360 OIS; default independent of rates; recovery of face value.
- Everything that is not credit (liquidity, specialness, supply, CTD status) loads onto λ or the residuals.
- Traded CDS is USD-denominated: the basis includes redenomination / quanto effects.
- Long/short P&L is residual compression in bp of yield, DV01-neutral, without carry, roll or transaction costs.

## Next steps

1. Daily fits and a proper backtest with carry, roll and bid/ask costs.
2. Explain residuals with bond-specific factors (amount outstanding, age, CTD, BTF-based scarcity).
3. Extend the pricer to semi-annual coupons; run BTPs and Bonos, use Bunds as the risk-free leg.
