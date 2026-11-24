# oat-hazard-rv

Relative value on French government bonds (OATs) with a hazard-rate model instead of Z-spread.

**Question:** does coupon-adjusted RV (residuals from one implied default curve fitted across all OATs) give cleaner rich/cheap signals than Z-spread RV?

Z-spread discounts every promised cash flow at €STR + z, which assumes 100% survival and no recovery. Under credit stress a low-coupon bond (price 70) loses far less in default than a par bond, since the claim is on face value. Z-spread then makes low coupons look rich and high coupons cheap, even with identical credit. A reduced-form model prices that effect explicitly:

```
P = Σ CF_i · D(t_i) · Q(t_i)  +  100 · R · Σ D(m_k) · [Q(m_{k-1}) − Q(m_k)]
Q(t) = exp(−∫ λ),   D = €STR OIS discount factor,   spread ≈ λ (1 − R)
```

One piecewise-constant λ curve (0-3Y, 3-7Y, 7Y+) is fitted across all OATs on each date by minimising duration-scaled price errors (≈ yield errors). Residuals (market yield − model yield, > 0 = cheap) are the trade signals. The fitted curve gives an implied CDS curve to compare with traded France CDS.

The project follows a practitioner's global-fit approach, including the point that under QE repo traded far below €STR, so the fit wants λ < 0 (survival above 100%). The code lets λ go negative on purpose so the breakdown is visible, and `--funding` discounts at €STR + (GC − €STR) to test whether a funding adjustment repairs it.

## Layout

```
oathazard/        library
  curves.py       €STR / EONIA OIS bootstrap, log-linear discount curve
  bonds.py        OAT schedule, accrued (ACT/ACT ICMA), yield, duration
  hazard.py       hazard pricing, Z-spread, global fit, implied CDS
  data.py         turns the raw Bloomberg pull into model inputs
scripts/
  fetch_bloomberg.py   pull data via blpapi (Terminal must be open)
  run_history.py       fit every date -> results/<tag>/fits.parquet, signals.parquet
  analyze.py           coupon bias, mean reversion, rank IC, long/short by regime
  make_synthetic.py    fake pull with a known truth, to test without Bloomberg
tests/                 unit tests (toy example from the notes, bootstrap, yields)
config.yaml            tickers and model settings
```

## Run it

```bash
pip install -r requirements.txt
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

# 0. matured OATs: Terminal SRCH <GO>, ticker FRTR, include matured, fixed coupon,
#    export ISINs to data/oat_isins_extra.csv (govtList only returns live bonds)
python scripts/fetch_bloomberg.py --check     # every ticker OK?  fix config.yaml if not
python scripts/fetch_bloomberg.py             # full pull, 2010 -> today, into data/raw/

python scripts/run_history.py                 # -> results/R40/
python scripts/run_history.py --funding       # -> results/R40_funding/
python scripts/run_history.py --recovery 0.25 # recovery sensitivity
python scripts/analyze.py --tag R40
```

Test without Bloomberg:

```bash
python -m pytest -q
python scripts/make_synthetic.py
python scripts/run_history.py --raw data/synthetic --tag synth
python scripts/analyze.py --tag synth
```

On synthetic data (true recovery 40%, GC far below €STR under QE), the plain fit goes negative through QE as expected, and the funding-adjusted fit recovers the true λ.

## Data pulled

| Item | Source | Use |
|---|---|---|
| OAT universe + static (coupon, maturity, issue date, amount outstanding) | govtList FRTR + SRCH export | cash flows, filters |
| OAT daily clean prices (BGN) and yields | BDH | the cross-section |
| Amount outstanding history | bulk field `AMOUNT_OUTSTANDING_HISTORY` | bond-specific factor |
| €STR fixing, €STR OIS (EONIA OIS − 8.5bp before Oct 2019) | BDH | risk-free curve |
| France sovereign CDS 1-10Y (USD, CR14) | BDH | implied vs traded basis |
| French GC repo | BDH | funding adjustment |
| Bund and OAT generic yields | BDH | benchmark |

Bloomberg data stays local (`data/` is git-ignored): Bloomberg's terms do not allow republishing it.

## Simplifications (v1) and known limits

- Fixed-coupon nominal annual OATs only (no OATi/€i). Coupon dates are maturity anniversaries; irregular first coupons, holidays and business-day rolls are ignored.
- Discounting on ACT/365 times; OIS on ACT/360; default independent of rates.
- Recovery fixed (40%) and on face value. λ and R trade off; run the recovery sensitivity.
- Everything that is not credit (liquidity, repo specialness, free float, CTD status) loads onto λ or into the residuals.
- CDS is USD-denominated while bonds are EUR: the CDS-bond basis includes redenomination / quanto effects.
- ECB holdings per ISIN are not published (only aggregates), so v1 uses amount outstanding as the bond-specific factor.
- Long/short P&L is residual compression in bp of yield, DV01-neutral, without carry, roll or transaction costs.

## Next steps

1. Run on real data; check the QE breakdown and the funding fix.
2. Extend to BTPs and Bonos (wider spreads, where the coupon effect should be stronger).
3. Explain residuals with bond-specific factors (amount outstanding, age, repo specialness proxy).
4. Proper backtest with carry/roll and bid-ask.

## Result sheet (LaTeX)

```bash
python scripts/make_report.py --tag R40 --compare R40_funding
# synthetic: python scripts/make_report.py --tag synth --compare synth_funding --raw data/synthetic --synthetic
```

Writes `reports/<tag>/results.tex` (+ `results.pdf` if pdflatex is installed) with: market inputs and the
full OAT cross-section on the last date (also as `data_snapshot.csv`), fitted hazard curve, implied vs traded
CDS, fit quality by regime, funding-adjusted comparison, and the signal comparison table.
