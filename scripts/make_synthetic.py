"""Generate a fake Bloomberg pull (same schema as fetch_bloomberg.py) with a KNOWN truth,
so the whole pipeline can be tested without a Terminal.

Truth: hazard rate path with a 2011-12 crisis and a 2024 stress, recovery 40%,
GC repo far below €STR during QE (bonds richer than risk-free discounting allows),
plus mean-reverting bond-specific noise (the "real" rich/cheap signal).

    python scripts/make_synthetic.py && python scripts/run_history.py --raw data/synthetic --tag synth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oathazard.bonds import Bond, settle_date  # noqa: E402
from oathazard.curves import ACT360, DiscountCurve  # noqa: E402
from oathazard.hazard import CrossSection, HazardCurve  # noqa: E402

rng = np.random.default_rng(7)
OUT = ROOT / "data" / "synthetic"
TENORS = ["A", "C", "F", "I", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "15", "20", "25", "30", "40", "50"]
TY = {"A": 1 / 12, "C": .25, "F": .5, "I": .75}


def level(d):  # overnight-ish rate path (decimal)
    y = d.year + d.dayofyear / 365
    return float(np.interp(y, [2010, 2012, 2014.5, 2016, 2022.3, 2023.7, 2025, 2027],
                           [0.008, 0.002, 0.0, -0.005, -0.005, 0.039, 0.02, 0.019]))


def hazard(d):  # flat-ish lambda path (decimal)
    y = d.year + d.dayofyear / 365
    return float(np.interp(y, [2010, 2011.5, 2011.9, 2012.6, 2014, 2020.2, 2020.4, 2022, 2024.4, 2024.6, 2027],
                           [0.004, 0.012, 0.022, 0.012, 0.005, 0.004, 0.007, 0.004, 0.006, 0.0105, 0.010]))


def funding(d):  # GC - €STR (decimal): deeply negative under QE
    y = d.year + d.dayofyear / 365
    return float(np.interp(y, [2010, 2015, 2016, 2021.5, 2022.5, 2023.5, 2027],
                           [0.0, -0.001, -0.004, -0.006, -0.002, 0.0003, 0.0006]))


def zero_curve(d):
    r0 = level(d)
    T = np.linspace(0, 60, 241)
    z = r0 + 0.012 * (1 - np.exp(-T / 4)) + 0.004 * (T / 30)
    return DiscountCurve(T, np.exp(-z * T))


def par_ois(curve):
    out = {}
    for t in TENORS:
        T = TY.get(t, None) or float(t)
        if T <= 1:
            out[t] = (1 / curve.df(T) - 1) / (T * ACT360)
        else:
            n = int(T)
            ann = sum(ACT360 * curve.df(k) for k in range(1, n + 1))
            out[t] = (1 - curve.df(n)) / ann
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2010-01-01", "2026-11-20", freq="W-FRI")
    # universe: one 10Y line per year, a 30Y every 3 years, coupons ~ yield at issue
    bonds = []
    for yr in range(1999, 2027):
        for tenor in ([10, 30] if yr % 3 == 0 else [10]):
            iss = pd.Timestamp(f"{yr}-0{(tenor % 9) + 1}-25")
            if iss > dates[-1]:
                continue
            yld = max(level(iss) + 0.012 + 0.006 * (tenor > 10) + 0.004, 0.0)
            cpn = np.floor(yld / 0.0025) * 0.0025
            mat = iss + pd.DateOffset(years=tenor)
            isin = f"FR{yr:04d}{tenor:02d}SYN"
            bonds.append(Bond(isin, cpn, mat, iss, f"FRTR {cpn*100:.2f} {mat:%m/%d/%y}"))
    noise = {b.isin: 0.0 for b in bonds}
    px_rows, mkt_rows = [], []
    for d in dates:
        settle = settle_date(d)
        base = zero_curve(d)
        live = [b for b in bonds if b.issue <= d and b.ttm(settle) > 0.3]
        lam = hazard(d)
        hc = HazardCurve((0.0, 3.0, 7.0), np.array([lam * 0.7, lam, lam * 1.15]))
        cs = CrossSection(live, np.full(len(live), 100.0), settle, base.shifted(funding(d)))
        dirty = cs.model_prices(hc, 0.40)
        for i, b in enumerate(live):
            noise[b.isin] = 0.85 * noise[b.isin] + rng.normal(0, 1.2)      # bp, AR(1)
            dirty_i = dirty[i] * (1 - noise[b.isin] * 1e-4 * cs.mdur[i])   # >0 noise = cheap
            px_rows.append({"date": d, "isin": b.isin, "security": f"{b.isin}@BGN Govt",
                            "PX_LAST": dirty_i - b.accrued(settle), "YLD_YTM_MID": np.nan})
        r = par_ois(base)
        on = level(d)
        rows = {"estr_fixing": on * 100 if d >= pd.Timestamp("2019-10-02") else np.nan,
                "eonia_fixing": (on + 0.00085) * 100,
                "repo_rfr_france": (on + funding(d)) * 100,
                "GDBR10": (base.zero(10) - 0.003) * 100}
        for t, v in r.items():
            if d >= pd.Timestamp("2019-10-02"):
                rows[f"estr_ois_{t}"] = v * 100
            rows[f"eonia_ois_{t}"] = (v + 0.00085) * 100
        for T in (1, 3, 5, 7, 10):
            rows[f"cds_{T}y"] = lam * 0.6 * 1e4 * (0.8 + 0.04 * T) + rng.normal(0, 2)
        for k, v in rows.items():
            mkt_rows.append({"date": d, "name": k, "security": k, "PX_LAST": v})
    st = pd.DataFrame([{"security": f"{b.isin} Govt", "ID_ISIN": b.isin, "CPN": b.coupon * 100,
                        "MATURITY": b.maturity, "ISSUE_DT": b.issue, "SECURITY_NAME": b.name,
                        "AMT_OUTSTANDING": 3e10} for b in bonds])
    st.to_parquet(OUT / "oat_static.parquet")
    pd.DataFrame(px_rows).to_parquet(OUT / "oat_prices.parquet")
    pd.DataFrame(mkt_rows).to_parquet(OUT / "market.parquet")
    print(f"synthetic: {len(bonds)} bonds, {len(dates)} dates -> {OUT}")


if __name__ == "__main__":
    main()
