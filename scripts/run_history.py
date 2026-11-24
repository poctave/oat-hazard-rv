"""Fit the hazard curve on every date and store fits + per-bond signals.

    python scripts/run_history.py                    # uses config.yaml
    python scripts/run_history.py --funding          # discount at €STR + (GC - €STR)
    python scripts/run_history.py --raw data/synthetic

Outputs (results/<tag>/):
    fits.parquet       one row per date: lambdas, rmse, breakdown flag, implied vs market CDS
    signals.parquet    one row per (date, bond): hazard residual, Z-spread, Z-spread residual
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oathazard.bonds import settle_date  # noqa: E402
from oathazard.data import RawData  # noqa: E402
from oathazard.hazard import CrossSection, cds_par_spread, fit_hazard, zspread_rv  # noqa: E402

CDS_TENORS = (1, 3, 5, 7, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--funding", action="store_true", help="apply funding adjustment (rate - €STR)")
    ap.add_argument("--funding-src", default="gc", choices=["gc", "btf3m"], help="gc: GC Pooling ON; btf3m: French 3M bills")
    ap.add_argument("--freq", default=None, help="override model.frequency (B, W-FRI, ...)")
    ap.add_argument("--recovery", default=None, help="fixed recovery (e.g. 0.4) or 'fit' to estimate it each date")
    ap.add_argument("--tag", default=None, help="output sub-folder name")
    ap.add_argument("--base", default="ois", choices=["ois", "bund"],
                    help="risk-free curve: €STR OIS (default) or Bund (France-vs-Germany credit)")
    ap.add_argument("--knots", default=None, help="hazard bucket edges, e.g. 0,3,7,15,30")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m = cfg["model"]
    fit_rec = args.recovery == "fit"
    recovery = None if fit_rec else (float(args.recovery) if args.recovery is not None else m["recovery"])
    funding = args.funding or m["funding_adjustment"]
    knots = tuple(float(k) for k in args.knots.split(",")) if args.knots else tuple(m["hazard_knots"])
    tag = args.tag or f"{'Rfit' if fit_rec else f'R{int(recovery * 100)}'}{'_funding' if funding else ''}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)

    data = RawData(Path(args.raw), cfg)
    idx = data.clean.index
    freq = args.freq or m["frequency"]
    dates = idx if freq == "B" else pd.Series(idx, index=idx).resample(freq).last().dropna().values
    dates = pd.DatetimeIndex(dates)

    fits, sigs = [], []
    x0 = None
    t0 = time.time()
    for n, d in enumerate(dates):
        settle = settle_date(d)
        try:
            curve, curve_src = data.ois_curve(d)
        except ValueError:
            continue
        fs = data.funding_spread(d, args.funding_src)
        if funding and np.isfinite(fs):
            curve = curve.shifted(fs)
        if args.base == "bund":
            fn = data.bund_spread_fn(d, curve)
            if fn is None:
                continue
            curve = curve.shifted_term(fn)
            curve_src += "+Bund"
        bonds, dirty = data.cross_section(d, settle)
        if len(bonds) < len(knots) + 3:
            continue
        cs = CrossSection(bonds, dirty, settle, curve)
        res = fit_hazard(cs, recovery, knots, m["allow_negative_lambda"], x0)
        x0 = res.hazard.lambdas
        rec_d = res.recovery
        z, zres = zspread_rv(cs)

        row = {"date": d, "n_bonds": cs.n, "rmse_bp": res.rmse_bp, "breakdown": res.breakdown,
               "success": res.success, "curve_src": curve_src, "funding_spread_bp": fs * 1e4,
               "ois_10y_zero": float(curve.zero(10.0))}
        row["recovery"] = res.recovery
        row["knots"] = ",".join(f"{k:g}" for k in knots)
        for j, lam in enumerate(res.hazard.lambdas):
            row[f"lambda_{j}"] = lam
        mk = data.market_cds(d)
        for T in CDS_TENORS:
            row[f"cds_impl_{T}y"] = cds_par_spread(res.hazard, curve, rec_d, T) * 1e4
            row[f"cds_mkt_{T}y"] = mk.get(T, np.nan)
        fits.append(row)
        for b, p, y, md_, rr, zz, zr, T in zip(bonds, dirty, cs.ytm, cs.mdur, res.resid_bp, z, zres, cs.ttm):
            sigs.append({"date": d, "isin": b.isin, "coupon": b.coupon, "ttm": T, "dirty": p,
                         "ytm": y, "mdur": md_, "haz_resid_bp": rr, "zspread_bp": zz, "z_resid_bp": zr})
        if n % 25 == 0:
            print(f"{d:%Y-%m-%d}  bonds={cs.n:3d}  rmse={res.rmse_bp:6.2f}bp  "
                  f"lambda={np.round(res.hazard.lambdas * 1e4, 1)}bp  [{time.time() - t0:5.0f}s]", flush=True)

    pd.DataFrame(fits).to_parquet(out / "fits.parquet")
    pd.DataFrame(sigs).to_parquet(out / "signals.parquet")
    print(f"done: {len(fits)} dates -> {out}")


if __name__ == "__main__":
    main()
