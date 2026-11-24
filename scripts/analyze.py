"""Compare hazard-model residuals with Z-spread residuals as RV signals.

    python scripts/analyze.py --tag R40

Tests, for each signal (>0 = cheap):
  1. Coupon bias      cross-sectional corr(signal, coupon), averaged over dates.
                      Z-spread residuals should load on coupon; hazard residuals should not.
  2. Mean reversion   pooled AR(1) of the residual per bond -> half-life in periods.
  3. Predictability   rank IC between signal_t and the subsequent residual compression.
  4. Long/short       long top quintile (cheap), short bottom quintile (rich), DV01-neutral,
                      P&L in bp of yield = residual compression. No carry, roll or costs.
All split by regime: pre-QE, QE (net purchases Mar-2015 to Jun-2022), post-QE.
Writes summary.md + charts into results/<tag>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = {"haz_resid_bp": "Hazard residual", "z_resid_bp": "Z-spread residual"}
REGIMES = [("pre-QE", None, "2015-03-08"), ("QE", "2015-03-09", "2022-06-30"), ("post-QE", "2022-07-01", None)]


def regime_slices(df):
    yield "all", df
    for name, a, b in REGIMES:
        m = pd.Series(True, index=df.index)
        if a:
            m &= df["date"] >= a
        if b:
            m &= df["date"] <= b
        if m.any():
            yield name, df[m]


def add_forward(sig: pd.DataFrame, col: str, h: int) -> pd.DataFrame:
    s = sig.sort_values(["isin", "date"]).copy()
    s[f"{col}_fwd"] = s.groupby("isin")[col].shift(-h)
    s[f"d_{col}"] = s[f"{col}_fwd"] - s[col]
    return s


def coupon_bias(df, col):
    c = df.groupby("date").apply(lambda g: g[col].corr(g["coupon"]) if len(g) > 4 else np.nan)
    return c.mean()


def ar1_half_life(df, col):
    x, y = df[col].values, df[f"{col}_fwd"].values
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan, np.nan
    phi = np.polyfit(x[m], y[m], 1)[0]
    hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
    return phi, hl


def rank_ic(df, col):
    def ic(g):
        g = g.dropna(subset=[col, f"d_{col}"])
        if len(g) < 6:
            return np.nan
        return g[col].rank().corr((-g[f"d_{col}"]).rank())
    return df.groupby("date").apply(ic).mean()


def long_short(df, col):
    def pnl(g):
        g = g.dropna(subset=[col, f"d_{col}"])
        if len(g) < 10:
            return np.nan
        q = g[col].quantile([0.2, 0.8])
        lng = g[g[col] >= q[0.8]]
        sht = g[g[col] <= q[0.2]]
        return (-lng[f"d_{col}"]).mean() + sht[f"d_{col}"].mean()
    p = df.groupby("date").apply(pnl).dropna()
    return p.mean(), (p.mean() / p.std() if p.std() > 0 else np.nan), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="R40")
    ap.add_argument("--horizon", type=int, default=1, help="periods ahead (1 = next fit date)")
    args = ap.parse_args()
    out = ROOT / "results" / args.tag
    fits = pd.read_parquet(out / "fits.parquet")
    sig = pd.read_parquet(out / "signals.parquet")
    for col in SIGNALS:
        sig = add_forward(sig, col, args.horizon)

    rows = []
    for reg, df in regime_slices(sig):
        for col, label in SIGNALS.items():
            phi, hl = ar1_half_life(df, col)
            mu, sr, n = long_short(df, col)
            rows.append({"regime": reg, "signal": label,
                         "corr_with_coupon": coupon_bias(df, col),
                         "AR1_phi": phi, "half_life_periods": hl,
                         "rank_IC": rank_ic(df, col),
                         "LS_mean_bp": mu, "LS_sharpe_per_period": sr, "n_dates": n,
                         "abs_resid_bp": df[col].abs().mean()})
    tab = pd.DataFrame(rows)

    # fit-quality / breakdown table
    f = fits.copy()
    fq = []
    for reg, df in regime_slices(f):
        fq.append({"regime": reg, "dates": len(df), "median_rmse_bp": df["rmse_bp"].median(),
                   "share_breakdown": df["breakdown"].mean(),
                   "mean_cds5_impl": df["cds_impl_5y"].mean(), "mean_cds5_mkt": df["cds_mkt_5y"].mean()})
    fq = pd.DataFrame(fq)

    # ---------------- charts
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for j, lab in enumerate(["0-3Y", "3-7Y", "7Y+"]):
        if f"lambda_{j}" in f:
            ax[0].plot(f["date"], f[f"lambda_{j}"] * 1e4, label=f"lambda {lab}")
    ax[0].axhline(0, color="k", lw=0.8)
    for d in f.loc[f["breakdown"], "date"]:
        ax[0].axvspan(d, d + pd.Timedelta(days=7), color="red", alpha=0.08, lw=0)
    ax[0].set_ylabel("bp"); ax[0].set_title("Fitted hazard rates (red = fit wants lambda < 0)"); ax[0].legend()
    ax[1].plot(f["date"], f["cds_impl_5y"], label="implied 5Y CDS (OATs)")
    ax[1].plot(f["date"], f["cds_mkt_5y"], label="market 5Y CDS (USD)")
    ax[1].set_ylabel("bp"); ax[1].legend(); ax[1].set_title("Bond-implied vs traded CDS")
    ax[2].plot(f["date"], f["rmse_bp"], color="grey")
    ax[2].set_ylabel("bp"); ax[2].set_title("Cross-sectional fit RMSE (yield)")
    for a in ax:
        for _, s, e in REGIMES:
            if s:
                a.axvline(pd.Timestamp(s), color="k", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(out / "hazard_history.png", dpi=130); plt.close(fig)

    last = sig["date"].max()
    snap = sig[sig["date"] == last]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col in zip(ax, SIGNALS):
        sc = a.scatter(snap["ttm"], snap[col], c=snap["coupon"] * 100, cmap="viridis")
        a.axhline(0, color="k", lw=0.8); a.set_xlabel("maturity (y)"); a.set_ylabel("bp (>0 cheap)")
        a.set_title(f"{SIGNALS[col]} on {last:%Y-%m-%d}")
    fig.colorbar(sc, ax=ax, label="coupon %"); fig.savefig(out / "snapshot.png", dpi=130); plt.close(fig)

    md = [f"# Results: {args.tag}\n", f"Horizon: {args.horizon} period(s). Signal > 0 = cheap.\n",
          "## Fit quality\n", fq.round(3).to_markdown(index=False), "\n\n## Signal comparison\n",
          tab.round(3).to_markdown(index=False), "\n\n![history](hazard_history.png)\n\n![snapshot](snapshot.png)\n"]
    (out / "summary.md").write_text("\n".join(md))
    print(fq.round(3).to_string(index=False)); print(); print(tab.round(3).to_string(index=False))
    print("->", out / "summary.md")


if __name__ == "__main__":
    main()
