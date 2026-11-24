"""Build a LaTeX result sheet (and PDF if pdflatex is installed).

    python scripts/make_report.py --tag R40 --compare R40_funding
    python scripts/make_report.py --tag synth --compare synth_funding --raw data/synthetic --synthetic

Output: reports/<tag>/results.tex, results.pdf, figures, and data_snapshot.csv
(the last cross-section, so the underlying data can be inspected directly).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import analyze  # noqa: E402
from oathazard.bonds import settle_date  # noqa: E402
from oathazard.data import RawData  # noqa: E402


# ------------------------------------------------------------------ latex helpers
def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def fmt(v, nd=2):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "--" if np.isnan(v) else r"$\infty$"
        return f"{v:,.{nd}f}"
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    return esc(v)


def table(df: pd.DataFrame, caption: str, nd=2, align=None, small=False, fit=False) -> str:
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    head = " & ".join(rf"\textbf{{{esc(c)}}}" for c in cols)
    body = "\n".join(" & ".join(fmt(v, nd) for v in row) + r" \\" for row in df.itertuples(index=False))
    size = r"\footnotesize" if small else r"\small"
    tab = rf"""\begin{{tabular}}{{{align}}}\toprule
{head} \\ \midrule
{body}
\bottomrule\end{{tabular}}"""
    if fit:
        tab = r"\resizebox{\linewidth}{!}{" + tab + "}"
    return rf"""\begin{{table}}[H]\centering{size}
\caption{{{caption}}}
{tab}\end{{table}}
"""


# ------------------------------------------------------------------ data section
def data_snapshot(raw: RawData, sig: pd.DataFrame, cfg: dict, out: Path):
    last = sig["date"].max()
    snap = sig[sig["date"] == last].copy()
    st = raw.static
    snap["maturity"] = snap["isin"].map(st["maturity"])
    snap["clean"] = snap["isin"].map(raw.clean.loc[last])
    snap = snap.sort_values("ttm")
    csv = snap[["isin", "coupon", "maturity", "clean", "dirty", "ytm", "mdur", "zspread_bp",
                "z_resid_bp", "haz_resid_bp"]].copy()
    csv.to_csv(out / "data_snapshot.csv", index=False)
    tab = pd.DataFrame({
        "ISIN": snap["isin"], "Cpn %": snap["coupon"] * 100,
        "Maturity": pd.to_datetime(snap["maturity"]).dt.strftime("%Y-%m-%d"),
        "Clean": snap["clean"], "YTM %": snap["ytm"] * 100, "ModDur": snap["mdur"],
        "Z (bp)": snap["zspread_bp"], "Z resid": snap["z_resid_bp"], "Haz resid": snap["haz_resid_bp"],
    })

    # market inputs on the last date
    curve, src = raw.ois_curve(last)
    row = raw._row(last)
    mkt = [("Overnight (€STR)", raw.overnight(last) * 100, "%"),
           ("GC repo − €STR", raw.funding_spread(last) * 1e4, "bp")]
    for T in (2, 5, 10, 30):
        mkt.append((f"OIS zero {T}Y ({src})", float(curve.zero(T)) * 100, "%"))
    for k in ("GDBR10", "GFRN10"):
        if k in row and np.isfinite(row[k]):
            mkt.append((f"{k} generic yield", row[k], "%"))
    for T, v in sorted(raw.market_cds(last).items()):
        mkt.append((f"France CDS {T}Y (USD)", v, "bp"))
    mkt = pd.DataFrame(mkt, columns=["Input", "Value", "Unit"])

    # data figure: yield curve of the cross-section + OIS zero curve
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sc = ax.scatter(snap["ttm"], snap["ytm"] * 100, c=snap["coupon"] * 100, cmap="viridis", zorder=3)
    T = np.linspace(0.25, max(snap["ttm"].max(), 30), 200)
    ax.plot(T, (np.exp(curve.zero(T)) - 1) * 100, "k--", lw=1, label=f"OIS zero ({src}, annual)")
    ax.set_xlabel("maturity (y)"); ax.set_ylabel("%"); ax.legend(loc="lower right")
    ax.set_title(f"OAT yields vs risk-free curve, {last:%Y-%m-%d}")
    fig.colorbar(sc, ax=ax, label="coupon %"); fig.tight_layout()
    fig.savefig(out / "fig_data.png", dpi=150); plt.close(fig)

    # universe through time
    n = sig.groupby("date").size()
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(n.index, n.values, color="grey"); ax.set_ylabel("# bonds in fit")
    ax.set_title("Cross-section size"); fig.tight_layout()
    fig.savefig(out / "fig_universe.png", dpi=150); plt.close(fig)
    return last, tab, mkt


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="R40")
    ap.add_argument("--compare", default=None, help="second run, e.g. the funding-adjusted one")
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="stamp the sheet as synthetic data")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--extra", default="", help="comma-separated extra runs for the sensitivity table")
    ap.add_argument("--note", default="", help="free text (LaTeX) inserted after Setup")
    ap.add_argument("--outdir", default=None, help="output folder (default reports/<tag>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.outdir) if args.outdir else ROOT / "reports" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    raw = RawData(Path(args.raw), cfg)

    extra = [t for t in args.extra.split(",") if t]
    runs = [args.tag] + ([args.compare] if args.compare else []) + [t for t in extra if t not in (args.tag, args.compare)]
    res = {}
    for t in runs:
        f, sig, fq, tab = analyze.compute(ROOT / "results" / t, args.horizon)
        analyze.charts(f, sig, ROOT / "results" / t)
        res[t] = (f, sig, fq, tab)
    f, sig, fq, tab = res[args.tag]

    # sensitivity across runs (whole sample)
    sens = []
    for t in runs:
        ff, _, fqq, tb = res[t]
        a = tb[(tb["regime"] == "all")].set_index("signal")
        sens.append({"Run": t, "Knots": ff["knots"].iloc[0] if "knots" in ff else "0,3,7",
                     "Recovery": (f"fit (mean {ff['recovery'].mean():.0%})" if "recovery" in ff and ff["recovery"].std() > 1e-6
                                  else (f"{ff['recovery'].iloc[0]:.0%}" if "recovery" in ff else "40%")),
                     "Base": "Bund" if "Bund" in str(ff["curve_src"].iloc[-1]) else "€STR OIS",
                     "Med. RMSE": ff["rmse_bp"].median(), "RMSE 2024+": ff.loc[ff["date"] >= "2024-01-01", "rmse_bp"].median(),
                     "Share lambda<0": ff["breakdown"].mean(),
                     "Corr(cpn) haz": a.loc["Hazard residual", "corr_with_coupon"],
                     "IC haz": a.loc["Hazard residual", "rank_IC"], "IC Z": a.loc["Z-spread residual", "rank_IC"]})
    sens = pd.DataFrame(sens)
    if "recovery" in res[runs[-1]][0] and any("fit" in str(x) for x in sens["Recovery"]):
        rf = next(res[t][0] for t in runs if res[t][0]["recovery"].std() > 1e-6)
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.plot(rf["date"], rf["recovery"] * 100, color="tab:purple"); ax.set_ylabel("%")
        ax.set_title("Recovery estimated jointly with the hazard curve (each date)")
        fig.tight_layout(); fig.savefig(out / "fig_recovery.png", dpi=150); plt.close(fig)
    shutil.copy(ROOT / "results" / args.tag / "hazard_history.png", out / "fig_history.png")
    shutil.copy(ROOT / "results" / args.tag / "snapshot.png", out / "fig_snapshot.png")
    if args.compare:
        shutil.copy(ROOT / "results" / args.compare / "hazard_history.png", out / "fig_history_cmp.png")

    last, snap_tab, mkt = data_snapshot(raw, sig, cfg, out)

    # headline numbers
    lf = f[f["date"] == f["date"].max()].iloc[0]
    kn = [float(k) for k in str(lf.get("knots", "0,3,7")).split(",")]
    labs = [f"{a:g}-{b:g}Y" for a, b in zip(kn[:-1], kn[1:])] + [f"{kn[-1]:g}Y+"]
    lam_last = pd.DataFrame({"Bucket": labs, "Lambda (bp)": [lf[f"lambda_{j}"] * 1e4 for j in range(len(labs))],
                             "Spread approx. (bp)": [lf[f"lambda_{j}"] * 1e4 * (1 - lf.get("recovery", 0.4)) for j in range(len(labs))]})
    cds = pd.DataFrame({"Tenor": [f"{T}Y" for T in (1, 3, 5, 7, 10)],
                        "Implied (bp)": [lf[f"cds_impl_{T}y"] for T in (1, 3, 5, 7, 10)],
                        "Market (bp)": [lf[f"cds_mkt_{T}y"] for T in (1, 3, 5, 7, 10)]})
    cds["Basis (bp)"] = cds["Market (bp)"] - cds["Implied (bp)"]

    def fq_tab(fq):
        return fq.rename(columns={"regime": "Regime", "dates": "Dates", "median_rmse_bp": "Median RMSE (bp)",
                                  "share_breakdown": "Share lambda<0", "mean_cds5_impl": "Implied 5Y (bp)",
                                  "mean_cds5_mkt": "Market 5Y (bp)"})

    sigtab = tab[["regime", "signal", "corr_with_coupon", "half_life_periods", "rank_IC",
                  "LS_mean_bp", "LS_sharpe_per_period", "abs_resid_bp"]].rename(columns={
        "regime": "Regime", "signal": "Signal", "corr_with_coupon": "Corr(coupon)",
        "half_life_periods": "Half-life", "rank_IC": "Rank IC", "LS_mean_bp": "L/S (bp)",
        "LS_sharpe_per_period": "L/S SR", "abs_resid_bp": "Mean |resid|"})

    stamp = (r"\begin{center}\fcolorbox{red}{red!8}{\parbox{0.9\linewidth}{\centering\textbf{SYNTHETIC DATA.} "
             r"Generated by \texttt{make\_synthetic.py} with a known truth (recovery 40\%, GC far below "
             r"€STR under QE). Numbers test the pipeline, they are not results on real OATs.}}\end{center}"
             if args.synthetic else "")
    m = cfg["model"]
    freq = {"W-FRI": "weekly (Friday)", "B": "daily"}.get(m["frequency"], m["frequency"])
    tex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=1.8cm]{{geometry}}
\usepackage[T1]{{fontenc}}\usepackage[utf8]{{inputenc}}\usepackage{{lmodern}}
\usepackage{{booktabs,graphicx,float,amsmath,xcolor,hyperref,textcomp}}
\DeclareUnicodeCharacter{{20AC}}{{\texteuro}}\DeclareUnicodeCharacter{{2212}}{{-}}
\hypersetup{{colorlinks,linkcolor=blue}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}
\title{{\textbf{{OAT hazard-rate relative value}}\\\large Result sheet, run \texttt{{{esc(args.tag)}}}}}
\author{{Octave, Master 203 thesis project (\texttt{{oat-hazard-rv}})}}
\date{{Generated {pd.Timestamp.today():%d %B %Y}}}
\begin{{document}}\maketitle
{stamp}

\section*{{Setup}}
Sample {f['date'].min():%Y-%m-%d} to {f['date'].max():%Y-%m-%d}, {len(f):,} fit dates ({freq}),
{sig['isin'].nunique()} distinct fixed-coupon OATs, {f['n_bonds'].median():.0f} bonds per date (median).
Main run: recovery $R={f['recovery'].iloc[0]*100 if 'recovery' in f else 40:.0f}\%$ of face, hazard buckets starting at {esc(f['knots'].iloc[0] if 'knots' in f else '0,3,7')} years,
negative $\lambda$ allowed so that the QE breakdown is visible. Risk-free curve: €STR OIS
(EONIA OIS $-8.5$bp before Oct 2019).
\[
P=\sum_i CF_i\,D(t_i)\,Q(t_i)+100R\sum_k D(m_k)\big[Q(m_{{k-1}})-Q(m_k)\big],\qquad
Q(t)=e^{{-\int_0^t\lambda}},\qquad s\approx\lambda(1-R).
\]
Residual $=$ market yield $-$ model yield (bp, $>0$ = cheap). Baseline: Z-spread minus a cubic in maturity.

{args.note}

\section{{Data}}
Figure~\ref{{fig:data}} shows the last cross-section; Table~\ref{{tab:snap}} lists every bond in it
(also saved as \texttt{{data\_snapshot.csv}}).
\begin{{figure}}[H]\centering\includegraphics[width=0.85\linewidth]{{fig_data.png}}\caption{{Cross-section on {last:%Y-%m-%d}.}}\label{{fig:data}}\end{{figure}}
{table(mkt, f"Market inputs on {last:%Y-%m-%d}", nd=3, align="lrl")}
{table(snap_tab, f"OAT cross-section on {last:%Y-%m-%d}", nd=2, small=True).replace(r"\caption{", r"\caption{\label{tab:snap}", 1)}
\begin{{figure}}[H]\centering\includegraphics[width=0.85\linewidth]{{fig_universe.png}}\end{{figure}}

\section{{Fitted credit curve}}
{table(lam_last, f"Fitted hazard rates on {f['date'].max():%Y-%m-%d}", nd=1)}
{table(cds, "Bond-implied vs traded France CDS (last date). Traded CDS is USD-denominated.", nd=1)}
\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_history.png}}
\caption{{Run \texttt{{{esc(args.tag)}}}: hazard rates (red = fit wants $\lambda<0$), implied vs traded 5Y CDS, fit RMSE. Dotted lines: QE start (Mar 2015) and end of net purchases (Jun 2022).}}\end{{figure}}
{table(fq_tab(fq), f"Fit quality by regime, run {esc(args.tag)}", nd=2)}
"""
    if args.compare:
        fq2 = res[args.compare][2]
        tex += rf"""
\subsection*{{Comparison run \texttt{{{esc(args.compare)}}}}}
{table(fq_tab(fq2), f"Fit quality by regime, run {esc(args.compare)}", nd=2)}
\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_history_cmp.png}}\caption{{Run \texttt{{{esc(args.compare)}}}.}}\end{{figure}}
"""
    tex += rf"""
\section{{Sensitivity across specifications}}
{table(sens, "All runs, whole sample. RMSE in bp of yield. Corr(cpn): cross-sectional correlation of the hazard residual with the coupon. IC: rank IC of residual vs next-week compression.", nd=3, small=True, align="llllrrrrrr", fit=True)}
{"\\begin{figure}[H]\\centering\\includegraphics[width=0.85\\linewidth]{fig_recovery.png}\\end{figure}" if (out / "fig_recovery.png").exists() else ""}

\section{{Relative-value signals: hazard residual vs Z-spread residual}}
Corr(coupon): average cross-sectional correlation with the coupon (Z-spread RV should load on it, the
hazard residual should not). Half-life in fit periods from a pooled AR(1). Rank IC: correlation between the
signal and the next-period residual compression. L/S: long top quintile (cheap), short bottom quintile
(rich), DV01-neutral, residual compression in bp of yield, no carry, roll or costs.
{table(sigtab, f"Signal comparison, run {esc(args.tag)}, horizon {args.horizon} period(s)", nd=3, small=True)}
\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_snapshot.png}}\caption{{Residuals on the last date, coloured by coupon.}}\end{{figure}}

\section*{{Caveats}}
\begin{{itemize}}\itemsep0pt
\item Recovery is assumed; $\lambda$ and $R$ trade off. Re-run with \texttt{{--recovery 0.25}}.
\item Non-credit effects (liquidity, repo specialness, free float, CTD status) load onto $\lambda$ or the residuals.
\item Coupon dates are maturity anniversaries; irregular first coupons, holidays and business-day rolls ignored.
\item CDS is USD, bonds EUR: the basis includes redenomination/quanto effects.
\end{{itemize}}
\end{{document}}
"""
    (out / "results.tex").write_text(tex, encoding="utf-8")
    print("->", out / "results.tex")
    if shutil.which("pdflatex"):
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "results.tex"], cwd=out,
                               capture_output=True, text=True)
        print("->", out / "results.pdf" if r.returncode == 0 else f"pdflatex failed, see {out/'results.log'}")


if __name__ == "__main__":
    main()
