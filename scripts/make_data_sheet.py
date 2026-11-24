"""PDF recap of every dataset pulled from Bloomberg: where it is, what it holds, coverage, gaps.

    python scripts/make_data_sheet.py      # -> reports/data/data_recap.pdf (+ .tex, figures)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report import esc, fmt  # noqa: E402

RAW = ROOT / "data" / "raw"
EX = RAW / "extra"
OUT = ROOT / "reports" / "data"
PC_PATH = r"C:\Users\suproc26\Downloads\work\oat-hazard-rv\data\raw"

DESCR = {
    "oat_static_all.parquet": "All FRTR lines from govtList (live + matured): static fields",
    "oat_static.parquet": "Model universe: fixed-coupon nominal annual OATs maturing after 2011",
    "oat_prices.parquet": "OAT daily clean price (BGN) and mid yield",
    "oat_amt_out_history.parquet": "OAT amount outstanding history (monthly steps, taps/buybacks)",
    "market.parquet": "€STR/EONIA fixings and OIS, France CDS, Bund/OAT generic yields",
    "extra/oat_quotes.parquet": "OAT bid/ask price and yield, Bloomberg Z-spread and ASW",
    "extra/market_extra.parquet": "ECB rates, Euribor, swaps, generic yields, sovereign CDS, iTraxx, vol, futures",
    "extra/discovered.parquet": "Series found by search: GC Pooling, BTF bills, €STR compounded, Bubill indices",
    **{f"extra/static_{t}.parquet": f"{n}: static fields (all lines in govtList)" for t, n in
       [("DBR", "Bunds"), ("OBL", "Bobls"), ("BKO", "Schatz"), ("BTPS", "BTPs"), ("SPGB", "Bonos")]},
    **{f"extra/prices_{t}.parquet": f"{n}: daily PX_LAST, PX_BID, PX_ASK, YLD_YTM_MID" for t, n in
       [("DBR", "Bunds"), ("OBL", "Bobls"), ("BKO", "Schatz"), ("BTPS", "BTPs"), ("SPGB", "Bonos")]},
}


def table(df, caption, nd=2, align=None, size=r"\footnotesize", long=False):
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    head = " & ".join(rf"\textbf{{{esc(c)}}}" for c in cols) + r" \\"
    body = "\n".join(" & ".join(fmt(v, nd) for v in row) + r" \\" for row in df.itertuples(index=False))
    if long:
        return (rf"""{{{size}\begin{{longtable}}{{{align}}}
\caption{{{caption}}}\\ \toprule {head} \midrule \endfirsthead
\toprule {head} \midrule \endhead
{body}
\bottomrule\end{{longtable}}}}
""")
    return (rf"""\begin{{table}}[H]\centering{size}\caption{{{caption}}}
\resizebox{{\ifdim\width>\linewidth\linewidth\else\width\fi}}{{!}}{{\begin{{tabular}}{{{align}}}\toprule
{head} \midrule
{body}
\bottomrule\end{{tabular}}}}\end{{table}}
""")


def dates_of(df):
    c = "date" if "date" in df else ("Date" if "Date" in df else None)
    return (pd.to_datetime(df[c]).min(), pd.to_datetime(df[c]).max()) if c else (pd.NaT, pd.NaT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. file inventory
    inv = []
    for f in sorted(list(RAW.glob("*.parquet")) + list(EX.glob("*.parquet"))):
        rel = f.relative_to(RAW).as_posix()
        df = pd.read_parquet(f)
        a, b = dates_of(df)
        key = "isin" if "isin" in df else ("ID_ISIN" if "ID_ISIN" in df else ("name" if "name" in df else "security"))
        inv.append({"File": rel, "Content": DESCR.get(rel, ""), "Rows": len(df), "Items": df[key].nunique(),
                    "From": a.strftime("%Y-%m-%d") if pd.notna(a) else "static",
                    "To": b.strftime("%Y-%m-%d") if pd.notna(b) else "", "MB": f.stat().st_size / 1e6})
    inv = pd.DataFrame(inv)

    # ---------------- 2. bond universes
    uni = []
    q = pd.read_parquet(EX / "oat_quotes.parquet")
    q["ba"] = q["PX_ASK"] - q["PX_BID"]
    ba_fr = q.groupby("isin")["ba"].median().median()
    st_all = pd.read_parquet(RAW / "oat_static_all.parquet")
    st = pd.read_parquet(RAW / "oat_static.parquet")
    px = pd.read_parquet(RAW / "oat_prices.parquet")
    uni.append({"Market": "France OAT (FRTR)", "Lines listed": len(st_all), "Fixed nominal kept": len(st),
                "With prices": px["isin"].nunique(), "Coupon freq": "annual",
                "Maturities": f"{pd.to_datetime(st['MATURITY']).min():%Y}-{pd.to_datetime(st['MATURITY']).max():%Y}",
                "Median bid-ask (pts)": ba_fr})
    for tk, name in [("DBR", "Germany Bund"), ("OBL", "Germany Bobl"), ("BKO", "Germany Schatz"),
                     ("BTPS", "Italy BTP"), ("SPGB", "Spain Bono")]:
        s = pd.read_parquet(EX / f"static_{tk}.parquet")
        p = pd.read_parquet(EX / f"prices_{tk}.parquet")
        fixed = s[s["CPN_TYP"].astype(str).str.upper().str.startswith("FIXED")]
        mat = pd.to_datetime(fixed["MATURITY"], errors="coerce")
        fr = pd.to_numeric(fixed["CPN_FREQ"], errors="coerce").mode()
        uni.append({"Market": f"{name} ({tk})", "Lines listed": len(s),
                    "Fixed nominal kept": p["isin"].nunique(), "With prices": p["isin"].nunique(),
                    "Coupon freq": {1: "annual", 2: "semi-annual"}.get(int(fr.iloc[0]) if len(fr) else 0, "mixed"),
                    "Maturities": f"{mat.min():%Y}-{mat.max():%Y}",
                    "Median bid-ask (pts)": (p["PX_ASK"] - p["PX_BID"]).groupby(p["isin"]).median().median()})
    uni = pd.DataFrame(uni)

    # ---------------- 3. market series
    m1 = pd.read_parquet(RAW / "market.parquet")
    m2 = pd.read_parquet(EX / "market_extra.parquet")
    dsc = pd.read_parquet(EX / "discovered.parquet")
    keep = dsc["security"].str.match(r"^(SGCP(ON|TN|1W|1M|3M|6M|12M|FR) |GBTF|XESTR|GETB1)")
    dsc = dsc[keep].assign(name=lambda d: d["security"])
    allm = pd.concat([m1.assign(src="market"), m2.assign(src="market_extra"), dsc.assign(src="discovered")],
                     ignore_index=True).dropna(subset=["PX_LAST"])

    def group(n):
        n = str(n)
        if n.startswith(("estr", "eonia", "XESTR")): return "1 €STR / EONIA"
        if n.startswith(("ecb", "euribor", "eusa")): return "2 Policy, Euribor, swaps"
        if n.startswith(("SGCP", "repo", "GBTF", "fr_", "de_", "GETB")): return "3 Funding, bills"
        if n.startswith(("GDBR", "GFRN", "GBTP", "GSPG")): return "4 Generic govt yields"
        if n.startswith("cds"): return "5 Sovereign CDS"
        if n.startswith("itraxx"): return "6 Credit indices"
        if n.endswith("_fut"): return "8 Futures"
        return "7 Vol, equity"
    series = (allm.groupby(["name", "security"])
              .agg(first=("date", "min"), last=("date", "max"), obs=("date", "size"),
                   last_value=("PX_LAST", "last")).reset_index())
    series["Group"] = series["name"].map(group)
    series = series.sort_values(["Group", "name"])
    ser_tab = pd.DataFrame({"Group": series["Group"].str[2:], "Name": series["name"], "Bloomberg ticker": series["security"],
                            "From": series["first"].dt.strftime("%Y-%m-%d"), "To": series["last"].dt.strftime("%Y-%m-%d"),
                            "Obs": series["obs"], "Last": series["last_value"]})

    # ---------------- 4. charts
    piv = allm.pivot_table(index="date", columns="name", values="PX_LAST").sort_index()
    w = piv.resample("W-FRI").last()

    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    a = ax[0, 0]
    for c, lab in [("estr_fixing", "€STR"), ("eonia_fixing", "EONIA"), ("ecb_dfr", "ECB deposit rate"),
                   ("SGCPON Index", "GC Pooling ON"), ("GBTF3MO Index", "France 3M BTF")]:
        if c in w: a.plot(w.index, w[c], label=lab, lw=1)
    a.set_title("Overnight, policy and funding rates (%)"); a.legend(fontsize=7)
    a = ax[0, 1]
    for c, lab in [("GFRN10", "France 10Y"), ("GDBR10", "Germany 10Y"), ("GBTPGR10", "Italy 10Y"),
                   ("GSPG10YR", "Spain 10Y"), ("estr_ois_10", "€STR OIS 10Y")]:
        if c in w: a.plot(w.index, w[c], label=lab, lw=1)
    a.set_title("10Y yields (%)"); a.legend(fontsize=7)
    a = ax[1, 0]
    for c, lab in [("cds_5y", "France"), ("cds_german_5y", "Germany"), ("cds_italy_5y", "Italy"),
                   ("cds_spain_5y", "Spain")]:
        if c in w: a.plot(w.index, w[c], label=lab, lw=1)
    a.set_title("Sovereign 5Y CDS, USD (bp)"); a.legend(fontsize=7)
    a = ax[1, 1]
    if "GFRN10" in w and "GDBR10" in w:
        a.plot(w.index, (w["GFRN10"] - w["GDBR10"]) * 100, label="OAT - Bund 10Y", color="tab:blue", lw=1)
    if "GFRN10" in w and "estr_ois_10" in w:
        a.plot(w.index, (w["GFRN10"] - w["estr_ois_10"]) * 100, label="OAT - €STR OIS 10Y", color="tab:red", lw=1)
    a.axhline(0, color="k", lw=0.6); a.set_title("French 10Y spreads (bp)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_market.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    n = px.dropna(subset=["PX_LAST"]).groupby("date")["isin"].nunique().resample("W-FRI").last()
    ax[0].plot(n.index, n.values, color="grey"); ax[0].set_title("OATs priced per day (all maturities)")
    bam = q.groupby("date")["ba"].median().resample("ME").median()
    ax[1].plot(bam.index, bam.values, color="tab:green"); ax[1].set_title("OAT median bid-ask (price points), monthly")
    fig.tight_layout(); fig.savefig(OUT / "fig_oat.png", dpi=140); plt.close(fig)

    # coverage heat strip for the main series
    cov_names = [c for c in ["estr_fixing", "eonia_fixing", "estr_ois_10", "eonia_ois_10", "SGCPON Index",
                             "GBTF3MO Index", "ecb_dfr", "eusa_10", "cds_1y", "cds_3y", "cds_5y", "cds_7y", "cds_10y",
                             "cdseur_5y", "cds_german_5y", "cds_italy_5y", "cds_spain_5y", "itraxx_main_5y",
                             "oat_fut", "v2x", "move"] if c in piv]
    cov = piv[cov_names].notna().resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(cov_names) + 1))
    ax.imshow(cov.T.values, aspect="auto", cmap="Greens", vmin=0, vmax=1,
              extent=[matplotlib.dates.date2num(cov.index[0]), matplotlib.dates.date2num(cov.index[-1]), len(cov_names), 0])
    ax.set_yticks(np.arange(len(cov_names)) + 0.5); ax.set_yticklabels(cov_names, fontsize=7)
    ax.xaxis_date(); ax.set_title("Coverage: share of days with data each month (white = missing)")
    fig.tight_layout(); fig.savefig(OUT / "fig_coverage.png", dpi=140); plt.close(fig)

    # ---------------- 5. LaTeX
    missing = pd.DataFrame([
        ("French GC repo (RepoFunds Rate France GC, REFRGCFR Index)", "Valid ticker, history not entitled", "Use BTF 3M - €STR or GC Pooling"),
        ("Broker French GC repo (RB000xxx Mmkt)", "Not entitled", "Same"),
        ("Repo rate / specialness per OAT", "No field on this terminal", "Residual factor proxies (age, amount outstanding)"),
        ("ECB/PSPP holdings per ISIN", "Not published per bond", "Amount outstanding, QE dummies"),
        ("GDBR 3M/6M/1Y, GFRN 1Y, GFRN50 generics", "Invalid tickers", "GETB1 (Bubill), GBTF bills, OAT 2072 itself"),
        ("iTraxx Senior Financials generic", "Invalid ticker", "iTraxx Main / Crossover"),
        ("France CDS EUR 3Y/7Y", "Only since Sep 2026", "USD curve"),
        ("France CDS USD 20Y/30Y", "Only since Sep 2025", "USD 10Y"),
    ], columns=["Item", "Status", "Workaround"])

    tex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=1.6cm,landscape]{{geometry}}
\usepackage[T1]{{fontenc}}\usepackage[utf8]{{inputenc}}\usepackage{{lmodern}}
\usepackage{{booktabs,longtable,graphicx,float,xcolor,hyperref,textcomp}}
\DeclareUnicodeCharacter{{20AC}}{{\texteuro}}\DeclareUnicodeCharacter{{2212}}{{-}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}
\title{{\textbf{{Data recap: Bloomberg pulls for the OAT hazard-rate thesis}}}}
\author{{Octave, Master 203 (\texttt{{oat-hazard-rv}})}}
\date{{Pulled 24 November 2026 from the Dauphine Bloomberg terminal}}
\begin{{document}}\maketitle

\section*{{Where the data is}}
\begin{{itemize}}\itemsep0pt
\item \textbf{{Dauphine PC (a215-06):}} \texttt{{{esc(PC_PATH)}}} and its \texttt{{extra}} subfolder. Parquet files, open with
\texttt{{pandas.read\_parquet}}. This is the master copy.
\item \textbf{{Claude workspace:}} identical copy, used for the fits and reports.
\item \textbf{{Not on GitHub}} (\texttt{{data/}} is git-ignored): Bloomberg's terms do not allow republishing the data. The code
in the repo (\texttt{{scripts/fetch\_bloomberg.py}}, \texttt{{scripts/fetch\_extra.py}}) re-pulls everything from any terminal.
\item Pulled through the Bloomberg Desktop API (\texttt{{blpapi}}) from Python, daily, 1 Jan 2010 to 24 Nov 2026.
\end{{itemize}}
Totals: {len(inv)} files, {inv['Rows'].sum():,} rows, {inv['MB'].sum():.1f} MB;
{uni['With prices'].sum()} government bonds with daily prices across 4 countries; {len(ser_tab)} market time series.

{table(inv, "File inventory (paths relative to data/raw)", nd=1, align="llrrllr", long=True)}

{table(uni, "Bond universes. Lines listed = everything Bloomberg's govtList returns for the ticker (live and matured); kept = fixed-coupon, non-linked, maturing after 2011.", nd=3, align="lrrrllr")}

\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_oat.png}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_market.png}}\caption{{Key series, weekly.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig_coverage.png}}\end{{figure}}

{table(missing, "Not available or partial, and the workaround used", nd=0, align="lll", size=r"\small")}

\section*{{Market series ({len(ser_tab)})}}
Rates in \%, CDS and iTraxx in bp, futures in price, vol indices in vol points. \texttt{{discovered}} series were found by keyword
search on the terminal and kept if they had history.
{table(ser_tab, "All market time series", nd=3, align="lllllrr", long=True)}
\end{{document}}
"""
    (OUT / "data_recap.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "data_recap.tex"], cwd=OUT,
                           capture_output=True, text=True)
    print("->", OUT / "data_recap.pdf", "OK" if r.returncode == 0 else "pdflatex error, see log")


if __name__ == "__main__":
    main()
