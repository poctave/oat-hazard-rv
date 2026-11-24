"""Second Bloomberg pull: everything else the thesis needs.

    python scripts/fetch_extra.py            # all blocks
    python scripts/fetch_extra.py --only market,discover

Blocks (each writes data/raw/extra/<block>*.parquet and logs to extra.log):
  oat_quotes   bid/ask + Bloomberg Z-spread / ASW for the OAT universe already pulled
  universes    Bunds (DBR/OBL/BKO), BTPs (BTPS), Bonos (SPGB): static + daily prices
  market       ECB rates, bills, Euribor swaps, other sovereign CDS, credit indices, vol, futures
  discover     searches for repo / GC / SovX / bill series, keeps those that have history
Only fields that exist are kept; invalid tickers are logged, never fatal.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_bloomberg import BBG, STATIC_FIELDS, bond_security  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = RAW / "extra"
LOG = open(ROOT / "extra.log", "a", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    print(f"{time.strftime('%H:%M:%S')} {msg}", file=LOG, flush=True)


def report(label, err):
    if err:
        log(f"  {label}: {len(err)} problem(s), first: {err[:5]}")


MARKET = {
    # ECB policy rates
    "ecb_dfr": "EUORDEPO Index", "ecb_mro": "EURR002W Index",
    # bills / short generics
    "fr_3m": "GFRN3M Index", "fr_6m": "GFRN6M Index", "fr_1y": "GFRN1Y Index",
    "de_3m": "GDBR3M Index", "de_6m": "GDBR6M Index", "de_1y": "GDBR1Y Index",
    "euribor_3m": "EUR003M Index", "euribor_6m": "EUR006M Index",
    # more generic govt yields
    **{f"GFRN{t}": f"GFRN{t} Index" for t in (7, 15, 20, 50)},
    **{f"GDBR{t}": f"GDBR{t} Index" for t in (7, 15, 20)},
    **{f"GBTPGR{t}": f"GBTPGR{t} Index" for t in (2, 5, 10, 30)},
    **{f"GSPG{t}YR": f"GSPG{t}YR Index" for t in (2, 5, 10, 30)},
    # Euribor 6M swaps (asset-swap reference curve)
    **{f"eusa_{t}": f"EUSA{t} Curncy" for t in (2, 5, 7, 10, 15, 20, 30)},
    # sovereign CDS
    **{f"cds_{c.lower()}_{t}y": f"{c} CDS USD SR {t}Y D14 Corp"
       for c in ("GERMAN", "ITALY", "SPAIN", "BELGIUM", "NETHER") for t in (5, 10)},
    **{f"cds_france_{t}y_extra": f"FRANCE CDS USD SR {t}Y D14 Corp" for t in (2, 4, 20, 30)},
    **{f"cdseur_france_{t}y": f"FRANCE CDS EUR SR {t}Y D14 Corp" for t in (1, 2, 3, 7)},
    # credit indices
    "itraxx_main_5y": "ITRX EUR CDSI GEN 5Y Corp", "itraxx_xover_5y": "ITRX XOVER CDSI GEN 5Y Corp",
    "itraxx_snrfin_5y": "ITRX SNRFIN CDSI GEN 5Y Corp",
    # vol / risk
    "v2x": "V2X Index", "vix": "VIX Index", "move": "MOVE Index", "sx5e": "SX5E Index", "cac": "CAC Index",
    # futures (generic front)
    "oat_fut": "OAT1 Comdty", "bund_fut": "RX1 Comdty", "btp_fut": "IK1 Comdty",
    "bobl_fut": "OE1 Comdty", "buxl_fut": "UB1 Comdty",
}
DISCOVER_QUERIES = ["RepoFunds Rate", "GC Pooling", "Eurepo", "repo France", "SovX",
                    "France Treasury Bill", "BTF", "German Bubill", "ESTR compounded", "EUR OIS ESTR"]
DISCOVER_KEYWORDS = ("repo", "gc", "sovx", "bill", "btf", "bubill", "estr")


def block_oat_quotes(b, start, end):
    st = pd.read_parquet(RAW / "oat_static.parquet")
    secs = {bond_security(i, "BGN"): i for i in st["ID_ISIN"]}
    fields = ["PX_BID", "PX_ASK", "Z_SPRD_MID", "ASSET_SWAP_SPD_MID", "YLD_YTM_BID", "YLD_YTM_ASK"]
    df, err = b.bdh(list(secs), fields, start, end)
    report("oat_quotes", err)
    df["isin"] = df["security"].map(secs)
    df.to_parquet(OUT / "oat_quotes.parquet")
    log(f"oat_quotes: {len(df):,} rows, fields present: {[c for c in fields if c in df]}")


def block_universes(b, start, end, tickers=("DBR", "OBL", "BKO", "BTPS", "SPGB")):
    for tk in tickers:
        keys = b.govt_list(tk)
        st, err = b.bdp(keys, STATIC_FIELDS)
        report(f"{tk} static", err)
        if st.empty:
            log(f"{tk}: no securities"); continue
        st["issuer_ticker"] = tk
        st.to_parquet(OUT / f"static_{tk}.parquet")
        m = pd.to_datetime(st["MATURITY"], errors="coerce") > pd.Timestamp(start) + pd.DateOffset(years=1)
        fixed = st["CPN_TYP"].astype(str).str.upper().str.startswith("FIXED")
        nominal = st["INFLATION_LINKED_INDICATOR"].astype(str).str.upper() != "Y"
        keep = st[m & fixed & nominal]
        log(f"{tk}: {len(st)} securities, {len(keep)} fixed nominal maturing after start")
        secs = {bond_security(i, "BGN"): i for i in keep["ID_ISIN"].dropna()}
        px, err = b.bdh(list(secs), ["PX_LAST", "PX_BID", "PX_ASK", "YLD_YTM_MID"], start, end)
        report(f"{tk} prices", err)
        if not px.empty:
            px["isin"] = px["security"].map(secs)
            px.to_parquet(OUT / f"prices_{tk}.parquet")
        log(f"{tk}: {len(px):,} price rows, {px['isin'].nunique() if not px.empty else 0} bonds")


def block_market(b, start, end):
    chk, err = b.bdp(list(MARKET.values()), ["SECURITY_DES"])
    bad = {e[0] for e in err if len(e) == 2}
    ok = {k: v for k, v in MARKET.items() if v not in bad}
    log(f"market: {len(ok)}/{len(MARKET)} tickers valid; invalid: {sorted(bad)}")
    df, err = b.bdh(list(ok.values()), ["PX_LAST"], start, end)
    report("market history", err)
    inv = {v: k for k, v in ok.items()}
    df["name"] = df["security"].map(inv)
    df.to_parquet(OUT / "market_extra.parquet")
    cov = df.groupby("name")["date"].agg(["min", "max", "size"])
    log("market coverage:\n" + cov.to_string())


def block_discover(b, start, end):
    found = {}
    for q in DISCOVER_QUERIES:
        req = b.ins.createRequest("instrumentListRequest")
        req.set("query", q); req.set("maxResults", 40)
        for msg in b._send(req):
            for r in msg.get("results", []):
                sec, desc = r.get("security", ""), r.get("description", "")
                if "<index>" in sec and any(k in (sec + desc).lower() for k in DISCOVER_KEYWORDS):
                    found[sec.replace("<index>", " Index")] = desc
    log(f"discover: {len(found)} candidate index series")
    df, err = b.bdh(list(found), ["PX_LAST"], start, end)
    good = df.groupby("security")["date"].agg(["min", "max", "size"]) if not df.empty else pd.DataFrame()
    if not df.empty:
        df["description"] = df["security"].map(found)
        df.to_parquet(OUT / "discovered.parquet")
    log(f"discover: {len(good)} with history ({len(err)} invalid / not entitled)")
    if len(good):
        good["description"] = good.index.map(found)
        log("discovered coverage:\n" + good.sort_values("size", ascending=False).to_string())


BLOCKS = {"market": block_market, "discover": block_discover, "oat_quotes": block_oat_quotes,
          "universes": block_universes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(BLOCKS))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    start = pd.Timestamp(cfg["start_date"]).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    OUT.mkdir(parents=True, exist_ok=True)
    b = BBG()
    for name in args.only.split(","):
        log(f"=== {name} ===")
        try:
            BLOCKS[name](b, start, end)
        except Exception as e:  # keep going with the other blocks
            log(f"{name} FAILED: {type(e).__name__}: {e}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
