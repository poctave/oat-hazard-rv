"""Pull everything the project needs from a logged-in Bloomberg Terminal (Desktop API).

Run on the machine where the Terminal is open:

    python scripts/fetch_bloomberg.py --check    # 1 min: checks every ticker resolves
    python scripts/fetch_bloomberg.py            # full pull into data/raw/

Requires blpapi:
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    import blpapi
except ImportError:  # pragma: no cover
    sys.exit("blpapi not installed, see the docstring at the top of this file.")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

STATIC_FIELDS = [
    "ID_ISIN", "SECURITY_NAME", "SECURITY_TYP", "CPN", "CPN_FREQ", "CPN_TYP", "MATURITY",
    "ISSUE_DT", "FIRST_CPN_DT", "DAY_CNT_DES", "INFLATION_LINKED_INDICATOR", "MTY_TYP",
    "AMT_ISSUED", "AMT_OUTSTANDING",
]
PRICE_FIELDS = ["PX_LAST", "YLD_YTM_MID"]


# --------------------------------------------------------------------------- session
class BBG:
    def __init__(self, host="localhost", port=8194):
        opts = blpapi.SessionOptions()
        opts.setServerHost(host)
        opts.setServerPort(port)
        self.session = blpapi.Session(opts)
        if not self.session.start():
            sys.exit("Could not start a Bloomberg session: is the Terminal open and logged in?")
        for svc in ("//blp/refdata", "//blp/instruments"):
            if not self.session.openService(svc):
                sys.exit(f"Could not open {svc}")
        self.ref = self.session.getService("//blp/refdata")
        self.ins = self.session.getService("//blp/instruments")

    def _send(self, req) -> list[dict]:
        self.session.sendRequest(req)
        out = []
        while True:
            ev = self.session.nextEvent(60_000)
            for msg in ev:
                if ev.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                    out.append(msg.toPy())
            if ev.eventType() == blpapi.Event.RESPONSE:
                return out
            if ev.eventType() == blpapi.Event.TIMEOUT:
                raise TimeoutError("Bloomberg request timed out")

    # reference data (static + bulk), one row per security
    def bdp(self, securities: list[str], fields: list[str]) -> tuple[pd.DataFrame, list]:
        rows, errors = [], []
        for chunk in _chunks(securities, 100):
            req = self.ref.createRequest("ReferenceDataRequest")
            for s in chunk:
                req.append("securities", s)
            for f in fields:
                req.append("fields", f)
            for msg in self._send(req):
                for sd in msg.get("securityData", []):
                    if "securityError" in sd:
                        errors.append((sd["security"], sd["securityError"].get("message")))
                        continue
                    for fe in sd.get("fieldExceptions", []):
                        errors.append((sd["security"], fe.get("fieldId"), fe.get("errorInfo", {}).get("message")))
                    rows.append({"security": sd["security"], **sd.get("fieldData", {})})
        return pd.DataFrame(rows), errors

    # historical data, long format (date, security, field...)
    def bdh(self, securities: list[str], fields: list[str], start: str, end: str) -> tuple[pd.DataFrame, list]:
        rows, errors = [], []
        for chunk in _chunks(securities, 25):
            req = self.ref.createRequest("HistoricalDataRequest")
            for s in chunk:
                req.append("securities", s)
            for f in fields:
                req.append("fields", f)
            req.set("startDate", start)
            req.set("endDate", end)
            req.set("periodicitySelection", "DAILY")
            for msg in self._send(req):
                sd = msg.get("securityData")
                if not sd:
                    continue
                if "securityError" in sd:
                    errors.append((sd["security"], sd["securityError"].get("message")))
                    continue
                for fe in sd.get("fieldExceptions", []):
                    errors.append((sd["security"], fe.get("fieldId"), fe.get("errorInfo", {}).get("message")))
                for fd in sd.get("fieldData", []):
                    rows.append({"security": sd["security"], **fd})
            print(f"  bdh: {len(rows):>9,} rows so far", flush=True)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df, errors

    def govt_list(self, ticker: str) -> list[str]:
        req = self.ins.createRequest("govtListRequest")
        req.set("ticker", ticker)
        req.set("partialMatch", False)
        req.set("maxResults", 2000)
        keys = []
        for msg in self._send(req):
            for r in msg.get("results", []):
                keys.append(r.get("parseky"))
        return [k for k in keys if k]


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _report(label, errors):
    if errors:
        print(f"  {label}: {len(errors)} problem(s)")
        for e in errors[:15]:
            print("    ", e)
        if len(errors) > 15:
            print("     ...")


# --------------------------------------------------------------------------- tickers
def ois_tickers(cfg, prefix):
    r = cfg["rates"]
    return {t: f"{prefix}{t}{r['ois_suffix']}" for t in r["ois_tenors"]}


def market_tickers(cfg) -> dict[str, str]:
    out = {
        "estr_fixing": cfg["rates"]["estr_fixing"],
        "eonia_fixing": cfg["rates"]["eonia_fixing"],
    }
    for t, tk in ois_tickers(cfg, cfg["rates"]["estr_ois_prefix"]).items():
        out[f"estr_ois_{t}"] = tk
    for t, tk in ois_tickers(cfg, cfg["rates"]["eonia_ois_prefix"]).items():
        out[f"eonia_ois_{t}"] = tk
    for y, tk in cfg["cds"]["tickers"].items():
        out[f"cds_{y}y"] = tk
    for name, tk in cfg["repo"]["gc_tickers"].items():
        out[f"repo_{name}"] = tk
    for tk in cfg["bunds"]["generic_yields"] + cfg["bunds"]["oat_generic_yields"]:
        out[tk.split()[0]] = tk
    return out


def bond_security(isin: str, src: str | None) -> str:
    return f"{isin}@{src} Govt" if src else f"{isin} Govt"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check that tickers resolve")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    start = pd.Timestamp(cfg["start_date"]).strftime("%Y%m%d")
    end = pd.Timestamp(cfg["end_date"] or dt.date.today()).strftime("%Y%m%d")
    RAW.mkdir(parents=True, exist_ok=True)
    bbg = BBG()

    mkt = market_tickers(cfg)
    if args.check:
        df, err = bbg.bdp(list(mkt.values()), ["SECURITY_DES", "PX_LAST"])
        ok = set(df["security"]) if not df.empty else set()
        bad = {e[0] for e in err if len(e) == 2}
        print("\nTicker check:")
        for k, tk in mkt.items():
            status = "OK " if tk in ok and tk not in bad else "BAD"
            print(f"  {status} {k:<22} {tk}")
        live = bbg.govt_list(cfg["universe"]["issuer_ticker"])
        print(f"\ngovtList {cfg['universe']['issuer_ticker']}: {len(live)} live securities")
        extra = ROOT / cfg["universe"]["extra_isin_file"]
        print(f"extra ISIN file: {'found' if extra.exists() else 'MISSING (only live bonds will be pulled)'}")
        return

    # 1) universe: live bonds from govtList + matured ISINs from the SRCH export
    print("1/5 universe")
    live = bbg.govt_list(cfg["universe"]["issuer_ticker"])
    st_live, err = bbg.bdp(live, ["ID_ISIN"])
    isins = set(st_live.get("ID_ISIN", pd.Series(dtype=str)).dropna())
    extra = ROOT / cfg["universe"]["extra_isin_file"]
    if extra.exists():
        ex = pd.read_csv(extra)
        col = next(c for c in ex.columns if "isin" in c.lower())
        isins |= set(ex[col].dropna().str.strip())
    isins = sorted(i for i in isins if isinstance(i, str) and i.startswith("FR"))
    print(f"  {len(isins)} ISINs")

    # 2) static data, keep nominal fixed-coupon bonds
    print("2/5 static")
    secs = [bond_security(i, None) for i in isins]
    static, err = bbg.bdp(secs, STATIC_FIELDS)
    _report("static", err)
    static.to_parquet(RAW / "oat_static_all.parquet")
    keep = static[(static["CPN_TYP"].astype(str).str.upper().str.startswith("FIXED"))
                  & (static["INFLATION_LINKED_INDICATOR"].astype(str).str.upper() != "Y")
                  & (static["CPN_FREQ"].astype(float) == 1)]
    keep.to_parquet(RAW / "oat_static.parquet")
    print(f"  kept {len(keep)} fixed-coupon nominal annual bonds out of {len(static)}")

    # bulk amount outstanding history (tap issuance / buybacks)
    hist_rows = []
    for s in keep["security"]:
        req_df, e = bbg.bdp([s], ["AMOUNT_OUTSTANDING_HISTORY"])
        if not req_df.empty and isinstance(req_df.iloc[0].get("AMOUNT_OUTSTANDING_HISTORY"), list):
            for r in req_df.iloc[0]["AMOUNT_OUTSTANDING_HISTORY"]:
                hist_rows.append({"security": s, **r})
    if hist_rows:
        pd.DataFrame(hist_rows).to_parquet(RAW / "oat_amt_out_history.parquet")
    print(f"  amount outstanding history: {len(hist_rows)} rows")

    # 3) daily prices
    print("3/5 bond prices (slow: ~10-20 min)")
    src = cfg["universe"]["pricing_source"]
    isin_of = dict(zip(keep["security"], keep["ID_ISIN"]))
    px_secs = {bond_security(isin_of[s], src): isin_of[s] for s in keep["security"]}
    px, err = bbg.bdh(list(px_secs), PRICE_FIELDS, start, end)
    _report("prices", err)
    px["isin"] = px["security"].map(px_secs)
    px.to_parquet(RAW / "oat_prices.parquet")

    # 4) rates, CDS, repo, Bunds
    print("4/5 market data")
    inv = {v: k for k, v in mkt.items()}
    md, err = bbg.bdh(list(mkt.values()), ["PX_LAST"], start, end)
    _report("market", err)
    md["name"] = md["security"].map(inv)
    md.to_parquet(RAW / "market.parquet")

    print("5/5 done ->", RAW)


if __name__ == "__main__":
    main()
