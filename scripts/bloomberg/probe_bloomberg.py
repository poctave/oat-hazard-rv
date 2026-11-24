"""Probe Bloomberg: which candidate tickers resolve, and how to list matured OATs.
Writes probe.log next to the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_bloomberg import BBG  # noqa: E402

out = open(ROOT / "probe.log", "w", encoding="utf-8")


def log(*a):
    print(*a, file=out, flush=True)


b = BBG()
cands = [
    "ESTRON Index", "EONIA Index",
    "EESWE1 Curncy", "EESWE10 Curncy", "EESWEC Curncy", "EESWE1Z Curncy",
    "EUSWE1 Curncy", "EUSWE10 Curncy", "EUSWEC Curncy",
    "FRTR CDS USD SR 5Y D14 Corp", "FRTR CDS EUR SR 5Y D14 Corp", "FRANCE CDS USD SR 5Y D14 Corp",
    "CFRTR1U5 Curncy", "FRGV5YUSAC=R",
    "RPFRGC Index", "RFRFRGC Index", "RFRFR Index", "EUREPO1W Index", "RPEGFR1W Index",
    "GDBR10 Index", "GFRN10 Index",
]
df, err = b.bdp(cands, ["SECURITY_DES", "NAME", "PX_LAST", "LAST_UPDATE_DT"])
log("== candidates ==")
log(df.to_string())
log("errors:", err)

for q in ["France repo", "RepoFunds France", "France GC", "France CDS", "ESTR OIS", "euro short term rate swap"]:
    req = b.ins.createRequest("instrumentListRequest")
    req.set("query", q)
    req.set("maxResults", 25)
    log(f"\n== instrumentList '{q}' ==")
    for msg in b._send(req):
        for r in msg.get("results", []):
            log("  ", r.get("security"), "|", r.get("description"))

log("\n== govtList FRTR ==")
keys = b.govt_list("FRTR")
log(len(keys), keys[:10])
st, e = b.bdp(keys, ["ID_ISIN", "MATURITY", "CPN", "CPN_TYP"])
log(st.sort_values("MATURITY").head(10).to_string() if not st.empty else "empty")

log("\n== govtList partial FRTR ==")
req = b.ins.createRequest("govtListRequest")
req.set("ticker", "FRTR"); req.set("partialMatch", True); req.set("maxResults", 2000)
n = 0
for msg in b._send(req):
    n += len(msg.get("results", []))
log(n)

log("\n== matured lookup test ==")
test = ["FR0010163543 Govt", "FR0000187361 Govt", "FR0010061242 Govt"]  # OATs matured 2015, 2010, 2020 (my recollection)
t, e = b.bdp(test, ["ID_ISIN", "SECURITY_NAME", "MATURITY", "CPN"])
log(t.to_string()); log(e)

log("\n== bulk fields that may list members ==")
for sec, fld in [("FRTR Govt", "GOVT_BOND_LIST"), ("GFRN10 Index", "CURVE_MEMBERS"),
                 ("YCGT0014 Index", "CURVE_MEMBERS"), ("YCGT0014 Index", "CURVE_TENOR_RATES")]:
    t, e = b.bdp([sec], [fld])
    log(sec, fld, "->", (str(t.iloc[0].get(fld))[:800] if not t.empty else "empty"), e)
out.close()
print("probe done")
