"""Try alternative French GC repo tickers and append the first that has history to market.parquet."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_bloomberg import BBG  # noqa: E402

CANDS = {"repo_gc_fr_oat_on": "RB00049V Mmkt", "repo_gc_fr_on": "RB00048Y Mmkt",
         "repo_gc_fr_oat_sn": "RB00003E Mmkt", "repo_gc_fr_nolink_tn": "RB00003K Mmkt",
         "repo_rfr_france_gc": "REFRGCFR Index", "repo_gc_fr_s3m": "RB0006UG Mmkt"}
b = BBG()
log = open(ROOT / "repo.log", "w")
df, err = b.bdh(list(CANDS.values()), ["PX_LAST", "PX_MID", "YLD_YTM_MID"], "20100101", pd.Timestamp.today().strftime("%Y%m%d"))
print("errors:", err, file=log)
if not df.empty:
    inv = {v: k for k, v in CANDS.items()}
    df["name"] = df["security"].map(inv)
    print(df.groupby("security").agg(first=("date", "min"), last=("date", "max"), n=("date", "size")).to_string(), file=log)
    print(df.groupby("security")[[c for c in ["PX_LAST", "PX_MID"] if c in df]].describe().T.to_string(), file=log)
    df.to_parquet(ROOT / "data" / "raw" / "repo_candidates.parquet")
log.close()
print("repo done")
