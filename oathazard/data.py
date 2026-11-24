"""Load the raw Bloomberg pull (data/raw/*.parquet) into model inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bonds import Bond
from .curves import DiscountCurve, bootstrap_ois

TENOR_LETTERS = {"A": 1 / 12, "B": 2 / 12, "C": 3 / 12, "D": 4 / 12, "E": 5 / 12, "F": 6 / 12,
                 "G": 7 / 12, "H": 8 / 12, "I": 9 / 12, "J": 10 / 12, "K": 11 / 12}


def tenor_years(code: str) -> float:
    return TENOR_LETTERS[code] if code in TENOR_LETTERS else float(code)


class RawData:
    def __init__(self, raw_dir: Path, cfg: dict):
        self.cfg = cfg
        st = pd.read_parquet(raw_dir / "oat_static.parquet")
        self.static = pd.DataFrame({
            "isin": st["ID_ISIN"].values,
            "coupon": st["CPN"].astype(float).values / 100.0,
            "maturity": pd.to_datetime(st["MATURITY"]).values,
            "issue": pd.to_datetime(st["ISSUE_DT"]).values,
            "name": st.get("SECURITY_NAME", st["ID_ISIN"]).values,
            "amt_out": pd.to_numeric(st.get("AMT_OUTSTANDING", np.nan), errors="coerce"),
        }).set_index("isin")
        px = pd.read_parquet(raw_dir / "oat_prices.parquet")
        self.clean = px.pivot_table(index="date", columns="isin", values="PX_LAST").sort_index()
        self.bbg_yield = (px.pivot_table(index="date", columns="isin", values="YLD_YTM_MID").sort_index()
                          if "YLD_YTM_MID" in px else None)
        md = pd.read_parquet(raw_dir / "market.parquet")[["date", "name", "PX_LAST"]]
        extra = raw_dir / "extra"
        parts = [md]
        if (extra / "market_extra.parquet").exists():
            parts.append(pd.read_parquet(extra / "market_extra.parquet")[["date", "name", "PX_LAST"]])
        if (extra / "discovered.parquet").exists():
            dsc = pd.read_parquet(extra / "discovered.parquet")
            # euro GC repo proxy: STOXX GC Pooling EUR overnight (ECB basket), daily since 2010
            gc = dsc[dsc["security"] == "SGCPON Index"][["date", "PX_LAST"]].assign(name="repo_gc_pooling_on")
            parts.append(gc)
            parts.append(dsc[["date", "security", "PX_LAST"]].rename(columns={"security": "name"}))
        md = pd.concat(parts, ignore_index=True)
        self.mkt = md.pivot_table(index="date", columns="name", values="PX_LAST").sort_index()
        self.bonds = {i: Bond(i, r.coupon, pd.Timestamp(r.maturity), pd.Timestamp(r.issue), str(r.name))
                      for i, r in self.static.iterrows()}

    # ---------------------------------------------------------------- curves
    def _row(self, date) -> pd.Series:
        m = self.mkt.loc[:date].ffill(limit=5)
        return m.iloc[-1] if len(m) else pd.Series(dtype=float)

    def ois_curve(self, date) -> tuple[DiscountCurve, str]:
        r = self._row(date)
        rc = self.cfg["rates"]
        for prefix, shift, label in (("estr_ois_", 0.0, "ESTR"),
                                     ("eonia_ois_", -rc["eonia_estr_spread"], "EONIA-8.5bp")):
            pts = {}
            for t in rc["ois_tenors"]:
                v = r.get(prefix + t, np.nan)
                if np.isfinite(v):
                    pts[tenor_years(t)] = v / 100.0 + shift
            if len(pts) >= 6 and max(pts) >= 10:
                return bootstrap_ois(pts), label
        raise ValueError(f"no usable OIS curve on {date:%Y-%m-%d}")

    def bund_spread_fn(self, date, curve: DiscountCurve):
        """Bund generic yield minus €STR OIS zero (cc), interpolated in maturity, flat outside 2-30Y.
        Used to move the risk-free curve to Germany, so lambda measures France-vs-Germany credit."""
        r = self._row(date)
        T, s = [], []
        for name, t in (("GDBR2", 2.0), ("GDBR5", 5.0), ("GDBR10", 10.0), ("GDBR30", 30.0)):
            v = r.get(name, np.nan)
            if np.isfinite(v):
                T.append(t); s.append(np.log1p(v / 100.0) - float(curve.zero(t)))
        if len(T) < 2:
            return None
        T, s = np.array(T), np.array(s)
        return lambda t: np.interp(np.asarray(t, float), T, s)

    def overnight(self, date) -> float:
        r = self._row(date)
        v = r.get("estr_fixing", np.nan)
        if not np.isfinite(v):
            v = r.get("eonia_fixing", np.nan) - self.cfg["rates"]["eonia_estr_spread"] * 100
        return v / 100.0

    def funding_spread(self, date, source: str = "gc") -> float:
        """Funding rate minus €STR (decimal). source='gc': GC repo series (GC Pooling ON if present);
        source='btf3m': French 3M BTF bill yield (collateral-scarcity proxy). NaN if unavailable."""
        r = self._row(date)
        if source == "btf3m":
            v = r.get("GBTF3MO Index", np.nan)
            return v / 100.0 - self.overnight(date) if np.isfinite(v) else np.nan
        cols = sorted((c for c in r.index if str(c).startswith("repo_")), key=lambda c: "_gc" not in str(c))
        vals = [r[c] for c in cols if np.isfinite(r[c])]
        if not vals:
            return np.nan
        return vals[0] / 100.0 - self.overnight(date)

    def market_cds(self, date) -> dict[int, float]:
        r = self._row(date)
        import re
        return {int(m.group(1)): r[c] for c in r.index
                if (m := re.fullmatch(r"cds_(\d+)y", str(c))) and np.isfinite(r[c])}

    # ---------------------------------------------------------------- bonds
    def cross_section(self, date, settle):
        u = self.cfg["universe"]
        if date not in self.clean.index:
            return [], np.array([])
        row = self.clean.loc[date].dropna()
        bonds, dirty = [], []
        for isin, clean in row.items():
            b = self.bonds.get(isin)
            if b is None:
                continue
            ttm = b.ttm(settle)
            if not (u["min_ttm_years"] <= ttm <= u["max_ttm_years"]):
                continue
            if b.issue is not None and (settle - b.issue).days < u["min_age_days"]:
                continue
            bonds.append(b)
            dirty.append(clean + b.accrued(settle))
        return bonds, np.array(dirty)
