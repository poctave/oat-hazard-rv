"""Fixed-coupon OAT cash flows, accrued interest and yields.

OAT conventions: annual coupon, ACT/ACT ICMA accrued, settlement T+2.
Simplifications (see README): coupon dates are maturity anniversaries (irregular
first coupons ignored), business-day adjustment ignored for cash-flow timing,
holidays ignored for settlement (weekends only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def settle_date(trade, lag: int = 2) -> pd.Timestamp:
    return (pd.Timestamp(trade) + pd.offsets.BDay(lag)).normalize()


def _anniv(mat: pd.Timestamp, k: int) -> pd.Timestamp:
    y = mat.year - k
    day = min(mat.day, 28) if mat.month == 2 else mat.day
    return pd.Timestamp(year=y, month=mat.month, day=day)


@dataclass
class Schedule:
    t: np.ndarray        # ACT/365 years from settle (for discounting)
    tt: np.ndarray       # ICMA periods from settle (for yields)
    amt: np.ndarray      # per 100 face, principal in last flow
    accrued: float
    ttm: float


@dataclass(eq=False)
class Bond:
    isin: str
    coupon: float          # decimal, e.g. 0.04
    maturity: pd.Timestamp
    issue: pd.Timestamp | None = None
    name: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    def __hash__(self):
        return hash(self.isin)

    def schedule(self, settle: pd.Timestamp) -> Schedule:
        settle = pd.Timestamp(settle)
        s = self._cache.get(settle)
        if s is not None:
            return s
        mat = pd.Timestamp(self.maturity)
        n = mat.year - settle.year + 1
        dates = [d for d in (_anniv(mat, k) for k in range(n, -1, -1)) if d > settle]
        if not dates:  # matured
            return Schedule(np.array([]), np.array([]), np.array([]), 0.0, (mat - settle).days / 365.0)
        days = np.array([(d - settle).days for d in dates], float)
        nxt = dates[0]
        prv = _anniv(nxt, 1)
        per = (nxt - prv).days
        start = max(prv, pd.Timestamp(self.issue)) if self.issue is not None else prv
        accr = 100.0 * self.coupon * max((settle - start).days, 0) / per
        amt = np.full(len(dates), 100.0 * self.coupon)
        amt[-1] += 100.0
        tt = (nxt - settle).days / per + np.arange(len(dates))
        s = Schedule(days / 365.0, tt, amt, accr, (mat - settle).days / 365.0)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[settle] = s
        return s

    # convenience wrappers
    def accrued(self, settle) -> float:
        return self.schedule(settle).accrued

    def cashflows(self, settle):
        s = self.schedule(settle)
        return s.t, s.amt

    def ttm(self, settle) -> float:
        return self.schedule(settle).ttm

    def price_from_yield(self, y: float, settle) -> float:
        """Dirty price per 100, annual compounding on ICMA periods."""
        s = self.schedule(settle)
        return float(np.sum(s.amt / (1.0 + y) ** s.tt))

    def yield_from_price(self, dirty: float, settle) -> float:
        s = self.schedule(settle)
        f = lambda y: float(np.sum(s.amt / (1.0 + y) ** s.tt)) - dirty
        return brentq(f, -0.5, 1.5, xtol=1e-12)

    def mod_duration(self, y: float, settle) -> float:
        s = self.schedule(settle)
        disc = s.amt / (1.0 + y) ** s.tt
        return float(np.sum(s.tt * disc) / np.sum(disc) / (1.0 + y))
