"""Risk-free discount curve: €STR (or EONIA) OIS bootstrap.

Conventions (kept simple on purpose, documented in README):
- time t in years = ACT/365 from the valuation date
- OIS accrual = ACT/360 (alpha = t * 365/360)
- swaps <= 1Y: single payment at maturity
- swaps > 1Y: annual fixed vs compounded overnight, par condition
      S * sum_i alpha_i D(t_i) + D(T) = 1
- interpolation: linear in log discount factor (piecewise flat forwards)
- extrapolation beyond the last pillar: flat last forward
"""
from __future__ import annotations

import numpy as np

ACT360 = 365.0 / 360.0


class DiscountCurve:
    def __init__(self, times: np.ndarray, dfs: np.ndarray):
        t = np.asarray(times, float)
        d = np.asarray(dfs, float)
        order = np.argsort(t)
        t, d = t[order], d[order]
        if t[0] > 0:
            t = np.r_[0.0, t]
            d = np.r_[1.0, d]
        self.times = t
        self.log_dfs = np.log(d)

    def df(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        out = np.interp(t, self.times, self.log_dfs)
        # flat forward extrapolation beyond last pillar
        last_fwd = -(self.log_dfs[-1] - self.log_dfs[-2]) / (self.times[-1] - self.times[-2])
        beyond = t > self.times[-1]
        if np.any(beyond):
            out = np.where(beyond, self.log_dfs[-1] - last_fwd * (t - self.times[-1]), out)
        return np.exp(out)

    def zero(self, t) -> np.ndarray:
        """Continuously compounded zero rate."""
        t = np.maximum(np.asarray(t, float), 1e-8)
        return -np.log(self.df(t)) / t

    def shifted(self, spread: float) -> "DiscountCurve":
        """Parallel shift of cc zero rates (e.g. funding spread GC - €STR)."""
        dfs = np.exp(self.log_dfs) * np.exp(-spread * self.times)
        return DiscountCurve(self.times.copy(), dfs)

    @classmethod
    def flat(cls, rate_cc: float, max_t: float = 60.0) -> "DiscountCurve":
        t = np.linspace(0, max_t, 121)
        return cls(t, np.exp(-rate_cc * t))


def bootstrap_ois(tenors_years: dict[float, float]) -> DiscountCurve:
    """Bootstrap from OIS par rates.

    tenors_years: {maturity_in_years: par_rate_in_decimal}. Needs at least one
    point; missing annual pillars above 1Y are filled by linear interpolation of
    par rates (standard practical shortcut).
    """
    items = sorted((float(k), float(v)) for k, v in tenors_years.items() if np.isfinite(v))
    if not items:
        raise ValueError("no OIS rates")
    times, dfs = [], []
    short = [(t, s) for t, s in items if t <= 1.0 + 1e-9]
    long_ = [(t, s) for t, s in items if t > 1.0 + 1e-9]
    for t, s in short:
        times.append(t)
        dfs.append(1.0 / (1.0 + s * t * ACT360))
    if long_:
        # annual grid of par rates from 1Y to last tenor
        known_t = np.array([t for t, _ in short if abs(t - 1.0) < 1e-9] + [t for t, _ in long_])
        known_s = np.array([s for t, s in short if abs(t - 1.0) < 1e-9] + [s for _, s in long_])
        last = int(round(long_[-1][0]))
        grid = np.arange(1, last + 1, dtype=float)
        par = np.interp(grid, known_t, known_s)
        annual_df: dict[int, float] = {}
        for i, (t, s) in enumerate(zip(grid, par)):
            n = int(t)
            if n == 1:
                d1 = 1.0 / (1.0 + s * ACT360)
                annual_df[1] = d1
                continue
            annuity = sum(ACT360 * annual_df[k] for k in range(1, n))
            annual_df[n] = (1.0 - s * annuity) / (1.0 + s * ACT360)
        for n, d in annual_df.items():
            if n == 1 and any(abs(t - 1.0) < 1e-9 for t in times):
                continue
            times.append(float(n))
            dfs.append(d)
    return DiscountCurve(np.array(times), np.array(dfs))
