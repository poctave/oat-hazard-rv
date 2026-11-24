"""Hazard-rate (reduced-form) pricing of OATs and the global cross-sectional fit.

Price per 100 (dirty), recovery of face value R paid at default:

    P = sum_i CF_i D(t_i) Q(t_i)                      (coupons + principal, paid if alive)
      + 100 R sum_k D(m_k) [Q(m_{k-1}) - Q(m_k)]      (recovery leg, monthly grid)

Q(t) = exp(-H(t)), H(t) = integral of a piecewise-constant hazard lambda(t).
D(t) = risk-free (€STR OIS) discount factor, optionally shifted by a funding spread.
Default assumed independent of rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from .bonds import Bond
from .curves import DiscountCurve

DEFAULT_KNOTS = (0.0, 3.0, 7.0)  # lambda buckets: 0-3Y, 3-7Y, 7Y+
REC_GRID_STEP = 1.0 / 12.0


@dataclass
class HazardCurve:
    knots: tuple = DEFAULT_KNOTS          # left edges of buckets
    lambdas: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01]))

    def cum_hazard(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        k = np.r_[np.asarray(self.knots, float), np.inf]
        H = np.zeros_like(t)
        for j, lam in enumerate(self.lambdas):
            H += lam * np.clip(t - k[j], 0.0, k[j + 1] - k[j])
        return H

    def survival(self, t) -> np.ndarray:
        return np.exp(-self.cum_hazard(t))


class CrossSection:
    """All bonds on one date, with cash flows pre-computed for fast repricing."""

    def __init__(self, bonds: list[Bond], dirty: np.ndarray, settle, curve: DiscountCurve):
        self.bonds = bonds
        self.dirty = np.asarray(dirty, float)
        self.settle = settle
        self.curve = curve
        cf_t, cf_a, cf_i = [], [], []
        rg_t0, rg_t1, rg_i = [], [], []
        self.ttm = np.zeros(len(bonds))
        self.ytm = np.zeros(len(bonds))
        self.mdur = np.zeros(len(bonds))
        for i, b in enumerate(bonds):
            t, a = b.cashflows(settle)
            cf_t.append(t); cf_a.append(a); cf_i.append(np.full(len(t), i))
            T = b.ttm(settle)
            self.ttm[i] = T
            g = np.r_[np.arange(0.0, T, REC_GRID_STEP), T]
            g = np.unique(g)
            rg_t0.append(g[:-1]); rg_t1.append(g[1:]); rg_i.append(np.full(len(g) - 1, i))
            y = b.yield_from_price(self.dirty[i], settle)
            self.ytm[i] = y
            self.mdur[i] = b.mod_duration(y, settle)
        self.cf_t = np.concatenate(cf_t); self.cf_a = np.concatenate(cf_a); self.cf_i = np.concatenate(cf_i)
        self.cf_D = curve.df(self.cf_t)
        self.rg_t0 = np.concatenate(rg_t0); self.rg_t1 = np.concatenate(rg_t1); self.rg_i = np.concatenate(rg_i)
        self.rg_D = curve.df(0.5 * (self.rg_t0 + self.rg_t1))
        self.n = len(bonds)

    # ---------- hazard pricing ----------
    def model_prices(self, hc: HazardCurve, recovery: float) -> np.ndarray:
        promised = np.bincount(self.cf_i, self.cf_a * self.cf_D * hc.survival(self.cf_t), self.n)
        dQ = hc.survival(self.rg_t0) - hc.survival(self.rg_t1)
        rec = np.bincount(self.rg_i, 100.0 * recovery * self.rg_D * dQ, self.n)
        return promised + rec

    # ---------- Z-spread ----------
    def zspreads(self) -> np.ndarray:
        out = np.empty(self.n)
        for i in range(self.n):
            m = self.cf_i == i
            t, a, D = self.cf_t[m], self.cf_a[m], self.cf_D[m]
            f = lambda z: np.sum(a * D * np.exp(-z * t)) - self.dirty[i]
            out[i] = brentq(f, -0.2, 1.0, xtol=1e-12)
        return out


@dataclass
class FitResult:
    hazard: HazardCurve
    recovery: float
    resid_bp: np.ndarray        # market yield - model yield, bp (>0 = cheap)
    rmse_bp: float
    success: bool
    breakdown: bool             # True if fit wants lambda < 0 (survival > 100%)


def fit_hazard(cs: CrossSection, recovery: float | None = 0.40, knots=DEFAULT_KNOTS,
               allow_negative: bool = True, x0=None, rec_bounds=(0.0, 0.6)) -> FitResult:
    """Global fit of one piecewise-constant hazard curve to all bonds.

    Objective: duration-scaled price errors = approx. yield errors, so long bonds
    do not dominate. allow_negative=True lets the fit go below zero so the QE
    breakdown (lambda < 0) is visible instead of being hidden by a bound.
    """
    k = len(knots)
    fit_r = recovery is None
    x0 = np.full(k, 0.005) if x0 is None else np.asarray(x0, float)[:k]
    if len(x0) < k:
        x0 = np.r_[x0, np.full(k - len(x0), 0.005)]
    lb = -0.05 if allow_negative else 0.0
    lo, hi = np.full(k, lb), np.full(k, 1.0)
    if fit_r:
        x0 = np.r_[x0, 0.3]; lo = np.r_[lo, rec_bounds[0]]; hi = np.r_[hi, rec_bounds[1]]

    def resid(x):
        hc = HazardCurve(tuple(knots), x[:k])
        pm = cs.model_prices(hc, x[k] if fit_r else recovery)
        return (pm - cs.dirty) / (cs.dirty * cs.mdur) * 1e4  # bp

    sol = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    if fit_r:
        recovery = float(sol.x[k])
    hc = HazardCurve(tuple(knots), sol.x[:k])
    # exact yield residuals
    pm = cs.model_prices(hc, recovery)
    y_model = np.array([b.yield_from_price(p, cs.settle) for b, p in zip(cs.bonds, pm)])
    r = (cs.ytm - y_model) * 1e4
    return FitResult(hc, recovery, r, float(np.sqrt(np.mean(r ** 2))), bool(sol.success),
                     bool(np.any(sol.x[:k] < 0)))


def zspread_rv(cs: CrossSection, degree: int = 3):
    """Baseline signal: Z-spread minus a polynomial fit in maturity (bp, >0 = cheap)."""
    z = cs.zspreads()
    deg = min(degree, max(cs.n - 2, 0))
    coef = np.polyfit(cs.ttm, z, deg)
    return z * 1e4, (z - np.polyval(coef, cs.ttm)) * 1e4


def cds_par_spread(hc: HazardCurve, curve: DiscountCurve, recovery: float, T: float) -> float:
    """Par CDS spread (decimal) implied by the hazard curve: quarterly premium,
    ACT/360, accrual-on-default approximated by half a period."""
    pay = np.arange(0.25, T + 1e-9, 0.25)
    prev = np.r_[0.0, pay[:-1]]
    alpha = 0.25 * 365.0 / 360.0
    Qp, Q0 = hc.survival(pay), hc.survival(prev)
    rpv01 = np.sum(alpha * curve.df(pay) * (Qp + 0.5 * (Q0 - Qp)))
    g = np.arange(0.0, T + 1e-9, REC_GRID_STEP)
    prot = (1 - recovery) * np.sum(curve.df(0.5 * (g[:-1] + g[1:])) * (hc.survival(g[:-1]) - hc.survival(g[1:])))
    return prot / rpv01
