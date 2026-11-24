import numpy as np
import pandas as pd
import pytest

from oathazard.bonds import Bond
from oathazard.curves import DiscountCurve, bootstrap_ois, ACT360
from oathazard.hazard import CrossSection, HazardCurve, cds_par_spread, fit_hazard

SETTLE = pd.Timestamp("2026-11-26")
FLAT = DiscountCurve.flat(np.log(1.025))  # flat €STR 2.5% annual, as in the PDF


def toy_bonds():
    return [Bond("5Y1", 0.01, pd.Timestamp("2031-11-25")), Bond("5Y4", 0.04, pd.Timestamp("2031-05-25")),
            Bond("10Y1", 0.01, pd.Timestamp("2036-11-25")), Bond("10Y45", 0.045, pd.Timestamp("2036-04-25")),
            Bond("20Y3", 0.03, pd.Timestamp("2046-05-25"))]


def priced(bonds, lam, rec=0.4, curve=FLAT):
    hc = HazardCurve((0.0,), np.array([lam]))
    cs = CrossSection(bonds, np.full(len(bonds), 100.0), SETTLE, curve)
    return cs.model_prices(hc, rec)


def test_zero_hazard_is_riskfree():
    b = toy_bonds()[0]
    t, a = b.cashflows(SETTLE)
    assert priced([b], 0.0)[0] == pytest.approx(np.sum(a * FLAT.df(t)), rel=1e-12)


def test_fit_recovers_flat_lambda_and_flags_perturbed_bonds():
    bonds = toy_bonds()
    p = priced(bonds, 0.02)
    p[0] -= 0.2   # 5Y 1% made cheap
    p[2] += 0.4   # 10Y 1% made rich
    cs = CrossSection(bonds, p, SETTLE, FLAT)
    r = fit_hazard(cs, 0.4, knots=(0.0,))
    assert r.hazard.lambdas[0] == pytest.approx(0.02, abs=5e-4)
    assert r.resid_bp[0] > 2 and r.resid_bp[2] < -2          # cheap / rich flagged
    assert abs(r.resid_bp[3]) < 2                             # high coupon 10Y not flagged


def test_credit_triangle():
    hc = HazardCurve((0.0,), np.array([0.02]))
    assert cds_par_spread(hc, FLAT, 0.4, 5.0) == pytest.approx(0.02 * 0.6, rel=0.02)


def test_zspread_coupon_bias_under_stress():
    """Same credit, different coupons: Z-spread says the high coupon is cheaper."""
    bonds = toy_bonds()[2:4]
    p = priced(bonds, 0.10)
    z = CrossSection(bonds, p, SETTLE, FLAT).zspreads()
    assert (z[1] - z[0]) * 1e4 > 50


def test_ois_bootstrap_reprices_par_swaps():
    pts = {0.5: 0.02, 1.0: 0.021, 2.0: 0.022, 5.0: 0.024, 10.0: 0.026, 30.0: 0.027}
    c = bootstrap_ois(pts)
    for T in (2, 5, 10, 30):
        ann = sum(ACT360 * c.df(k) for k in range(1, T + 1))
        par = (1 - c.df(T)) / ann
        assert par == pytest.approx(pts[float(T)], abs=1e-9)


def test_yield_roundtrip():
    b = Bond("X", 0.0325, pd.Timestamp("2035-05-25"))
    p = b.price_from_yield(0.031, SETTLE)
    assert b.yield_from_price(p, SETTLE) == pytest.approx(0.031, abs=1e-10)
