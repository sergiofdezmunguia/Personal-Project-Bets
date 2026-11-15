"""Fase 4: normalización del margen (proporcional, Shin y potencia), con cuotas inventadas."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from src import features as ft
from src import margin as mg
from src import referencias as rf

ODDS = np.array([[1.50, 4.20, 7.00], [2.40, 3.30, 3.10], [1.18, 7.5, 15.0], [3.9, 3.6, 1.95]])


@pytest.mark.parametrize("method", mg.METHODS)
def test_probabilities_sum_to_one_and_keep_the_order(method):
    p = mg.normalize(ODDS, method)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-12)
    assert (np.argsort(p, axis=1) == np.argsort(-ODDS, axis=1)).all()


def test_shin_matches_an_independent_root_finder():
    p, z = mg.shin(ODDS, return_param=True)
    for row, zr, pr in zip(ODDS, z, p, strict=True):
        q = 1 / row
        B = q.sum()
        f = lambda zz: (np.sqrt(zz**2 + 4 * (1 - zz) * q**2 / B) - zz).sum() / (2 * (1 - zz)) - 1  # noqa: E731
        z0 = brentq(f, 0, 0.5)
        assert zr == pytest.approx(z0, abs=1e-10)
        assert pr == pytest.approx((np.sqrt(z0**2 + 4 * (1 - z0) * q**2 / B) - z0) / (2 * (1 - z0)), abs=1e-9)


def test_power_matches_an_independent_root_finder():
    p, k = mg.power(ODDS, return_param=True)
    for row, kr, pr in zip(ODDS, k, p, strict=True):
        q = 1 / row
        k0 = brentq(lambda kk: (q**kk).sum() - 1, 1, 5)
        assert kr == pytest.approx(k0, abs=1e-10)
        assert pr == pytest.approx(q**k0, abs=1e-9)


@pytest.mark.parametrize("method", ["shin", "potencia"])
def test_margin_moves_from_the_favourite_to_the_longshot(method):
    """Shin y potencia cargan más margen en la sorpresa: el favorito sube frente a la proporcional."""
    prop, other = mg.proportional(ODDS), mg.normalize(ODDS, method)
    fav = np.argmin(ODDS, axis=1)
    longshot = np.argmax(ODDS, axis=1)
    rows = np.arange(len(ODDS))
    assert (other[rows, fav] > prop[rows, fav]).all()
    assert (other[rows, longshot] < prop[rows, longshot]).all()


@pytest.mark.parametrize("method", mg.METHODS)
def test_fair_odds_are_left_alone(method):
    fair = np.array([[2.0, 4.0, 4.0], [3.0, 3.0, 3.0]])
    assert np.allclose(mg.normalize(fair, method), 1 / fair)


@pytest.mark.parametrize("method", mg.METHODS)
def test_missing_or_invalid_odds_give_nan_rows(method):
    p = mg.normalize(np.array([[2.0, np.nan, 3.0], [1.0, 3.0, 3.0], [2.0, 3.5, 3.6]]), method)
    assert np.isnan(p[:2]).all() and np.isfinite(p[2]).all()


def test_shin_z_is_small_and_grows_with_the_margin():
    _, z1 = mg.shin([[2.0, 3.4, 3.8]], return_param=True)
    _, z2 = mg.shin([[1.9, 3.2, 3.6]], return_param=True)
    assert 0 < z1[0] < z2[0] < 0.1


def test_market_probs_and_opening_features_default_to_proportional():
    import pandas as pd

    df = pd.DataFrame(ODDS, columns=["pin_open_home", "pin_open_draw", "pin_open_away"])
    inv = 1 / ODDS
    assert np.allclose(rf.market_probs(df, "pin_open").to_numpy(), inv / inv.sum(1, keepdims=True))
    assert np.allclose(ft.opening_features(df).to_numpy(), inv / inv.sum(1, keepdims=True))
    X = ft.opening_features(df).assign(elo_diff=1.0)
    Xs = ft.with_opening(X, df, "shin")
    assert np.allclose(Xs[["open_home", "open_draw", "open_away"]].to_numpy(), mg.shin(ODDS))
    assert (Xs["elo_diff"] == 1.0).all() and not np.allclose(X["open_home"], Xs["open_home"])
