"""Fase 4: CLV de apuestas simuladas, con instantáneas inventadas.

Antifuga: la decisión solo puede ver instantáneas con hora de captura ≤ hora de decisión y
anteriores al saque. Se comprueba alterando lo posterior (no cambia ninguna apuesta) y con un
mutante que lee la última cuota sin mirar la hora (la comprobación lo detecta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clv as cl
from src import features as ft
from src import margin as mg

T0 = pd.Timestamp("2026-11-20 12:00", tz="UTC")


def snap(event, minutes_before_ko, prices, book="casa", ko=T0, kind="back"):
    """Una instantánea 1X2 vista `minutes_before_ko` minutos antes del saque."""
    return [{"source": "inventada", "event_key": event, "league": "XX", "kickoff_utc": ko, "home_team": "A",
             "away_team": "B", "market": "1x2", "line": "", "bookmaker": book, "outcome": o, "kind": kind,
             "price": pr, "source_updated_utc": pd.NaT, "captured_utc": ko - pd.Timedelta(minutes=minutes_before_ko)}
            for o, pr in zip(cl.OUTCOMES, prices, strict=True)]  # fmt: skip


@pytest.fixture
def book():
    rows = []
    rows += snap("e1", 2880, (2.20, 3.40, 3.40))  # T-48 h
    rows += snap("e1", 600, (2.00, 3.50, 3.90))   # T-10 h
    rows += snap("e1", 5, (1.90, 3.60, 4.20))     # cierre
    rows += snap("e1", -30, (1.01, 30.0, 50.0))   # en juego: nunca cuenta
    rows += snap("e2", 1440, (3.00, 3.30, 2.40))
    rows += snap("e2", 10, (3.30, 3.30, 2.25))
    return pd.DataFrame(rows)


def preds(decision_min_before_ko, p=(0.52, 0.26, 0.22), events=("e1",)):
    return pd.DataFrame([dict(zip(cl.PROB, p, strict=True)) | {"decision_utc": T0 - pd.Timedelta(minutes=decision_min_before_ko)}
                         for _ in events], index=list(events))  # fmt: skip


def test_price_at_decision_is_the_last_one_seen_before_it(book):
    at = cl.latest_before(book, pd.Series({"e1": T0 - pd.Timedelta(hours=20)}))
    w = cl.wide_1x2(at, "casa")
    assert w.loc["e1", cl.OUTCOMES].tolist() == [2.20, 3.40, 3.40]
    at = cl.latest_before(book, pd.Series({"e1": T0 - pd.Timedelta(minutes=600)}))  # igual: se ve
    assert cl.wide_1x2(at, "casa").loc["e1", "home"] == 2.00


def test_closing_ignores_in_play_snapshots(book):
    c = cl.closing(book)
    w = cl.wide_1x2(c, "casa")
    assert w.loc["e1", cl.OUTCOMES].tolist() == [1.90, 3.60, 4.20]
    assert c.loc[c["event_key"] == "e1", "gap_min"].eq(5).all()


def test_clv_values_by_hand(book):
    bets, info = cl.simulate(book, preds(2880), bet_book="casa", close_book="casa", threshold=0.0,
                             method="proporcional")  # fmt: skip
    assert info["apuestas"] == 1 and bets["outcome"].iloc[0] == "home"
    q = 1 / np.array([1.90, 3.60, 4.20])
    p_close = q[0] / q.sum()
    assert bets["price"].iloc[0] == 2.20
    assert bets["clv_ev"].iloc[0] == pytest.approx(2.20 * p_close - 1)
    assert bets["clv_log"].iloc[0] == pytest.approx(np.log(2.20 / 1.90))
    assert bets["ev"].iloc[0] == pytest.approx(0.52 * 2.20 - 1)


def test_threshold_and_commission(book):
    kw = dict(bet_book="casa", close_book="casa", method="shin")
    assert cl.simulate(book, preds(2880), threshold=0.15, **kw)[1]["apuestas"] == 0  # EV = 14,4 %
    bets, _ = cl.simulate(book, preds(2880), threshold=0.0, commission=0.05, **kw)
    eff = 1 + 1.20 * 0.95
    assert bets["ev"].iloc[0] == pytest.approx(0.52 * eff - 1)
    assert bets["clv_ev"].iloc[0] == pytest.approx(eff * mg.shin([[1.90, 3.60, 4.20]])[0, 0] - 1)


def test_changing_anything_after_the_decision_does_not_change_the_bets(book):
    dec = 600
    base, _ = cl.simulate(book, preds(dec), bet_book="casa", close_book="casa", threshold=0.0, method="shin")
    later = book.copy()
    after = later["captured_utc"] > T0 - pd.Timedelta(minutes=dec)
    later.loc[after, "price"] = later.loc[after, "price"] * 1.7
    moved, _ = cl.simulate(later, preds(dec), bet_book="casa", close_book="casa", threshold=0.0, method="shin")
    cols = ["event_key", "outcome", "price", "p_model", "ev"]
    pd.testing.assert_frame_equal(base[cols], moved[cols])
    assert not np.allclose(base["clv_ev"], moved["clv_ev"])  # el cierre sí cambia el CLV


def test_ignoring_the_capture_time_is_caught(book, monkeypatch):
    """Mutante: precio = última cuota guardada antes del saque, sin mirar la hora de decisión. La
    guarda de `simulate` lo detecta (una cuota usada es posterior a la decisión)."""
    real = cl.latest_before

    def leaky(snaps, cutoff, *, strict=False):
        return real(snaps, pd.Series(T0, index=cutoff.index), strict=True)

    monkeypatch.setattr(cl, "latest_before", leaky)
    with pytest.raises(AssertionError, match="posterior a la decisión"):
        cl.simulate(book, preds(600, (0.6, 0.2, 0.2)), bet_book="casa", close_book="casa", threshold=0.0, method="shin")


def test_predictions_made_at_or_after_kickoff_are_dropped(book):
    bets, info = cl.simulate(book, preds(0), bet_book="casa", close_book="casa", threshold=0.0, method="shin")
    assert info["tardias"] == 1 and bets.empty


def test_market_as_model_never_bets_and_closing_as_model_has_positive_clv(book):
    """Controles: la apertura sin margen no tiene valor esperado positivo (paga el margen); el cierre
    usado como predicción (una fuga deliberada) da CLV positivo en todas sus apuestas."""
    for ev, dec, close in (("e1", 2880, (1.90, 3.60, 4.20)), ("e2", 1440, (3.30, 3.30, 2.25))):
        opening = cl.wide_1x2(cl.latest_before(book, pd.Series({ev: T0 - pd.Timedelta(minutes=dec)})), "casa")
        p_open = mg.shin(opening[cl.OUTCOMES].to_numpy())[0]
        assert cl.simulate(book, preds(dec, p_open, (ev,)), bet_book="casa", close_book="casa", threshold=0.0,
                           method="shin")[1]["apuestas"] == 0  # fmt: skip
        oracle = mg.shin([close])[0]
        bets, _ = cl.simulate(book, preds(dec, oracle, (ev,)), bet_book="casa", close_book="casa", threshold=0.0,
                              method="shin")  # fmt: skip
        assert len(bets) == 1 and (bets["clv_ev"] > 0).all()


def test_bets_without_a_close_are_counted(book):
    only_open = book[book["captured_utc"] < T0 - pd.Timedelta(hours=20)]
    other = pd.DataFrame(snap("e1", 5, (1.9, 3.6, 4.2), book="otra"))
    both = pd.concat([only_open, other])
    kw = dict(bet_book="casa", threshold=0.0, method="shin")
    bets, info = cl.simulate(both, preds(2880), close_book="sin_datos", **kw)
    assert info["apuestas"] == 1 and info["sin_cierre"] == 1 and bets.empty
    # sin tope, la apertura de T-48 h haría de «cierre»; con tope de 60 min no cuenta
    assert cl.simulate(both, preds(2880), close_book="casa", **kw)[1]["sin_cierre"] == 0
    bets, info = cl.simulate(both, preds(2880), close_book="casa", max_close_gap_min=60, **kw)
    assert info["sin_cierre"] == 1 and bets.empty


def test_summarize_gives_an_interval_around_the_mean():
    rng = np.random.default_rng(0)
    n = 400
    bets = pd.DataFrame({"clv_ev": rng.normal(0.01, 0.05, n), "clv_log": rng.normal(0.01, 0.05, n),
                         "ev": rng.uniform(0, 0.1, n), "price": rng.uniform(1.5, 4, n)})  # fmt: skip
    s = cl.summarize(bets, np.repeat(np.arange(40), 10), n_boot=300, won=rng.random(n) < 0.4)
    assert s["n"] == n and s["semanas"] == 40
    assert s["clv_ev_ic"][0] < s["clv_ev"] < s["clv_ev_ic"][1]
    assert "roi" in s and cl.summarize(bets.iloc[:0], np.array([]))["n"] == 0


def test_history_snapshots_keep_open_before_close_before_kickoff():
    dates = pd.Series(pd.date_range("2024-08-05", periods=14, freq="D"))  # todos los días de la semana
    df = pd.DataFrame({"date": dates, "match_id": [f"m{i}" for i in range(14)], "league": "XX",
                       "home_team": "A", "away_team": "B"})  # fmt: skip
    for pre, base in (("pin_open", 2.0), ("pin_close", 1.9)):
        df[f"{pre}_home"], df[f"{pre}_draw"], df[f"{pre}_away"] = base, 3.5, 4.0
    s = cl.history_snapshots(df, ft.opening_anchor(df["date"]))
    for _, g in s.groupby("event_key"):
        opened = g[g["price"] == 2.0]["captured_utc"].max()
        closed = g[g["price"] == 1.9]["captured_utc"].min()
        assert opened <= closed < g["kickoff_utc"].iloc[0]
    dec = ft.opening_anchor(df["date"]).dt.tz_localize("UTC")
    p = pd.DataFrame({"p_home": 0.6, "p_draw": 0.2, "p_away": 0.2, "decision_utc": dec.to_numpy()},
                     index=df["match_id"].to_numpy())  # fmt: skip
    bets, info = cl.simulate(s, p, bet_book="pinnacle", close_book="pinnacle", threshold=0.0, method="shin")
    assert info["apuestas"] == 14 and (bets["price"] == 2.0).all()  # siempre la apertura, nunca el cierre
