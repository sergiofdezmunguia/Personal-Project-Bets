"""Fase 4: recolector de cuotas con respuestas inventadas (nunca toca la red)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import clv as cl
from src import football_data as fd
from src import odds_collector as oc

FIXTURES = """﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A,PPH,PPD,PPA,AvgH,AvgD,AvgA,BFEH,BFED,BFEA,B365>2.5,B365<2.5,Avg>2.5,Avg<2.5
SP1,21/11/2026,16:15,Equipo Uno,Equipo Dos,X,2.10,3.40,3.60,2.05,3.30,3.50,2.08,3.35,3.55,2.18,3.55,3.80,1.90,1.95,1.88,1.92
E0,28/03/2027,15:00,Club Tres,Club Cuatro,Y,1.80,3.80,4.50,1.75,3.75,4.40,1.78,3.70,4.45,1.84,3.95,4.80,,,1.70,2.10
E0,29/03/2027,20:00,Club Cinco,Club Seis,Z,1.80,0,4.50,,,,,,,,,,,,,
"""
VERSION = "Fri, 20 Nov 2026 14:16:25 GMT"


def odds_api_events(price_home=2.1, update="2026-11-20T10:00:00Z"):
    return [{"id": "abc123", "sport_key": "soccer_spain_la_liga", "commence_time": "2026-11-21T15:15:00Z",
             "home_team": "Equipo Uno", "away_team": "Equipo Dos",
             "bookmakers": [{"key": "betfair_ex_eu", "title": "Betfair", "last_update": update, "markets": [
                 {"key": "h2h", "outcomes": [{"name": "Equipo Uno", "price": price_home},
                                             {"name": "Equipo Dos", "price": 3.6}, {"name": "Draw", "price": 3.4}]},
                 {"key": "h2h_lay", "outcomes": [{"name": "Equipo Uno", "price": price_home + 0.04},
                                                 {"name": "Equipo Dos", "price": 3.7}, {"name": "Draw", "price": 3.5}]},
                 {"key": "totals", "outcomes": []}]}]}]  # fmt: skip


class FakeHTTP:
    """Servidor inventado: devuelve 304 si la petición trae If-Modified-Since igual a la versión."""

    def __init__(self, body: str, version: str = VERSION):
        self.body, self.version, self.calls = body.encode("utf-8"), version, []

    def __call__(self, url, headers=None, timeout=30):
        self.calls.append((url, dict(headers or {})))
        if (headers or {}).get("If-Modified-Since") == self.version:
            return oc.Response(304, b"", {})
        return oc.Response(200, self.body, {"last-modified": self.version})


def test_fixtures_are_parsed_into_snapshots():
    t = pd.Timestamp("2026-11-20 18:00", tz="UTC")
    s = oc.parse_fd_fixtures(FIXTURES, t, VERSION)
    assert list(s.columns) == cl.SNAP_COLS + ["source_version"]
    one = s[(s["event_key"].str.startswith("SP1")) & (s["bookmaker"] == "bet365") & (s["market"] == "1x2")]
    assert one.set_index("outcome")["price"].to_dict() == {"home": 2.10, "draw": 3.40, "away": 3.60}
    assert (s["captured_utc"] == t).all() and (s["price"] > 1).all()
    # cuota 0 y vacías se descartan; la fila 3 solo conserva las dos cuotas válidas de bet365
    assert set(s.loc[s["home_team"] == "Club Cinco", "outcome"]) == {"home", "away"}
    assert set(s["bookmaker"]) >= {"bet365", "paddypower", "avg", "betfair_exchange"}
    assert len(s[(s["market"] == "totals") & (s["home_team"] == "Club Tres")]) == 2  # solo la media


def test_kickoff_is_converted_from_uk_time_to_utc():
    s = oc.parse_fd_fixtures(FIXTURES, pd.Timestamp("2026-11-20", tz="UTC"), VERSION)
    ko = s.groupby("home_team")["kickoff_utc"].first()
    assert ko["Equipo Uno"] == pd.Timestamp("2026-11-21 16:15", tz="UTC")  # GMT en noviembre
    assert ko["Club Tres"] == pd.Timestamp("2027-03-28 14:00", tz="UTC")  # BST desde el 28-mar-2027


def test_event_key_matches_the_football_data_match_id():
    """El cierre llega después en el CSV de la temporada: la clave debe coincidir con `match_id`."""
    s = oc.parse_fd_fixtures(FIXTURES, pd.Timestamp("2026-11-20", tz="UTC"), VERSION)
    raw = pd.DataFrame({"Date": ["21/11/2026"], "Time": ["16:15"], "HomeTeam": ["Equipo Uno"],
                        "AwayTeam": ["Equipo Dos"]})  # fmt: skip
    assert fd.normalize(raw, "SP1", "2627")["match_id"].iloc[0] in set(s["event_key"])


def test_store_is_idempotent_and_keeps_the_first_capture_time(tmp_path):
    st = oc.Store(tmp_path / "c.sqlite")
    t1, t2 = pd.Timestamp("2026-11-20 18:00", tz="UTC"), pd.Timestamp("2026-11-20 22:00", tz="UTC")
    n = st.add(oc.parse_fd_fixtures(FIXTURES, t1, VERSION))
    assert n > 0
    assert st.add(oc.parse_fd_fixtures(FIXTURES, t2, VERSION)) == 0
    snaps = st.snapshots()
    assert len(snaps) == n and (snaps["captured_utc"] == t1).all()
    new_version = FIXTURES.replace("2.10,3.40,3.60", "2.00,3.50,3.80")
    assert st.add(oc.parse_fd_fixtures(new_version, t2, "Tue, 24 Nov 2026 14:00:00 GMT")) > 0
    s = cl.check_snapshots(st.snapshots())
    at = cl.latest_before(s, pd.Series({s["event_key"].iloc[0]: t1}))
    assert cl.wide_1x2(at, "bet365").iloc[0]["home"] == 2.10  # a t1 aún no existía la nueva versión


def test_collect_football_data_uses_conditional_requests(tmp_path):
    st = oc.Store(tmp_path / "c.sqlite")
    http = FakeHTTP(FIXTURES)
    first = oc.collect_football_data(st, get=http)
    second = oc.collect_football_data(st, get=http)
    assert first["estado"] == 200 and first["nuevas"] > 0
    assert second == {"estado": 304, "filas": 0, "nuevas": 0}
    assert "If-Modified-Since" not in http.calls[0][1] and http.calls[1][1]["If-Modified-Since"] == VERSION
    assert list(st.captures()["status"]) == ["200", "304"]
    assert len(st.snapshots()) == first["nuevas"]


def test_odds_api_parser_maps_outcomes_and_lay_prices():
    s = oc.parse_odds_api(odds_api_events(), pd.Timestamp("2026-11-20 12:00", tz="UTC"), "SP1")
    back = s[s["kind"] == "back"].set_index("outcome")["price"].to_dict()
    assert back == {"home": 2.1, "away": 3.6, "draw": 3.4}
    assert (s["kind"] == "lay").sum() == 3 and (s["market"] == "1x2").all()
    assert s["kickoff_utc"].iloc[0] == pd.Timestamp("2026-11-21 15:15", tz="UTC")


def test_odds_api_requires_a_key_and_respects_the_reserve(tmp_path, monkeypatch):
    st = oc.Store(tmp_path / "c.sqlite")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        oc.collect_odds_api(st, ["SP1"])
    calls = []

    def get(url, headers=None, timeout=30):
        calls.append(url)
        return oc.Response(200, json.dumps(odds_api_events()).encode(), {"x-requests-remaining": "15"})

    out = oc.collect_odds_api(st, ["SP1", "E0"], api_key="clave-inventada", get=get, pause=0)
    assert len(calls) == 1 and "parada" in out  # 15 < 20 de reserva: no gasta más
    again = oc.collect_odds_api(st, ["SP1"], api_key="clave-inventada", get=get, pause=0)
    assert again["ligas"]["SP1"]["nuevas"] == 0  # misma last_update: idempotente
    assert "clave-inventada" not in st.captures().to_string()  # la clave no se guarda


def test_collected_snapshots_feed_the_clv_simulation(tmp_path):
    """De extremo a extremo con datos inventados: dos capturas y un cierre."""
    st = oc.Store(tmp_path / "c.sqlite")
    t1 = pd.Timestamp("2026-11-20 12:00", tz="UTC")
    st.add(oc.parse_odds_api(odds_api_events(2.30, "2026-11-20T11:00:00Z"), t1, "SP1"))
    st.add(oc.parse_odds_api(odds_api_events(1.95, "2026-11-21T15:10:00Z"),
                             pd.Timestamp("2026-11-21 15:12", tz="UTC"), "SP1"))  # fmt: skip
    p = pd.DataFrame({"p_home": [0.5], "p_draw": [0.27], "p_away": [0.23], "decision_utc": [t1]}, index=["toa_abc123"])
    bets, info = cl.simulate(st.snapshots(), p, bet_book="betfair_ex_eu", close_book="betfair_ex_eu", threshold=0.0,
                             method="shin", commission=0.05, max_close_gap_min=30)  # fmt: skip
    assert info["apuestas"] == 1 and bets["price"].iloc[0] == 2.30 and bets["close_price"].iloc[0] == 1.95
    assert bets["clv_ev"].iloc[0] > 0
