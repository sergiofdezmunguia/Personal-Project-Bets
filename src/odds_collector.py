"""Fase 4: recolector idempotente de cuotas propias con hora de captura.

Guarda instantáneas (`src/clv.SNAP_COLS`) en una base SQLite **fuera de git** (por defecto
`data/odds/cuotas.sqlite`, o `BETS_ODDS_DIR`). Es idempotente: cada instantánea tiene una clave
única (fuente, partido, mercado, línea, casa, resultado, tipo y versión de la fuente) y se inserta con
`INSERT OR IGNORE`, así que repetir una captura sin cambios no añade filas y conserva la primera
`captured_utc` (la hora a la que la vimos por primera vez, que es la que cuenta para decidir).

Fuentes (ver `docs/clv.md`):

- **football-data `fixtures.csv`** (sin clave ni registro): próximos partidos con cuotas de varias
  casas, recogidas por football-data los martes y viernes por la tarde. Versión = `Last-Modified`.
  Se pide con `If-Modified-Since`, así que una captura sin cambios es una respuesta 304 vacía.
- **The Odds API v4** (necesita clave gratuita con registro: **no configurada**). El analizador está
  probado con respuestas inventadas; la descarga exige `ODDS_API_KEY` en `.env`. Versión =
  `last_update` de cada casa.

Nada de esto coloca apuestas ni usa cuentas de casas de apuestas.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.clv import SNAP_COLS
from src.football_data import team_id

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Personal-Project-Bets/0.4 (investigacion personal; recolector con pausas y cache)"
FD_FIXTURES = "https://football-data.co.uk/fixtures.csv"
FD_TZ = "Europe/London"  # hipótesis: notes.txt no dice la zona; football-data es británico
FD_BOOKS_1X2 = {"B365": "bet365", "BFD": "betfred", "BV": "betvictor", "BW": "bwin", "PP": "paddypower",
                "SKB": "skybet", "Max": "max", "Avg": "avg", "BFE": "betfair_exchange", "PS": "pinnacle"}  # fmt: skip
FD_BOOKS_OU = {"B365": "bet365", "Max": "max", "Avg": "avg", "BFE": "betfair_exchange", "P": "pinnacle"}
TOA_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
TOA_SPORTS = {"SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division", "E0": "soccer_epl",
              "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "D1": "soccer_germany_bundesliga",
              "I1": "soccer_italy_serie_a", "F1": "soccer_france_ligue_one"}  # fmt: skip  # sin verificar con clave
PAUSE_S = 2.0
UNIQUE = ["source", "event_key", "market", "line", "bookmaker", "outcome", "kind", "source_version"]


def default_dir() -> Path:
    return Path(os.environ.get("BETS_ODDS_DIR", ROOT / "data" / "odds"))


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


# ----------------------------------------------------------------------------- analizadores


def parse_fd_fixtures(text: str, captured_utc, source_version: str) -> pd.DataFrame:
    """`fixtures.csv` de football-data → instantáneas (1X2 y más/menos de 2,5, cuotas previas)."""
    raw = pd.read_csv(io.StringIO(text.lstrip("﻿")), on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    date = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
    tm = raw["Time"].fillna("12:00") if "Time" in raw else "12:00"
    local = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + tm.astype(str), errors="coerce")
    kickoff = local.dt.tz_localize(FD_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    home, away = raw["HomeTeam"].str.strip(), raw["AwayTeam"].str.strip()
    key = raw["Div"] + "_" + date.dt.strftime("%Y%m%d") + "_" + home.map(team_id) + "_" + away.map(team_id)
    base = pd.DataFrame({"source": "football-data-fixtures", "event_key": key, "league": raw["Div"],
                         "kickoff_utc": kickoff, "home_team": home, "away_team": away, "kind": "back"})  # fmt: skip
    parts = []
    for col, book in FD_BOOKS_1X2.items():
        for s, outcome in (("H", "home"), ("D", "draw"), ("A", "away")):
            if f"{col}{s}" in raw:
                parts.append(base.assign(market="1x2", line="", bookmaker=book, outcome=outcome,
                                         price=pd.to_numeric(raw[f"{col}{s}"], errors="coerce")))  # fmt: skip
    for col, book in FD_BOOKS_OU.items():
        for s, outcome in ((">", "over"), ("<", "under")):
            if f"{col}{s}2.5" in raw:
                parts.append(base.assign(market="totals", line="2.5", bookmaker=book, outcome=outcome,
                                         price=pd.to_numeric(raw[f"{col}{s}2.5"], errors="coerce")))  # fmt: skip
    if not parts:
        return pd.DataFrame(columns=SNAP_COLS + ["source_version"])
    out = pd.concat(parts, ignore_index=True)
    out["source_updated_utc"] = pd.to_datetime(source_version, utc=True, errors="coerce")
    out["captured_utc"] = pd.Timestamp(captured_utc)
    out["source_version"] = source_version
    out = out[(out["price"] > 1) & out["kickoff_utc"].notna()]
    return out[SNAP_COLS + ["source_version"]].reset_index(drop=True)


def parse_odds_api(events: list[dict], captured_utc, league: str) -> pd.DataFrame:
    """Respuesta de `GET /v4/sports/{sport}/odds` (cuotas decimales) → instantáneas."""
    rows = []
    for ev in events:
        names = {ev["home_team"]: "home", ev["away_team"]: "away", "Draw": "draw"}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] not in ("h2h", "h2h_lay"):
                    continue
                for oc in mk["outcomes"]:
                    rows.append({"source": "the-odds-api", "event_key": f"toa_{ev['id']}", "league": league,
                                 "kickoff_utc": ev["commence_time"], "home_team": ev["home_team"],
                                 "away_team": ev["away_team"], "market": "1x2", "line": "",
                                 "bookmaker": bk["key"], "outcome": names.get(oc["name"]),
                                 "kind": "lay" if mk["key"] == "h2h_lay" else "back", "price": oc["price"],
                                 "source_updated_utc": bk.get("last_update"), "captured_utc": captured_utc,
                                 "source_version": bk.get("last_update") or ""})  # fmt: skip
    out = pd.DataFrame(rows, columns=SNAP_COLS + ["source_version"])
    if out.empty:
        return out
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    out["source_updated_utc"] = pd.to_datetime(out["source_updated_utc"], utc=True)
    out["captured_utc"] = pd.Timestamp(captured_utc)
    return out[out["outcome"].notna() & (pd.to_numeric(out["price"]) > 1)].reset_index(drop=True)


# ----------------------------------------------------------------------------- almacén


class Store:
    """Instantáneas en SQLite con clave única (idempotente) y un registro de capturas."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            cols = ", ".join(f"{c} TEXT" if c != "price" else "price REAL" for c in SNAP_COLS + ["source_version"])
            con.execute(f"CREATE TABLE IF NOT EXISTS snapshots ({cols}, UNIQUE ({', '.join(UNIQUE)}))")
            con.execute("CREATE TABLE IF NOT EXISTS captures (captured_utc TEXT, source TEXT, status TEXT, "
                        "source_version TEXT, n_rows INTEGER, n_new INTEGER, detail TEXT)")  # fmt: skip

    def _con(self):
        return sqlite3.connect(self.path)

    def add(self, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        r = rows[SNAP_COLS + ["source_version"]].copy()
        for c in ("kickoff_utc", "source_updated_utc", "captured_utc"):
            r[c] = pd.to_datetime(r[c], utc=True).map(lambda t: t.isoformat() if pd.notna(t) else None)
        r["line"] = r["line"].fillna("").astype(str)
        cols = list(r.columns)
        with self._con() as con:
            before = con.total_changes
            con.executemany(f"INSERT OR IGNORE INTO snapshots ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            r.astype(object).where(r.notna(), None).itertuples(index=False, name=None))  # fmt: skip
            return con.total_changes - before

    def log_capture(self, source, status, version, n_rows, n_new, detail="") -> None:
        with self._con() as con:
            con.execute("INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (now_utc().isoformat(), source, status, version, n_rows, n_new, detail))  # fmt: skip

    def last_version(self, source: str) -> str | None:
        with self._con() as con:
            r = con.execute("SELECT source_version FROM captures WHERE source = ? AND status = '200' "
                            "ORDER BY captured_utc DESC LIMIT 1", (source,)).fetchone()  # fmt: skip
        return r[0] if r else None

    def snapshots(self) -> pd.DataFrame:
        with self._con() as con:
            df = pd.read_sql_query("SELECT * FROM snapshots", con)
        for c in ("kickoff_utc", "source_updated_utc", "captured_utc"):
            df[c] = pd.to_datetime(df[c], utc=True)
        return df

    def captures(self) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql_query("SELECT * FROM captures", con)


# ----------------------------------------------------------------------------- descargas


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict


def http_get(url: str, headers: dict | None = None, timeout: float = 30) -> Response:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Response(r.status, r.read(), {k.lower(): v for k, v in r.headers.items()})
    except urllib.error.HTTPError as e:
        return Response(e.code, b"", {k.lower(): v for k, v in (e.headers or {}).items()})


def collect_football_data(store: Store, get=http_get) -> dict:
    """Una captura de `fixtures.csv`. Con `If-Modified-Since` una repetición sin cambios es un 304."""
    src = "football-data-fixtures"
    last = store.last_version(src)
    conditional = last and not last.startswith("sha256:")  # solo una fecha HTTP vale para la cabecera
    r = get(FD_FIXTURES, {"If-Modified-Since": last} if conditional else None)
    if r.status == 304:
        store.log_capture(src, "304", last, 0, 0)
        return {"estado": 304, "filas": 0, "nuevas": 0}
    if r.status != 200:
        store.log_capture(src, str(r.status), None, 0, 0)
        return {"estado": r.status, "filas": 0, "nuevas": 0}
    # sin Last-Modified, la versión es la huella del contenido: repetir la misma descarga no duplica filas
    version = r.headers.get("last-modified") or "sha256:" + hashlib.sha256(r.body).hexdigest()[:16]
    rows = parse_fd_fixtures(r.body.decode("utf-8-sig", errors="replace"), now_utc(), version)
    n_new = store.add(rows)
    store.log_capture(src, "200", version, len(rows), n_new)
    return {"estado": 200, "filas": len(rows), "nuevas": n_new, "version": version}


def collect_odds_api(store: Store, leagues: list[str], *, api_key: str | None = None, regions: str = "eu",
                     reserve: int = 20, get=http_get, pause: float = PAUSE_S) -> dict:  # fmt: skip
    """Una captura de The Odds API (1 crédito por liga con una región y el mercado h2h).
    Se detiene si quedan menos de `reserve` créditos. Requiere `ODDS_API_KEY`."""
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Falta ODDS_API_KEY en .env (The Odds API necesita registro: ver docs/clv.md)")
    src, out = "the-odds-api", {"ligas": {}}
    for i, lg in enumerate(leagues):
        if i:
            time.sleep(pause)
        url = TOA_BASE.format(sport=TOA_SPORTS[lg]) + f"?apiKey={key}&regions={regions}&markets=h2h&oddsFormat=decimal"
        r = get(url)
        remaining = r.headers.get("x-requests-remaining")
        if r.status != 200:
            store.log_capture(src, str(r.status), None, 0, 0, f"{lg}; restantes {remaining}")
            out["ligas"][lg] = {"estado": r.status}
            continue
        rows = parse_odds_api(json.loads(r.body), now_utc(), lg)
        n_new = store.add(rows)
        store.log_capture(src, "200", None, len(rows), n_new, f"{lg}; restantes {remaining}")
        out["ligas"][lg] = {"estado": 200, "filas": len(rows), "nuevas": n_new, "restantes": remaining}
        if remaining is not None and float(remaining) < reserve:
            out["parada"] = f"quedan {remaining} créditos (< {reserve})"
            break
    return out
