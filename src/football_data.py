"""Histórico unificado de football-data.co.uk (fase 1 del plan).

Lee los CSV que deja `scripts/descargar_football_data.py` en `data/raw/football-data/` y los
convierte en una sola tabla con nombres de columna estables entre temporadas:

- Las columnas de mercado cambian de nombre con los años (`BbAvH` hasta 2018/19, `AvgH` desde
  2019/20) y algunos ficheros traen BOM en la primera columna. Aquí se unifican.
- `PP*` (2026/27) **no es Pinnacle**: su margen medio es del 8,3 % frente al 3,4 % de Pinnacle
  (`PS*`) en 2024/25. Se guarda aparte como `pp_*` y nunca cuenta como referencia.
- La referencia de cierre (`ref_close_*`) sigue `docs/decisions.md`: Pinnacle si existe; si no,
  Betfair Exchange; si no, la media del mercado. `ref_source` dice cuál se usó en cada partido.

Integridad temporal: la tabla solo describe partidos. No calcula variables a partir de otros
partidos; eso lo hacen las fases siguientes respetando la hora de la apuesta.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LEAGUES = {
    "SP1": ("España", "La Liga"),
    "SP2": ("España", "LaLiga 2"),
    "E0": ("Inglaterra", "Premier League"),
    "E1": ("Inglaterra", "Championship"),
    "E2": ("Inglaterra", "League One"),
    "D1": ("Alemania", "Bundesliga"),
    "I1": ("Italia", "Serie A"),
    "F1": ("Francia", "Ligue 1"),
}
FINAL_TEST_FROM = pd.Timestamp("2025-07-01")  # 2025/26: prueba final, intacta (CLAUDE.md)

# nombre estable -> nombres posibles en football-data, por orden de preferencia
COLUMNS: dict[str, tuple[str, ...]] = {
    "home_goals": ("FTHG",),
    "away_goals": ("FTAG",),
    "result": ("FTR",),
    "ht_home_goals": ("HTHG",),
    "ht_away_goals": ("HTAG",),
    "home_shots": ("HS",),
    "away_shots": ("AS",),
    "home_shots_target": ("HST",),
    "away_shots_target": ("AST",),
    "home_corners": ("HC",),
    "away_corners": ("AC",),
    "home_fouls": ("HF",),
    "away_fouls": ("AF",),
    "home_yellow": ("HY",),
    "away_yellow": ("AY",),
    "home_red": ("HR",),
    "away_red": ("AR",),
    "referee": ("Referee",),
}
for side, s in (("home", "H"), ("draw", "D"), ("away", "A")):
    COLUMNS |= {
        f"pin_open_{side}": (f"PS{s}",),
        f"pin_close_{side}": (f"PSC{s}",),
        f"avg_open_{side}": (f"Avg{s}", f"BbAv{s}"),
        f"avg_close_{side}": (f"AvgC{s}",),
        f"max_open_{side}": (f"Max{s}", f"BbMx{s}"),
        f"max_close_{side}": (f"MaxC{s}",),
        f"bfe_open_{side}": (f"BFE{s}",),
        f"bfe_close_{side}": (f"BFEC{s}",),
        f"pp_open_{side}": (f"PP{s}",),
        f"pp_close_{side}": (f"PPC{s}",),
    }
for side, s in (("over", ">"), ("under", "<")):
    COLUMNS |= {
        f"pin_close_{side}25": (f"PC{s}2.5",),
        f"avg_close_{side}25": (f"AvgC{s}2.5",),
        f"bfe_close_{side}25": (f"BFEC{s}2.5",),
    }
COLUMNS |= {
    "ah_close_line": ("AHCh",),
    "pin_close_ah_home": ("PCAHH",),
    "pin_close_ah_away": ("PCAHA",),
    "avg_close_ah_home": ("AvgCAHH",),
    "avg_close_ah_away": ("AvgCAHA",),
}
REF_SOURCES = (("pinnacle", "pin_close"), ("betfair", "bfe_close"), ("media", "avg_close"))


def season_label(code: str) -> str:
    """'1213' -> '2012/13'."""
    start = 2000 + int(code[:2])
    return f"{start}/{code[2:]}"


def team_id(name: str) -> str:
    """Identidad de equipo: el nombre de football-data normalizado. Es único entre países en
    las 8 ligas descargadas (comprobado) y estable entre divisiones, así que un ascenso no corta
    el historial. Los cambios de nombre se resolverían con `TEAM_ALIASES`."""
    name = TEAM_ALIASES.get(name, name)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


TEAM_ALIASES: dict[str, str] = {}


def read_raw(path: Path) -> pd.DataFrame:
    """Un CSV de football-data tal cual, sin BOM ni filas vacías."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    return df.dropna(how="all").dropna(subset=["HomeTeam", "AwayTeam"])


def normalize(raw: pd.DataFrame, league: str, season_code: str) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["league"] = league
    out["country"] = LEAGUES.get(league, ("?", "?"))[0]
    out["season"] = season_label(season_code)
    out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
    out["kickoff"] = raw["Time"] if "Time" in raw else np.nan
    out["home_team"] = raw["HomeTeam"].str.strip()
    out["away_team"] = raw["AwayTeam"].str.strip()
    out["home_team_id"] = out["home_team"].map(team_id)
    out["away_team_id"] = out["away_team"].map(team_id)
    for name, sources in COLUMNS.items():
        src = next((c for c in sources if c in raw.columns), None)
        out[name] = raw[src] if src is not None else np.nan
    out["match_id"] = (
        league + "_" + out["date"].dt.strftime("%Y%m%d") + "_" + out["home_team_id"] + "_"
        + out["away_team_id"]
    )  # fmt: skip
    return out


def add_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Cierre de referencia (1X2) y su probabilidad sin margen (normalización proporcional)."""
    df = df.copy()
    df["ref_source"] = None
    for side in ("home", "draw", "away"):
        df[f"ref_close_{side}"] = np.nan
    for source, prefix in REF_SOURCES:
        cols = [f"{prefix}_{s}" for s in ("home", "draw", "away")]
        ok = df["ref_source"].isna() & df[cols].notna().all(axis=1) & (df[cols] > 1).all(axis=1)
        for side, col in zip(("home", "draw", "away"), cols, strict=True):
            df.loc[ok, f"ref_close_{side}"] = df.loc[ok, col]
        df.loc[ok, "ref_source"] = source
    inv = 1 / df[[f"ref_close_{s}" for s in ("home", "draw", "away")]]
    df["ref_overround"] = inv.sum(axis=1, min_count=3)
    for side in ("home", "draw", "away"):
        df[f"ref_prob_{side}"] = inv[f"ref_close_{side}"] / df["ref_overround"]
    return df


def load_all(raw_dir: Path) -> pd.DataFrame:
    """Todas las ligas y temporadas, ordenadas por fecha, con referencia de cierre."""
    frames = []
    for path in sorted(Path(raw_dir).glob("*_*.csv")):
        league, code = path.stem.split("_")
        frames.append(normalize(read_raw(path), league, code))
    if not frames:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}: ejecuta scripts/descargar_football_data.py")
    df = pd.concat(frames, ignore_index=True)
    df = add_reference(df)
    df["is_final_test"] = df["date"] >= FINAL_TEST_FROM
    return df.sort_values(["date", "league", "home_team"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------- calidad


def quality(df: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    """Indicadores del informe de calidad. Solo recuentos y proporciones."""
    by = df.groupby(["league", "season"])
    cover = lambda col: by[col].apply(lambda s: s.notna().mean())  # noqa: E731
    seasons = by.size().rename("partidos").to_frame()
    seasons["fecha_ok"] = cover("date")
    seasons["hora"] = cover("kickoff")
    seasons["tiros"] = cover("home_shots")
    seasons["pinnacle_cierre"] = cover("pin_close_home")
    seasons["betfair_cierre"] = cover("bfe_close_home")
    seasons["media_cierre"] = cover("avg_close_home")
    seasons["con_referencia"] = cover("ref_close_home")
    ref = df.groupby(["season", "ref_source"], dropna=False).size().unstack(fill_value=0)
    margin = df.groupby(["season", "ref_source"])["ref_overround"].mean().unstack() - 1
    teams = df.groupby("league")["home_team_id"].nunique().rename("equipos")
    return {
        "partidos": len(df),
        "duplicados": int(df.duplicated("match_id").sum()),
        "fechas_ilegibles": int(df["date"].isna().sum()),
        "sin_resultado": int(df["result"].isna().sum()),
        "resultado_incoherente": int(
            (
                df["result"].notna()
                & (
                    np.sign(df["home_goals"] - df["away_goals"])
                    .map({1: "H", 0: "D", -1: "A"})
                    .ne(df["result"])
                )
            ).sum()
        ),
        "cuotas_imposibles": int(
            (df[[c for c in df.columns if c.startswith(("pin_", "avg_", "bfe_", "max_"))
                 and not c.endswith("line")]] <= 1).sum().sum()
        ),  # fmt: skip
        "por_temporada": seasons,
        "referencia": ref,
        "margen_referencia": margin,
        "equipos": teams,
    }
