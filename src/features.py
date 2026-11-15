"""Fase 3: variables previas al partido para los modelos de ML.

Dos horas de decisión (ver `docs/decisions.md`, 2026-11-15, fase 3):

- **saque** (`kickoff_anchor`): justo antes del partido. El partido del día D ve los resultados de
  días < D, como en la fase 2. Es la hora de la cuota de cierre.
- **apertura** (`opening_anchor`): cuando football-data recoge las cuotas previas, «el viernes por
  la tarde para los partidos de fin de semana y el martes por la tarde para los de entre semana»
  (notes.txt). Un partido de viernes a lunes solo ve resultados anteriores a su viernes; uno de
  martes a jueves, anteriores a su martes. Es la hora de la cuota de apertura de Pinnacle.

Todas las variables de un partido con ancla A solo dependen de resultados de partidos con fecha < A
(tests en `tests/test_ml.py`). Las fechas del calendario (descanso, partidos jugados en la temporada)
se conocen de antemano y se usan tal cual. **Ninguna variable lee cuotas de cierre**; la cuota de
apertura solo entra en los modelos de residuos, y solo con el ancla de apertura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dixon_coles import DixonColes
from src.dixon_coles import monday as monday_anchor
from src.dixon_coles import predict_walk_forward as dc_walk_forward
from src.ratings import TIER, Elo, PiRating, run_online

FORM_WINDOWS = (5, 20)
FORM_STATS = ("gf", "ga", "sf", "sa", "tf", "ta", "cf", "ca", "pts")  # goles, tiros, a puerta, córners
RAW_STATS = {"gf": "goals", "sf": "shots", "tf": "shots_target", "cf": "corners"}
LEAGUE_CODE = {lg: i for i, lg in enumerate(sorted(TIER))}
FORBIDDEN = ("close", "ref_", "result", "goals", "_prob")  # ninguna variable puede llamarse así


def kickoff_anchor(dates: pd.Series) -> pd.Series:
    """Decisión antes del saque: se ve todo lo anterior al día del partido."""
    return pd.Series(pd.to_datetime(dates).dt.normalize(), index=dates.index)


def opening_anchor(dates: pd.Series) -> pd.Series:
    """Decisión al recoger las cuotas previas: martes para mar–jue, viernes para vie–lun."""
    d = pd.to_datetime(dates).dt.normalize()
    wd = d.dt.day_of_week  # lunes = 0
    back = np.where(wd.between(1, 3), wd - 1, (wd - 4) % 7)
    return pd.Series(d - pd.to_timedelta(back, unit="D"), index=dates.index)


ANCHORS = {"saque": kickoff_anchor, "apertura": opening_anchor}
DC_ANCHORS = {"saque": monday_anchor, "apertura": opening_anchor}  # Dixon-Coles se reajusta en cada ancla


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido, con sus estadísticas a favor y en contra."""
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        p = pd.DataFrame({"row": df.index, "side": side, "date": df["date"].to_numpy(),
                          "season": df["season"].to_numpy(), "team": df[f"{side}_team_id"].to_numpy()})  # fmt: skip
        for short, raw in RAW_STATS.items():
            p[short] = df[f"{side}_{raw}"].to_numpy(float) if f"{side}_{raw}" in df else np.nan
            p[short[0] + "a"] = df[f"{other}_{raw}"].to_numpy(float) if f"{other}_{raw}" in df else np.nan
        hg, ag = df[f"{side}_goals"].to_numpy(float), df[f"{other}_goals"].to_numpy(float)
        p["pts"] = np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0))
        parts.append(p)
    return pd.concat(parts, ignore_index=True).sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def form_features(df: pd.DataFrame, anchor) -> pd.DataFrame:
    """Medias móviles por equipo (últimos 5 y 20 partidos, de cualquier campo) vistas en el ancla.

    Las medias se calculan **por equipo** y se consultan con `merge_asof` estricto: el partido del
    día D toma la media tras el último partido del equipo con fecha < anchor(D)."""
    long = _long(df)
    g = long.groupby("team", sort=False)
    rolled = long[["team", "date"]].copy()
    for n in FORM_WINDOWS:
        r = g[list(FORM_STATS)].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        for st in FORM_STATS:
            rolled[f"{st}{n}"] = r[st]
    rolled["n_prev"] = g.cumcount() + 1
    cut = pd.Series(pd.to_datetime(anchor(df["date"])), index=df.index)
    out = pd.DataFrame(index=df.index)
    stat_cols = [c for c in rolled.columns if c not in ("team", "date")]
    for side in ("home", "away"):
        left = pd.DataFrame({"row": df.index, "team": df[f"{side}_team_id"].to_numpy(), "cut": cut.to_numpy()})
        left = left.sort_values("cut", kind="stable")
        m = pd.merge_asof(left, rolled.sort_values("date", kind="stable"), left_on="cut", right_on="date",
                          by="team", allow_exact_matches=False, direction="backward")  # fmt: skip
        m = m.set_index("row").reindex(df.index)
        for c in stat_cols:
            out[f"{side}_{c}"] = m[c].to_numpy(float)
        out[f"{side}_n_prev"] = out[f"{side}_n_prev"].fillna(0).clip(upper=100)
    return out


def schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario (se publican antes de la temporada): descanso y partidos jugados."""
    long = _long(df)
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest"] = (long["date"] - long["prev_date"]).dt.days.clip(upper=30)
    long["played_season"] = long.groupby(["team", "season"]).cumcount()
    out = pd.DataFrame(index=df.index)
    for side in ("home", "away"):
        sub = long[long["side"] == side].set_index("row").reindex(df.index)
        out[f"{side}_rest"] = sub["rest"].to_numpy(float)
        out[f"{side}_played_season"] = sub["played_season"].to_numpy(float)
    out["league_code"] = df["league"].map(LEAGUE_CODE).to_numpy(float)
    out["tier"] = df["league"].map(TIER).to_numpy(float)
    return out


def rating_features(df: pd.DataFrame, anchor, elo: Elo, pi: PiRating) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    f, h, a = run_online(df, elo, anchor=anchor, return_states=True)
    out["elo_diff"], out["elo_home"], out["elo_away"] = f, h[:, 0], a[:, 0]
    f, h, a = run_online(df, pi, anchor=anchor, return_states=True)
    out["pi_diff"], out["pi_home_h"], out["pi_away_a"] = f, h[:, 0], a[:, 1]
    return out


def dc_features(df: pd.DataFrame, decision: str, cfg: DixonColes, predict_from: pd.Timestamp,
                n_jobs: int = 1) -> pd.DataFrame:  # fmt: skip
    p = dc_walk_forward(df, cfg, predict_from=predict_from, n_jobs=n_jobs, anchor=DC_ANCHORS[decision])
    return p.rename(columns={"p_home": "dc_home", "p_draw": "dc_draw", "p_away": "dc_away"})


def opening_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apertura de Pinnacle sin margen (normalización proporcional). Solo para el ancla de apertura."""
    odds = df[["pin_open_home", "pin_open_draw", "pin_open_away"]].to_numpy(float, copy=True)
    odds[~(odds > 1)] = np.nan
    inv = 1 / odds
    p = inv / inv.sum(axis=1, keepdims=True)
    return pd.DataFrame(p, index=df.index, columns=["open_home", "open_draw", "open_away"])


def build(df: pd.DataFrame, decision: str, *, elo: Elo, pi: PiRating, dc: DixonColes,
          dc_from: pd.Timestamp, n_jobs: int = 1) -> pd.DataFrame:  # fmt: skip
    """Todas las variables de una hora de decisión, alineadas con `df`."""
    anchor = ANCHORS[decision]
    parts = [rating_features(df, anchor, elo, pi), dc_features(df, decision, dc, dc_from, n_jobs),
             form_features(df, anchor), schedule_features(df)]  # fmt: skip
    if decision == "apertura":
        parts.append(opening_features(df))
    out = pd.concat(parts, axis=1)
    check_columns(out.columns)
    return out


def check_columns(columns) -> None:
    """Defensa por nombre: ninguna variable puede venir de cierres, referencias o resultados."""
    bad = [c for c in columns if any(f in c for f in FORBIDDEN)]
    if bad:
        raise ValueError(f"Variables prohibidas: {bad}")
