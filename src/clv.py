"""Fase 4: CLV (closing line value) de apuestas simuladas a partir de instantáneas de cuotas.

Una **instantánea** es una cuota vista en un momento: partido, mercado, casa, resultado, cuota y
`captured_utc`, la hora UTC a la que *nosotros* la vimos por primera vez (formato en `SNAP_COLS`,
lo escribe `src/odds_collector.py`). Con ellas:

1. **Precio al decidir** (`latest_before`): para cada partido, la última cuota con
   `captured_utc ≤ hora de decisión` y anterior al saque. Nada posterior a la decisión se lee.
2. **Decisión** (`decide`): se apuesta 1 unidad al resultado de mayor valor esperado
   p_modelo · cuota_efectiva − 1 si supera un umbral fijado de antemano. La cuota efectiva
   descuenta la comisión sobre la ganancia (exchange): 1 + (cuota − 1)(1 − c).
3. **Cierre** (`closing`): la última cuota de la casa de referencia antes del saque.
4. **CLV** (`clv`):
   - `clv_ev` = cuota_efectiva · p_cierre_sin_margen − 1: valor esperado de la apuesta si el
     cierre sin margen es la probabilidad verdadera. Es la métrica de la puerta de la fase 4.
   - `clv_log` = ln(cuota tomada / cuota de cierre bruta del mismo resultado y casa de cierre).

Ninguna función coloca apuestas ni habla con casas: solo simula sobre datos ya guardados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import margin as mg
from src import metrics as mt

SNAP_COLS = ["source", "event_key", "league", "kickoff_utc", "home_team", "away_team", "market", "line",
             "bookmaker", "outcome", "kind", "price", "source_updated_utc", "captured_utc"]  # fmt: skip
OUTCOMES = ["home", "draw", "away"]
PROB = ["p_home", "p_draw", "p_away"]
_KEY = ["event_key", "market", "line", "bookmaker", "outcome", "kind"]


def check_snapshots(snaps: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SNAP_COLS if c not in snaps]
    if missing:
        raise ValueError(f"Faltan columnas en las instantáneas: {missing}")
    out = snaps.copy()
    for c in ("kickoff_utc", "captured_utc"):
        out[c] = pd.to_datetime(out[c], utc=True)
    out["line"] = out["line"].fillna("")
    return out


def latest_before(snaps: pd.DataFrame, cutoff: pd.Series, *, strict: bool = False) -> pd.DataFrame:
    """Última instantánea de cada (partido, mercado, línea, casa, resultado, tipo) vista antes de
    `cutoff[event_key]` (≤, o < si `strict`) y siempre antes del saque."""
    s = check_snapshots(snaps)
    cut = pd.to_datetime(cutoff, utc=True).rename("cutoff")
    s = s.merge(cut, left_on="event_key", right_index=True, how="inner")
    seen = s["captured_utc"] < s["cutoff"] if strict else s["captured_utc"] <= s["cutoff"]
    s = s[seen & (s["captured_utc"] < s["kickoff_utc"])]
    s = s.sort_values("captured_utc", kind="stable")
    return s.groupby(_KEY, sort=False, dropna=False).tail(1).drop(columns="cutoff")


def closing(snaps: pd.DataFrame) -> pd.DataFrame:
    """Cierre: última instantánea antes del saque, con los minutos que faltaban (`gap_min`)."""
    s = check_snapshots(snaps)
    ko = s.groupby("event_key")["kickoff_utc"].first()
    out = latest_before(s, ko, strict=True)
    out["gap_min"] = (out["kickoff_utc"] - out["captured_utc"]).dt.total_seconds() / 60
    return out


def wide_1x2(rows: pd.DataFrame, bookmaker: str, kind: str = "back") -> pd.DataFrame:
    """Cuotas 1X2 de una casa en columnas home/draw/away (más la hora de la instantánea)."""
    r = rows[(rows["market"] == "1x2") & (rows["bookmaker"] == bookmaker) & (rows["kind"] == kind)]
    w = r.pivot_table(index="event_key", columns="outcome", values="price", aggfunc="last")
    w = w.reindex(columns=OUTCOMES)
    w["captured_utc"] = r.groupby("event_key")["captured_utc"].max()
    return w.dropna(subset=OUTCOMES)


def effective(price, commission: float = 0.0):
    """Cuota neta de comisión sobre la ganancia (exchange). Con c = 0 es la misma cuota."""
    return 1 + (np.asarray(price, float) - 1) * (1 - commission)


def decide(preds: pd.DataFrame, prices: pd.DataFrame, *, threshold: float, commission: float = 0.0) -> pd.DataFrame:
    """Una apuesta de 1 unidad por partido como máximo: el resultado con más valor esperado si
    supera `threshold`. `preds` va indexado por event_key con PROB; `prices` con OUTCOMES."""
    common = preds.index.intersection(prices.index)
    p = preds.loc[common, PROB].to_numpy(float)
    o = prices.loc[common, OUTCOMES].to_numpy(float)
    ev = p * effective(o, commission) - 1
    ok = np.isfinite(ev).all(axis=1)
    best = np.where(ok, np.nanargmax(np.where(np.isfinite(ev), ev, -np.inf), axis=1), 0)
    best_ev = ev[np.arange(len(ev)), best]
    take = ok & (best_ev > threshold)
    idx = np.flatnonzero(take)
    return pd.DataFrame({
        "event_key": common[idx],
        "outcome": np.array(OUTCOMES)[best[idx]],
        "price": o[idx, best[idx]],
        "p_model": p[idx, best[idx]],
        "ev": best_ev[idx],
        "price_captured_utc": prices.loc[common[idx], "captured_utc"].to_numpy(),
    })  # fmt: skip


def clv(bets: pd.DataFrame, close: pd.DataFrame, *, method: str, commission: float = 0.0) -> pd.DataFrame:
    """Añade el CLV de cada apuesta frente al cierre (`close`: OUTCOMES por event_key)."""
    out = bets[bets["event_key"].isin(close.index)].copy()
    c = close.loc[out["event_key"], OUTCOMES].to_numpy(float)
    fair = mg.normalize(c, method)
    k = out["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(int)
    rows = np.arange(len(out))
    out["close_price"] = c[rows, k]
    out["p_close"] = fair[rows, k]
    out["clv_ev"] = effective(out["price"], commission) * out["p_close"] - 1
    out["clv_log"] = np.log(out["price"].to_numpy(float) / out["close_price"].to_numpy(float))
    return out


def simulate(snaps: pd.DataFrame, preds: pd.DataFrame, *, bet_book: str, close_book: str, threshold: float,
             method: str, commission: float = 0.0,
             max_close_gap_min: float | None = None) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """De extremo a extremo: precio al decidir, decisión y CLV frente al cierre.

    `preds`: indexado por event_key, con PROB y `decision_utc` (la hora a la que existía la
    predicción). Las predicciones hechas al saque o después se descartan y se cuentan. Con
    `max_close_gap_min`, un «cierre» visto más de esos minutos antes del saque no cuenta como cierre
    (la apuesta queda `sin_cierre`)."""
    s = check_snapshots(snaps)
    ko = s.groupby("event_key")["kickoff_utc"].first()
    dec = pd.to_datetime(preds["decision_utc"], utc=True)
    late = dec.index.isin(ko.index) & (dec >= ko.reindex(dec.index)).to_numpy()
    p = preds[~late]
    at = latest_before(s, pd.to_datetime(p["decision_utc"], utc=True))
    prices = wide_1x2(at, bet_book)
    bets = decide(p, prices, threshold=threshold, commission=commission)
    last = closing(s)
    if max_close_gap_min is not None:
        last = last[last["gap_min"] <= max_close_gap_min]
    close = wide_1x2(last, close_book)
    if len(bets) and (pd.to_datetime(bets["price_captured_utc"], utc=True).to_numpy()
                      > pd.to_datetime(p.loc[bets["event_key"], "decision_utc"], utc=True).to_numpy()).any():  # fmt: skip
        raise AssertionError("Una cuota usada es posterior a la decisión")
    out = clv(bets, close, method=method, commission=commission)
    info = {"predicciones": int(len(preds)), "tardias": int(late.sum()), "con_precio": int(len(prices)),
            "apuestas": int(len(bets)), "sin_cierre": int(len(bets) - len(out))}  # fmt: skip
    return out, info


def summarize(bets: pd.DataFrame, clusters: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
              won: np.ndarray | None = None, commission: float = 0.0) -> dict:  # fmt: skip
    """Media del CLV con IC 95 % por bootstrap de conglomerados (semanas) y, si se da el
    resultado (`won`), el ROI de apuestas planas de 1 unidad."""
    if len(bets) == 0:
        return {"n": 0}
    series = {"clv_ev": bets["clv_ev"].to_numpy(float), "clv_log": bets["clv_log"].to_numpy(float)}
    if won is not None:
        series["roi"] = np.where(won, effective(bets["price"], commission) - 1, -1.0)
    b = mt.cluster_bootstrap(series, clusters, n_boot=n_boot, seed=seed)
    out = {"n": int(len(bets)), "semanas": int(len(set(clusters))),
           "clv_pos": float((bets["clv_ev"] > 0).mean()), "ev_medio": float(bets["ev"].mean())}  # fmt: skip
    for k, row in b.iterrows():
        out[k] = row["media"]
        out[f"{k}_ic"] = [row["ic_inf"], row["ic_sup"]]
    return out


def history_snapshots(df: pd.DataFrame, open_anchor: pd.Series, *, open_prefix: str = "pin_open",
                      close_prefix: str = "pin_close", book: str = "pinnacle") -> pd.DataFrame:  # fmt: skip
    """Convierte el histórico de football-data (apertura y cierre) en instantáneas con un reloj
    sintético, para pasar por el mismo código que las cuotas propias:

    - apertura vista a las 00:00 UTC del día ancla (martes o viernes, `features.opening_anchor`);
    - cierre visto a las 23:58 del día del partido y saque a las 23:59 (antes de 2019/20 no hay
      hora de saque, así que solo se respeta el orden apertura → cierre → saque).
    """
    day = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize("UTC")
    anchor = pd.to_datetime(open_anchor).dt.normalize().dt.tz_localize("UTC")
    base = pd.DataFrame({"source": "football-data-historico", "event_key": df["match_id"].to_numpy(),
                         "league": df["league"].to_numpy(), "kickoff_utc": (day + pd.Timedelta(minutes=1439)).to_numpy(),
                         "home_team": df["home_team"].to_numpy(), "away_team": df["away_team"].to_numpy(),
                         "market": "1x2", "line": "", "bookmaker": book, "kind": "back"})  # fmt: skip
    parts = []
    for prefix, seen in ((open_prefix, anchor), (close_prefix, day + pd.Timedelta(minutes=1438))):
        for o in OUTCOMES:
            p = base.copy()
            p["outcome"] = o
            p["price"] = df[f"{prefix}_{o}"].to_numpy(float)
            p["source_updated_utc"] = pd.NaT
            p["captured_utc"] = seen.to_numpy()
            parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    return out[out["price"] > 1][SNAP_COLS].reset_index(drop=True)
