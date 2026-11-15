"""Fase 2: modelos de referencia 1X2 con validación por ventanas temporales.

Temporadas (ver `docs/decisions.md`, 2026-11-15):
- 2012/13: arranque de los ratings; no se evalúa.
- 2013/14–2016/17: ajuste de hiperparámetros (rejilla pequeña, todo registrado).
- 2017/18–2024/25: evaluación por ventanas con la configuración elegida en el ajuste.
- 2025/26 y posteriores: prueba final. **No se leen**: `load_history` descarta sus CSV sin abrirlos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import football_data as fd
from src import margin as mg
from src import metrics as mt
from src.dixon_coles import DixonColes
from src.dixon_coles import predict_walk_forward as dc_walk_forward
from src.ratings import Elo, PiRating, run_online, walk_forward_probs

BURN_IN = "2012/13"
TUNING = ["2013/14", "2014/15", "2015/16", "2016/17"]
EVALUATION = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
LAST_ALLOWED_CODE = "2425"  # último fichero de temporada que se puede abrir
PROB = ["p_home", "p_draw", "p_away"]


def load_history(raw_dir: Path) -> pd.DataFrame:
    """Partidos 2012/13–2024/25 con referencia de cierre. Ni siquiera abre los CSV de 2025/26+."""
    frames = []
    for path in sorted(Path(raw_dir).glob("*_*.csv")):
        league, code = path.stem.split("_")
        if code > LAST_ALLOWED_CODE:  # códigos "1213"…"2425" ordenan bien como texto hasta 2099
            continue
        frames.append(fd.normalize(fd.read_raw(path), league, code))
    if not frames:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}")
    df = fd.add_reference(pd.concat(frames, ignore_index=True))
    df = df[df["date"].notna() & df["result"].isin(["H", "D", "A"])]
    if (df["date"] >= fd.FINAL_TEST_FROM).any():
        raise ValueError("Se han colado partidos de la prueba final")
    return df.sort_values(["date", "league", "home_team"], kind="stable").reset_index(drop=True)


def market_probs(df: pd.DataFrame, prefix: str, method: str = "proporcional") -> pd.DataFrame:
    """Probabilidad sin margen de un juego de cuotas 1X2 (proporcional, Shin o potencia;
    ver `src/margin.py`). La proporcional es la de las fases 2 y 3."""
    odds = df[[f"{prefix}_{s}" for s in ("home", "draw", "away")]].to_numpy(float, copy=True)
    return pd.DataFrame(mg.normalize(odds, method), index=df.index, columns=PROB)


def predict(df: pd.DataFrame, variant: str, params: dict, seasons: list[str], n_jobs: int = 1) -> pd.DataFrame:
    """Probabilidades de una variante para las filas de `seasons` (el resto queda en NaN)."""
    mask = df["season"].isin(seasons)
    if variant == "mercado_cierre":
        out = market_probs(df, "pin_close")  # solo Pinnacle: sin recurrir a Betfair ni a la media
    elif variant == "mercado_apertura":
        out = market_probs(df, "pin_open")
    elif variant in ("elo", "pi"):
        model = Elo(**params) if variant == "elo" else PiRating(**params)
        feat = run_online(df, model)
        out = walk_forward_probs(df, feat, seasons, fit_from=TUNING[0])
    elif variant == "dixon_coles":
        start, end = df.loc[mask, "date"].min(), df.loc[mask, "date"].max()
        past = df[df["date"] <= end]  # nada posterior a las temporadas pedidas
        out = dc_walk_forward(past, DixonColes(**params), predict_from=start, n_jobs=n_jobs)
        out = out.reindex(df.index)
    else:
        raise ValueError(f"Variante desconocida: {variant}")
    return out.where(mask, np.nan)


def evaluate(df: pd.DataFrame, preds: dict[str, pd.DataFrame], *, reference: str = "mercado_cierre",
             n_boot: int = 2000) -> dict:  # fmt: skip
    """Métricas en el conjunto común (todas las variantes con probabilidad), con IC por bootstrap
    de semanas y diferencia emparejada frente a la referencia."""
    ok = np.logical_and.reduce([p[PROB].notna().all(axis=1).to_numpy() for p in preds.values()])
    sub = df[ok]
    y = mt.outcome_index(sub["result"])
    week = sub["date"].dt.strftime("%G-%V").to_numpy()
    series, summary = {}, {}
    for name, p in preds.items():
        arr = p.loc[ok, PROB].to_numpy(float)
        series[f"{name}|logloss"] = mt.log_loss(arr, y)
        series[f"{name}|rps"] = mt.rps(arr, y)
        series[f"{name}|brier"] = mt.brier(arr, y)
        summary[name] = {"ece": mt.ece(arr, y)}
    for name in preds:
        if name != reference:
            for m in ("logloss", "rps"):
                series[f"{name}|delta_{m}"] = series[f"{name}|{m}"] - series[f"{reference}|{m}"]
    boot = mt.cluster_bootstrap(series, week, n_boot=n_boot)
    for key, row in boot.iterrows():
        name, metric = key.split("|")
        summary[name][metric] = row["media"]
        summary[name][f"{metric}_ic"] = [row["ic_inf"], row["ic_sup"]]
    any_pred = np.logical_or.reduce([p[PROB].notna().all(axis=1).to_numpy() for p in preds.values()])
    return {"n": int(ok.sum()), "n_descartados": int((any_pred & ~ok).sum()),
            "semanas": int(len(set(week))), "modelos": summary}
