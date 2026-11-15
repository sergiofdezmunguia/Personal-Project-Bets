"""Métricas de predicción 1X2: log loss, RPS, Brier, calibración e intervalos por bootstrap.

Convención: las probabilidades van en columnas (local, empate, visitante) y el resultado se
codifica 0 = local, 1 = empate, 2 = visitante. El RPS usa ese orden, que es el natural.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME = {"H": 0, "D": 1, "A": 2}
EPS = 1e-15


def outcome_index(result: pd.Series) -> np.ndarray:
    return result.map(OUTCOME).to_numpy(dtype=int)


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pérdida por partido (log natural)."""
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score por partido, para 3 categorías ordenadas."""
    onehot = np.eye(3)[y]
    cum = np.cumsum(p, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]
    return (cum**2).sum(axis=1) / 2


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - np.eye(3)[y]) ** 2).sum(axis=1)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Fiabilidad por resultado: probabilidad media prevista frente a frecuencia observada."""
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for k, name in enumerate(("local", "empate", "visitante")):
        idx = np.clip(np.digitize(p[:, k], edges[1:-1]), 0, bins - 1)
        hit = (y == k).astype(float)
        for b in range(bins):
            m = idx == b
            if m.any():
                rows.append({"resultado": name, "tramo": f"{edges[b]:.1f}–{edges[b + 1]:.1f}",
                             "n": int(m.sum()), "prevista": p[m, k].mean(),
                             "observada": hit[m].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado, media de los tres resultados."""
    t = calibration_table(p, y, bins)
    per = t.groupby("resultado").apply(
        lambda g: (g["n"] * (g["prevista"] - g["observada"]).abs()).sum() / g["n"].sum(),
        include_groups=False,
    )
    return float(per.mean())


def cluster_bootstrap(values: dict[str, np.ndarray], clusters: np.ndarray, *, n_boot: int = 2000,
                      seed: int = 0, level: float = 0.95) -> pd.DataFrame:  # fmt: skip
    """Media e intervalo de cada serie por bootstrap de conglomerados (p. ej. semana de jornada).

    Remuestrea conglomerados completos con reemplazo, porque los partidos de una misma jornada no
    son independientes. Todas las series se remuestrean con los mismos conglomerados, así que las
    diferencias emparejadas (modelo − mercado) conservan su correlación.
    """
    codes, uniq = pd.factorize(pd.Series(clusters))
    n_c = len(uniq)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, n_c))
    for b in range(n_boot):
        counts[b] = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)
    size = np.bincount(codes, minlength=n_c).astype(float)
    denom = counts @ size
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = []
    for name, v in values.items():
        sums = np.bincount(codes, weights=v, minlength=n_c)
        boot = counts @ sums / denom
        out.append({"serie": name, "media": float(v.mean()), "ic_inf": float(np.quantile(boot, lo)),
                    "ic_sup": float(np.quantile(boot, hi))})  # fmt: skip
    return pd.DataFrame(out).set_index("serie")
