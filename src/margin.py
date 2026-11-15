"""Normalización del margen: de cuotas decimales a probabilidades que suman 1 (fase 4).

Tres métodos, sin parámetros libres (cada partido resuelve su propio parámetro):

- **proporcional:** p_i = q_i / Σq, con q_i = 1/cuota. Reparte el margen en proporción a la
  probabilidad, así que sobrestima las sorpresas (el margen real se carga más en ellas).
- **shin:** modelo de Shin (1993) con una fracción z de apostantes con información privilegiada.
  p_i = (√(z² + 4(1−z)·q_i²/Σq) − z) / (2(1−z)), con z tal que Σp = 1 (Jullien y Salanié 2000;
  Štrumbelj 2014). Carga más margen en las cuotas altas.
- **potencia:** p_i = q_i^k con k ≥ 1 tal que Σp = 1 (Vovk y Zhdanov 2009; Clarke et al. 2017).

Las filas con alguna cuota ≤ 1 o ausente dan NaN. Si Σq ≤ 1 (sin margen, o arbitraje) los tres
métodos devuelven la proporcional (z = 0, k = 1).
"""

from __future__ import annotations

import numpy as np

METHODS = ("proporcional", "shin", "potencia")
_ITER = 80  # bisección: 2^-80 de anchura final, muy por debajo del redondeo de las cuotas


def _implied(odds) -> tuple[np.ndarray, np.ndarray]:
    o = np.array(odds, dtype=float, copy=True)
    if o.ndim == 1:
        o = o[None, :]
    bad = ~(o > 1).all(axis=1)
    o[bad] = np.nan
    return 1.0 / o, bad


def proportional(odds) -> np.ndarray:
    q, _ = _implied(odds)
    return q / q.sum(axis=1, keepdims=True)


def _shin_p(q: np.ndarray, B: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = z[:, None]
    return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / B[:, None]) - z) / (2 * (1 - z))


def shin(odds, return_param: bool = False):
    """Probabilidades de Shin y, si se pide, la z de cada fila."""
    q, bad = _implied(odds)
    B = q.sum(axis=1)
    with_margin = ~bad & (B > 1)
    lo, hi = np.zeros(len(q)), np.full(len(q), 0.999)
    for _ in range(_ITER):  # Σp(z) decrece con z: Σp(0) = √B > 1
        mid = (lo + hi) / 2
        s = _shin_p(q, np.where(with_margin, B, 1.0), mid).sum(axis=1)
        go_up = s > 1
        lo, hi = np.where(go_up, mid, lo), np.where(go_up, hi, mid)
    z = np.where(with_margin, (lo + hi) / 2, 0.0)
    p = np.where(with_margin[:, None], _shin_p(q, np.where(with_margin, B, 1.0), z), q / B[:, None])
    p = p / p.sum(axis=1, keepdims=True)  # quita el error residual de la bisección
    z[bad] = np.nan
    return (p, z) if return_param else p


def power(odds, return_param: bool = False):
    """Probabilidades de potencia y, si se pide, la k de cada fila."""
    q, bad = _implied(odds)
    B = q.sum(axis=1)
    with_margin = ~bad & (B > 1)
    lq = np.log(np.where(np.isfinite(q), q, 0.5))
    lo, hi = np.ones(len(q)), np.full(len(q), 50.0)
    for _ in range(_ITER):  # Σq^k decrece con k porque q < 1
        mid = (lo + hi) / 2
        s = np.exp(lq * mid[:, None]).sum(axis=1)
        go_up = s > 1
        lo, hi = np.where(go_up, mid, lo), np.where(go_up, hi, mid)
    k = np.where(with_margin, (lo + hi) / 2, 1.0)
    p = np.where(with_margin[:, None], np.exp(lq * k[:, None]), q / B[:, None])
    p = p / p.sum(axis=1, keepdims=True)
    p[bad] = np.nan
    k[bad] = np.nan
    return (p, k) if return_param else p


def normalize(odds, method: str = "proporcional") -> np.ndarray:
    if method == "proporcional":
        return proportional(odds)
    if method == "shin":
        return shin(odds)
    if method == "potencia":
        return power(odds)
    raise ValueError(f"Método de normalización desconocido: {method}")
