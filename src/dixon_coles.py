"""Dixon-Coles (1997) con decaimiento temporal, reajustado cada semana.

Modelo por país (las divisiones de un país se enlazan por ascensos y descensos):

    log λ_local     = μ + h + ataque_i + defensa_j
    log λ_visitante = μ +     ataque_j + defensa_i

con la corrección τ(ρ) de Dixon-Coles para 0-0, 1-0, 0-1 y 1-1, y peso exp(−ξ·días) por antigüedad.
Cada equipo se encoge hacia la media de su liga (A_L, D_L, libres) con fuerza `ridge`; un equipo
que llegó hace menos de un año se encoge hacia esa media desplazada por `newcomer` (más débil).

Integridad temporal: para predecir los partidos de la semana que empieza el lunes t, se ajusta solo
con partidos de fecha < t (y como mucho `window_days` atrás). Ningún partido de la semana entra en
su propio ajuste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.ratings import check_input

COUNTRY_POOL = {"E0": "ENG", "E1": "ENG", "E2": "ENG", "SP1": "ESP", "SP2": "ESP",
                "D1": "GER", "I1": "ITA", "F1": "FRA"}  # fmt: skip
MAX_GOALS = 10


@dataclass
class DixonColes:
    xi: float = 0.002  # decaimiento por día (vida media ≈ 347 días)
    ridge: float = 5.0
    newcomer: float = 0.15  # ataque −newcomer y defensa +newcomer para recién llegados
    window_days: int = 4 * 365
    newcomer_days: int = 365


def _unpack(theta, n_t, n_l):
    mu, home, rho = theta[:3]
    a = theta[3 : 3 + n_t]
    d = theta[3 + n_t : 3 + 2 * n_t]
    A = theta[3 + 2 * n_t : 3 + 2 * n_t + n_l]
    D = theta[3 + 2 * n_t + n_l :]
    return mu, home, rho, a, d, A, D


def _nll_grad(theta, hi, ai, x, y, w, team_league, new_flag, cfg: DixonColes, n_t, n_l):
    mu, home, rho, a, d, A, D = _unpack(theta, n_t, n_l)
    eh = mu + home + a[hi] + d[ai]
    ea = mu + a[ai] + d[hi]
    lh, la = np.exp(eh), np.exp(ea)
    tau = np.ones_like(lh)
    g_eh, g_ea, g_rho = np.zeros_like(lh), np.zeros_like(lh), np.zeros_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m11] = 1 - rho
    clipped = tau < 1e-10
    tau = np.maximum(tau, 1e-10)
    g_eh[m00] = -lh[m00] * la[m00] * rho / tau[m00]
    g_ea[m00] = g_eh[m00]
    g_rho[m00] = -lh[m00] * la[m00] / tau[m00]
    g_eh[m01] = lh[m01] * rho / tau[m01]
    g_rho[m01] = lh[m01] / tau[m01]
    g_ea[m10] = la[m10] * rho / tau[m10]
    g_rho[m10] = la[m10] / tau[m10]
    g_rho[m11] = -1 / tau[m11]
    g_eh[clipped] = g_ea[clipped] = g_rho[clipped] = 0  # en la zona recortada log τ es constante

    ll = w * (x * eh - lh + y * ea - la + np.log(tau))
    deh = w * (x - lh + g_eh)
    dea = w * (y - la + g_ea)
    drho = (w * g_rho).sum()

    prior_a = A[team_league] - cfg.newcomer * new_flag
    prior_d = D[team_league] + cfg.newcomer * new_flag
    ra, rd = a - prior_a, d - prior_d
    tiny = 1e-3
    pen = 0.5 * cfg.ridge * ((ra**2).sum() + (rd**2).sum()) + 0.5 * tiny * ((A**2).sum() + (D**2).sum())

    ga = np.bincount(hi, deh, n_t) + np.bincount(ai, dea, n_t) - cfg.ridge * ra
    gd = np.bincount(ai, deh, n_t) + np.bincount(hi, dea, n_t) - cfg.ridge * rd
    gA = np.bincount(team_league, cfg.ridge * ra, n_l) - tiny * A
    gD = np.bincount(team_league, cfg.ridge * rd, n_l) - tiny * D
    grad = np.concatenate([[deh.sum() + dea.sum(), deh.sum(), drho], ga, gd, gA, gD])
    scale = w.sum()
    return -(ll.sum() - pen) / scale, -grad / scale


def score_probs(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Probabilidades 1X2 a partir de las medias de goles, con la corrección τ."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lh[:, None])
    pa = poisson.pmf(g[None, :], la[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= np.maximum(1 - lh * la * rho, 0)
    m[:, 0, 1] *= np.maximum(1 + lh * rho, 0)
    m[:, 1, 0] *= np.maximum(1 + la * rho, 0)
    m[:, 1, 1] *= max(1 - rho, 0)
    home = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1).astype(bool)  # filas = goles local
    out = np.column_stack([m[:, home].sum(1), m[:, np.eye(MAX_GOALS + 1, dtype=bool)].sum(1),
                           m[:, home.T].sum(1)])  # fmt: skip
    return out / out.sum(axis=1, keepdims=True)


def predict_walk_forward(df: pd.DataFrame, cfg: DixonColes, predict_from: pd.Timestamp,
                         n_jobs: int = 1) -> pd.DataFrame:  # fmt: skip
    """Probabilidades 1X2 para los partidos con fecha ≥ `predict_from`, reajustando cada lunes.
    Cada país se ajusta por separado (en paralelo si `n_jobs` > 1)."""
    check_input(df)
    out = pd.DataFrame(np.nan, index=df.index, columns=["p_home", "p_draw", "p_away"])
    pools = [g for _, g in df.groupby(df["league"].map(COUNTRY_POOL), sort=True)]
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(n_jobs, len(pools))) as ex:
            parts = list(ex.map(_predict_pool, pools, [cfg] * len(pools), [predict_from] * len(pools)))
    else:
        parts = [_predict_pool(g, cfg, predict_from) for g in pools]
    for part in parts:
        out.loc[part.index, :] = part.to_numpy()
    return out


def _predict_pool(g: pd.DataFrame, cfg: DixonColes, predict_from: pd.Timestamp) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=g.index, columns=["p_home", "p_draw", "p_away"])
    dates = g["date"]
    weeks = (dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")).dt.normalize()
    first_seen = pd.concat([g[["date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                            g[["date", "away_team_id"]].rename(columns={"away_team_id": "t"})]
                           ).groupby("t")["date"].min()  # fmt: skip
    leagues = sorted(g["league"].unique())
    lg_idx = {lg: i for i, lg in enumerate(leagues)}
    warm: dict[str, tuple[float, float]] = {}
    glob = np.array([0.3, 0.25, -0.05])
    for monday in sorted(weeks[dates >= predict_from].unique()):
        monday = pd.Timestamp(monday)
        target = g[(weeks == monday) & (dates >= predict_from)]
        train = g[(dates < monday) & (dates >= monday - pd.Timedelta(days=cfg.window_days))]
        if len(train) < 50:
            continue
        teams = sorted(set(train["home_team_id"]) | set(train["away_team_id"]))
        t_idx = {t: i for i, t in enumerate(teams)}
        n_t, n_l = len(teams), len(leagues)
        # liga actual de cada equipo: la de su último partido antes del lunes
        last = pd.concat([train[["date", "home_team_id", "league"]].rename(columns={"home_team_id": "t"}),
                          train[["date", "away_team_id", "league"]].rename(columns={"away_team_id": "t"})]
                         ).sort_values("date").groupby("t")["league"].last()  # fmt: skip
        team_league = np.array([lg_idx[last[t]] for t in teams])
        new_flag = np.array([(monday - first_seen[t]).days < cfg.newcomer_days for t in teams], float)
        hi = train["home_team_id"].map(t_idx).to_numpy()
        ai = train["away_team_id"].map(t_idx).to_numpy()
        x = train["home_goals"].to_numpy(float)
        y = train["away_goals"].to_numpy(float)
        w = np.exp(-cfg.xi * (monday - train["date"]).dt.days.to_numpy(float))

        theta0 = np.concatenate([glob, [warm.get(t, (0, 0))[0] for t in teams],
                                 [warm.get(t, (0, 0))[1] for t in teams], np.zeros(2 * n_l)])  # fmt: skip
        bounds = [(None, None), (None, None), (-0.3, 0.3)] + [(None, None)] * (2 * n_t + 2 * n_l)
        res = minimize(_nll_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       args=(hi, ai, x, y, w, team_league, new_flag, cfg, n_t, n_l))  # fmt: skip
        mu, home, rho, a, d, A, D = _unpack(res.x, n_t, n_l)
        glob = res.x[:3]
        warm.update({t: (a[i], d[i]) for t, i in t_idx.items()})

        def params(team: str, league: str) -> tuple[float, float]:
            if team in t_idx:
                return a[t_idx[team]], d[t_idx[team]]
            li = lg_idx[league]  # sin datos: media de la liga en que va a jugar, como recién llegado
            return A[li] - cfg.newcomer, D[li] + cfg.newcomer

        ph = np.array([params(t, lg) for t, lg in zip(target["home_team_id"], target["league"], strict=True)])
        pa = np.array([params(t, lg) for t, lg in zip(target["away_team_id"], target["league"], strict=True)])
        lh = np.exp(mu + home + ph[:, 0] + pa[:, 1])
        la = np.exp(mu + pa[:, 0] + ph[:, 1])
        out.loc[target.index, :] = score_probs(lh, la, rho)
    return out
