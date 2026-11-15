"""Ratings en línea (Elo y pi-ratings) y su traducción a probabilidades 1X2.

Integridad temporal:
- Los partidos se procesan por día natural. La variable de un partido se calcula con los ratings
  al empezar su día, y los ratings se actualizan solo cuando ha terminado el día entero. Así, dos
  partidos del mismo día nunca se ven entre sí (football-data no tiene hora antes de 2019/20).
- La traducción rating → 1X2 es un logit ordenado que, para predecir la temporada S, se ajusta solo
  con las temporadas anteriores a S (validación por ventanas, walk-forward).

Equipos que entran sin historial (ascienden desde una división no descargada o aparecen en la primera
temporada): empiezan en la media de su liga en ese momento menos `newcomer_offset`. En la primera
temporada no hay medias, así que se usa un valor por categoría (`TIER`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

TIER = {"E0": 1, "E1": 2, "E2": 3, "SP1": 1, "SP2": 2, "D1": 1, "I1": 1, "F1": 1}
REQUIRED = ["date", "season", "league", "home_team_id", "away_team_id", "home_goals", "away_goals"]


# ----------------------------------------------------------------------------- modelos


@dataclass
class Elo:
    """Elo con ventaja local y multiplicador por diferencia de goles (estilo World Football Elo)."""

    k: float = 20.0
    home_adv: float = 60.0
    goal_mult: bool = True
    season_regress: float = 0.0  # fracción de vuelta a la media de su liga al empezar temporada
    newcomer_offset: float = 100.0
    base: float = 1500.0
    tier_gap: float = 150.0
    size: int = field(default=1, init=False)

    def initial(self, league: str) -> np.ndarray:
        return np.array([self.base - self.tier_gap * (TIER.get(league, 1) - 1)])

    def newcomer(self, league_mean: np.ndarray) -> np.ndarray:
        return league_mean - self.newcomer_offset

    def feature(self, h: np.ndarray, a: np.ndarray) -> float:
        return float(h[0] - a[0])

    def update(self, h: np.ndarray, a: np.ndarray, hg: int, ag: int) -> tuple[np.ndarray, np.ndarray]:
        exp_h = 1 / (1 + 10 ** (-(h[0] + self.home_adv - a[0]) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        mult = 1.0
        if self.goal_mult and margin == 2:
            mult = 1.5
        elif self.goal_mult and margin >= 3:
            mult = (11 + margin) / 8
        delta = self.k * mult * (score - exp_h)
        return h + delta, a - delta


@dataclass
class PiRating:
    """Pi-ratings de Constantinou y Fenton (2013): rating en casa y fuera por equipo, en escala de
    diferencia de goles esperada. Estado = [rating en casa, rating fuera]."""

    lam: float = 0.035  # aprendizaje del rating del lado jugado
    gamma: float = 0.7  # cuánto se traslada al otro lado
    b: float = 10.0
    c: float = 3.0
    season_regress: float = 0.0
    newcomer_offset: float = 0.2
    tier_gap: float = 0.3
    size: int = field(default=2, init=False)

    def initial(self, league: str) -> np.ndarray:
        return np.full(2, -self.tier_gap * (TIER.get(league, 1) - 1))

    def newcomer(self, league_mean: np.ndarray) -> np.ndarray:
        return league_mean - self.newcomer_offset

    def _goals(self, r: float) -> float:
        return float(np.sign(r) * (self.b ** (abs(r) / self.c) - 1))

    def feature(self, h: np.ndarray, a: np.ndarray) -> float:
        return self._goals(h[0]) - self._goals(a[1])

    def update(self, h: np.ndarray, a: np.ndarray, hg: int, ag: int) -> tuple[np.ndarray, np.ndarray]:
        err = (hg - ag) - self.feature(h, a)
        psi = self.c * np.log10(1 + abs(err)) * np.sign(err)
        h, a = h.copy(), a.copy()
        dh = psi * self.lam
        h[0] += dh
        h[1] += dh * self.gamma
        da = -psi * self.lam
        a[1] += da
        a[0] += da * self.gamma
        return h, a


# ----------------------------------------------------------------------------- motor en línea


def check_input(df: pd.DataFrame, final_test_from: pd.Timestamp | None = None) -> None:
    from src.football_data import FINAL_TEST_FROM

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    limit = final_test_from or FINAL_TEST_FROM
    if (df["date"] >= limit).any():
        raise ValueError("Hay partidos de la prueba final (2025/26 o posterior): no se usan hasta congelar")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Los partidos deben venir ordenados por fecha")


def run_online(df: pd.DataFrame, model) -> np.ndarray:
    """Variable previa al partido para cada fila de `df` (mismo orden). Solo usa días anteriores."""
    check_input(df)
    state: dict[str, np.ndarray] = {}
    last_league: dict[str, str] = {}
    last_season: dict[str, str] = {}
    snapshot: dict[str, dict[str, np.ndarray]] = {}  # temporada -> media por liga al empezar
    seen_seasons: set[str] = set()
    feat = np.empty(len(df))

    def league_means() -> dict[str, np.ndarray]:
        groups: dict[str, list[np.ndarray]] = {}
        for t, lg in last_league.items():
            groups.setdefault(lg, []).append(state[t])
        return {lg: np.mean(v, axis=0) for lg, v in groups.items()}

    cols = df[["date", "season", "league", "home_team_id", "away_team_id", "home_goals", "away_goals"]]
    rows = list(cols.itertuples(index=False, name=None))
    dates = df["date"].to_numpy()
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and dates[end] == dates[start]:
            end += 1
        day = rows[start:end]
        for season in {r[1] for r in day} - seen_seasons:  # primera vez que aparece la temporada
            snapshot[season] = league_means()
            seen_seasons.add(season)
        # 1) estado previo de cada equipo (alta o regreso a la media al cambiar de temporada)
        for _, season, league, home, away, _, _ in day:
            for team in (home, away):
                if team not in state:
                    means = league_means()
                    state[team] = model.newcomer(means[league]) if league in means else model.initial(league)
                elif last_season[team] != season and model.season_regress:
                    m = snapshot[season].get(last_league[team])
                    if m is not None:
                        state[team] = m + (1 - model.season_regress) * (state[team] - m)
                last_season[team] = season
        # 2) predicción de todos los partidos del día con el estado previo
        for i, (_, _, _, home, away, _, _) in enumerate(day):
            feat[start + i] = model.feature(state[home], state[away])
        # 3) actualización con los resultados del día, partiendo todos del estado previo
        new = {}
        for _, _, league, home, away, hg, ag in day:
            h, a = model.update(new.get(home, state[home]), new.get(away, state[away]), int(hg), int(ag))
            new[home], new[away] = h, a
            last_league[home] = last_league[away] = league
        state.update(new)
        start = end
    return feat


# ----------------------------------------------------------------------------- rating → 1X2


@dataclass
class OrderedLogit:
    """P(visitante) = σ(c1 − βx); P(visitante o empate) = σ(c2 − βx); c2 > c1."""

    params: np.ndarray | None = None

    @staticmethod
    def _probs(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        c1, gap, beta = theta
        c2 = c1 + np.exp(gap)
        pa = expit(c1 - beta * x)
        pad = expit(c2 - beta * x)
        return np.column_stack([1 - pad, pad - pa, pa])

    def fit(self, x: np.ndarray, y: np.ndarray) -> OrderedLogit:
        scale = np.std(x) or 1.0

        def nll(theta):
            p = self._probs(theta, x / scale)
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

        res = minimize(nll, np.array([-1.0, 0.0, 0.5]), method="BFGS")
        c1, gap, beta = res.x
        self.params = np.array([c1, gap, beta / scale])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._probs(self.params, x)


def walk_forward_probs(df: pd.DataFrame, feature: np.ndarray, eval_seasons: list[str],
                       fit_from: str) -> pd.DataFrame:  # fmt: skip
    """Probabilidades 1X2 por temporada con un logit ordenado ajustado solo en temporadas previas
    (desde `fit_from`; si no hay ninguna, todas las anteriores)."""
    from src.metrics import outcome_index

    y = outcome_index(df["result"])
    season = df["season"].to_numpy()
    out = pd.DataFrame(np.nan, index=df.index, columns=["p_home", "p_draw", "p_away"])
    for s in eval_seasons:
        train = (season < s) & (season >= fit_from)
        if not train.any():
            train = season < s
        test = season == s
        if not train.any() or not test.any():
            continue
        m = OrderedLogit().fit(feature[train], y[train])
        out.loc[test, :] = m.predict(feature[test])
    return out
