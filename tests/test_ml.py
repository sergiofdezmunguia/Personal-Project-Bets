"""Fase 3: variables y modelos de ML, con datos inventados.

Propiedad antifuga: con la hora de decisión cuya ancla es A(D), cambiar los resultados y
estadísticas de los partidos con fecha ≥ c no cambia nada (variables ni predicciones) de los
partidos con A(D) ≤ c. Con la apertura eso incluye partidos posteriores a c (el desfase entre la
recogida de cuotas y el partido). Cada comprobación se valida con un **mutante**: una versión con una
fuga deliberada que la comprobación debe detectar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as ft
from src import ml
from src.dixon_coles import DixonColes
from src.ratings import Elo, PiRating
from tests.test_referencias import synthetic

DC_FROM = pd.Timestamp("2013-08-01")
SMALL = {"lgbm": {"leaves": 4, "rounds": 30, "lr": 0.1, "min_leaf": 20},
         "catboost": {"depth": 3, "iterations": 40, "lr": 0.1},
         "residuo_lgbm": {"leaves": 4, "rounds": 30, "lr": 0.05, "min_leaf": 20},
         "residuo_logit": {"alpha": 1e-3}}  # fmt: skip
STATS = ["home_goals", "away_goals", "home_shots", "away_shots", "home_shots_target",
         "away_shots_target", "home_corners", "away_corners"]  # fmt: skip


def with_market(df: pd.DataFrame, seed: int = 4) -> pd.DataFrame:
    """Añade estadísticas del partido y cuotas de apertura y cierre inventadas."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    for side in ("home", "away"):
        g = df[f"{side}_goals"].to_numpy()
        df[f"{side}_shots"] = rng.poisson(9 + 2 * g)
        df[f"{side}_shots_target"] = np.minimum(df[f"{side}_shots"], rng.poisson(2 + g))
        df[f"{side}_corners"] = rng.poisson(5, n)
    logits = np.column_stack([0.3 + rng.normal(0, 0.5, n), np.full(n, -0.1), rng.normal(0, 0.5, n)])
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    for k, s in enumerate(("home", "draw", "away")):
        df[f"pin_close_{s}"] = 1 / (p[:, k] * 1.03)
        df[f"pin_open_{s}"] = df[f"pin_close_{s}"] * np.exp(rng.normal(0, 0.04, n))
        df[f"ref_prob_{s}"] = p[:, k]
    return df


def scramble(df: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """Resultados y estadísticas nuevos al azar en los partidos con fecha ≥ cutoff."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    late = out["date"] >= cutoff
    for c in STATS:
        out.loc[late, c] = rng.integers(0, 6 if "goals" in c else 20, late.sum())
    hg, ag = out["home_goals"], out["away_goals"]
    out["result"] = np.select([hg > ag, hg == ag], ["H", "D"], "A")
    return out


def build(df, decision):
    return ft.build(df, decision, elo=Elo(), pi=PiRating(), dc=DixonColes(), dc_from=DC_FROM)


def visible_mask(df, decision, cutoff):
    return (ft.ANCHORS[decision](df["date"]) <= cutoff).to_numpy()


def assert_unchanged(fn, df, decision, cutoff):
    """`fn(df)` → DataFrame alineado con df. Lo visible en el ancla no debe cambiar."""
    base, alt = fn(df), fn(scramble(df, cutoff))
    m = visible_mask(df, decision, cutoff) & base.notna().any(axis=1).to_numpy()
    assert m.sum() > 0
    np.testing.assert_allclose(alt.to_numpy(float)[m], base.to_numpy(float)[m], rtol=0, atol=1e-9)
    return m


def cutoffs(df, qs=(0.4, 0.6, 0.8, 0.95)):
    days = np.sort(df["date"].unique())
    return [pd.Timestamp(days[int(q * len(days))]) for q in qs]


@pytest.fixture(scope="module")
def df():
    return with_market(synthetic(step_days=2))  # jornadas cada 2 días: caen en todos los días


# ----------------------------------------------------------------------------- anclas


def test_opening_anchor_follows_the_odds_collection_days():
    days = pd.Series(pd.date_range("2024-01-01", periods=14))  # lunes 1-ene
    a = ft.opening_anchor(days)
    names = dict(zip(days.dt.day_name(), a.dt.day_name(), strict=False))
    assert names == {"Monday": "Friday", "Tuesday": "Tuesday", "Wednesday": "Tuesday", "Thursday": "Tuesday",
                     "Friday": "Friday", "Saturday": "Friday", "Sunday": "Friday"}  # fmt: skip
    assert (a <= days).all() and a.is_monotonic_increasing
    assert (ft.kickoff_anchor(days) == days).all()


# ----------------------------------------------------------------------------- variables


@pytest.mark.parametrize("decision", ["saque", "apertura"])
def test_features_only_use_results_before_the_anchor(df, decision):
    lagged = 0
    for c in cutoffs(df):
        m = assert_unchanged(lambda d: build(d, decision), df, decision, c)
        lagged += int((m & (df["date"] > c).to_numpy()).sum())
    if decision == "apertura":
        assert lagged > 0  # la comprobación cubre partidos posteriores al corte


def test_opening_features_with_kickoff_anchor_are_caught(df, monkeypatch):
    """Mutante: la apertura usando todo lo anterior al día del partido (ve el partido del viernes
    para decidir el lunes)."""
    monkeypatch.setitem(ft.ANCHORS, "apertura", ft.kickoff_anchor)
    monkeypatch.setitem(ft.DC_ANCHORS, "apertura", ft.kickoff_anchor)
    leaky = lambda d: build(d, "apertura")  # noqa: E731
    days = [pd.Timestamp(x) for x in np.sort(df.loc[df["season"] == "2014/15", "date"].unique())]
    base = leaky(df)
    with pytest.raises(AssertionError):
        for c in days:  # basta con que un corte caiga entre el ancla y el partido
            alt = leaky(scramble(df, c))
            m = (ft.opening_anchor(df["date"]) <= c).to_numpy() & base.notna().any(axis=1).to_numpy()
            np.testing.assert_allclose(alt.to_numpy(float)[m], base.to_numpy(float)[m], atol=1e-9)


def leaky_form_ungrouped(df, anchor):
    """Mutante: media móvil sobre la tabla ordenada por fecha, sin agrupar por equipo."""
    out = ft.form_features(df, anchor)
    roll = df["home_goals"].rolling(5, min_periods=1).mean()
    out["home_gf5"] = roll.to_numpy()
    return out


def leaky_form_includes_match(df, anchor):
    """Mutante: la consulta as-of admite la fecha del ancla (incluye el propio día)."""
    long = ft._long(df)
    long["gf5"] = long.groupby("team")["gf"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    left = pd.DataFrame({"row": df.index, "team": df["home_team_id"], "cut": anchor(df["date"])})
    m = pd.merge_asof(left.sort_values("cut"), long.drop(columns="row").sort_values("date"), left_on="cut", right_on="date",
                      by="team", allow_exact_matches=True).set_index("row").reindex(df.index)  # fmt: skip
    return pd.DataFrame({"home_gf5": m["gf5"].to_numpy()}, index=df.index)


@pytest.mark.parametrize("mutant", [leaky_form_ungrouped, leaky_form_includes_match], ids=["sin_agrupar", "incluye_dia"])
def test_form_mutants_are_caught(df, mutant):
    with pytest.raises(AssertionError):
        for c in cutoffs(df):
            assert_unchanged(lambda d: mutant(d, ft.kickoff_anchor), df, "saque", c)


def test_form_features_are_per_team_and_strictly_before():
    rows = [("2014-08-10", "a", "b", 1, 0), ("2014-08-10", "c", "d", 5, 5),
            ("2014-08-17", "b", "a", 2, 2), ("2014-08-17", "a", "c", 0, 0)]  # «a» juega dos veces
    df = pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = "2014/15"
    f = ft.form_features(df.iloc[:3], ft.kickoff_anchor)
    assert np.isnan(f.loc[0, "home_gf5"]) and f.loc[0, "home_n_prev"] == 0
    assert f.loc[2, "home_gf5"] == 0 and f.loc[2, "away_gf5"] == 1  # b marcó 0; a marcó 1
    assert f.loc[2, "away_pts5"] == 3 and f.loc[2, "home_n_prev"] == 1


def test_no_feature_reads_closing_odds(df):
    X = build(df, "apertura")
    assert not [c for c in X.columns if any(k in c for k in ("close", "ref_", "result"))]
    with pytest.raises(ValueError, match="prohibidas"):
        ft.check_columns(["elo_diff", "pin_close_home"])


# ----------------------------------------------------------------------------- modelos


def predictions(df, variant, params=None, fit=ml.fit_predict):
    X = build(df, ml.DECISION[variant])
    return walk(df, X, variant, params or SMALL[variant], fit)


def walk(df, X, variant, params, fit):
    y = ml.outcome_index(df["result"])
    season = df["season"].to_numpy()
    ok = ml.usable(variant, X)
    out = pd.DataFrame(np.nan, index=df.index, columns=ml.PROB)
    for s in ("2014/15",):
        tr, te = (season >= ml.TRAIN_FROM) & (season < s) & ok, (season == s) & ok
        out.loc[te, :] = fit(variant, params, X[tr], y[tr], X[te])
    return out


def test_walk_forward_matches_the_reference_loop(df):
    X = build(df, "apertura")
    a = ml.walk_forward(df, X, "residuo_logit", SMALL["residuo_logit"], ["2014/15"])
    b = walk(df, X, "residuo_logit", SMALL["residuo_logit"], ml.fit_predict)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float))


@pytest.mark.parametrize("variant", list(ml.DECISION))
def test_model_predictions_only_use_the_past(df, variant):
    """De punta a punta: variables recalculadas y modelo reentrenado sobre datos alterados."""
    days = np.sort(df.loc[df["season"] == "2014/15", "date"].unique())
    for q in (0.3, 0.7):
        c = pd.Timestamp(days[int(q * len(days))])
        assert_unchanged(lambda d: ml.walk_forward(d, build(d, ml.DECISION[variant]), variant,
                                                   SMALL[variant], ["2014/15"]),
                         df, ml.DECISION[variant], c)  # fmt: skip


@pytest.mark.parametrize("variant", list(ml.DECISION))
def test_predictions_ignore_closing_odds(df, variant):
    alt = df.copy()
    rng = np.random.default_rng(1)
    for c in [c for c in df.columns if "close" in c or c.startswith("ref_")]:
        alt[c] = alt[c] * rng.uniform(0.5, 2, len(alt))
    a, b = predictions(df, variant), predictions(alt, variant)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_closing_odds_mutant_is_caught(df):
    """Mutante: el cierre de Pinnacle, renombrado para burlar el filtro por nombre, como variable."""

    def leaky_fit(variant, params, X_tr, y_tr, X_te):
        X_tr, X_te = X_tr.copy(), X_te.copy()
        X_tr["x"], X_te["x"] = df.loc[X_tr.index, "pin_close_home"], cur.loc[X_te.index, "pin_close_home"]
        return ml.fit_predict(variant, params, X_tr, y_tr, X_te)

    alt = df.copy()
    alt["pin_close_home"] *= np.random.default_rng(2).uniform(0.5, 2, len(alt))
    cur = df
    a = predictions(df, "lgbm", fit=leaky_fit)
    cur = alt
    b = predictions(alt, "lgbm", fit=leaky_fit)
    assert not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("variant", ["lgbm", "residuo_logit"])
def test_training_on_the_evaluated_season_is_caught(df, variant):
    """Mutante: el modelo de la temporada S se entrena también con S (todas sus filas)."""

    def leaky(d):
        X = build(d, ml.DECISION[variant])
        y = ml.outcome_index(d["result"])
        ok = ml.usable(variant, X)
        tr = (d["season"].to_numpy() >= ml.TRAIN_FROM) & ok
        te = (d["season"] == "2014/15").to_numpy() & ok
        out = pd.DataFrame(np.nan, index=d.index, columns=ml.PROB)
        out.loc[te, :] = ml.fit_predict(variant, SMALL[variant], X[tr], y[tr], X[te])
        return out

    days = np.sort(df.loc[df["season"] == "2014/15", "date"].unique())
    with pytest.raises(AssertionError):
        assert_unchanged(leaky, df, ml.DECISION[variant], pd.Timestamp(days[len(days) // 2]))


def test_scaler_fitted_on_all_rows_is_caught(df, monkeypatch):
    """Mutante: el residuo lineal estandariza con la media y la desviación de entrenamiento + prueba."""
    orig = ml.OffsetSoftmax.fit
    holder = {}

    def leaky_fit(self, X, base, y):
        orig(self, X, base, y)
        both = pd.concat([X, holder["test"]])
        self.mean_, self.std_ = both.mean(), both.std().replace(0, 1).fillna(1)
        return self

    def fit(variant, params, X_tr, y_tr, X_te):
        holder["test"] = ml.logit_design(X_te)
        return ml.fit_predict(variant, params, X_tr, y_tr, X_te)

    monkeypatch.setattr(ml.OffsetSoftmax, "fit", leaky_fit)
    days = np.sort(df.loc[df["season"] == "2014/15", "date"].unique())
    with pytest.raises(AssertionError):
        assert_unchanged(lambda d: predictions(d, "residuo_logit", fit=fit), df, "apertura",
                         pd.Timestamp(days[len(days) // 2]))  # fmt: skip


def test_final_test_rows_are_rejected(df):
    late = df.copy()
    late.loc[late.index[-1], "date"] = pd.Timestamp("2025-08-15")
    with pytest.raises(ValueError, match="prueba final"):
        build(late, "saque")


# ----------------------------------------------------------------------------- residuos


def test_offset_softmax_with_huge_penalty_returns_the_base(df):
    X = build(df, "apertura")
    X = X[X[ml.OPEN].notna().all(axis=1)]
    y = ml.outcome_index(df.loc[X.index, "result"])
    base = X[ml.OPEN].to_numpy(float)
    m = ml.OffsetSoftmax(alpha=1e6).fit(ml.logit_design(X), base, y)
    p = m.predict(ml.logit_design(X), base)
    # con w ≈ 0 solo quedan los interceptos: p ∝ base · exp(b)
    ratio = p / base
    assert np.allclose(ratio / ratio[:, :1], (ratio / ratio[:, :1])[0], atol=1e-3)
    assert np.allclose(p.sum(1), 1)


def test_offset_softmax_gradient_matches_finite_differences():
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)))
    base = rng.dirichlet([3, 2, 2], 200)
    y = rng.integers(0, 3, 200)
    m = ml.OffsetSoftmax(0.01)
    m.mean_, m.std_ = X.mean(), X.std()
    Z, onehot = m._z(X), np.eye(3)[y]

    def loss(theta):
        from scipy.special import log_softmax

        ls = log_softmax(m._logits(theta, Z, base), axis=1)
        return -(ls * onehot).sum(1).mean() + 0.01 * (theta[:8] ** 2).sum()

    theta = rng.normal(0, 0.3, 10)
    # reproduce el gradiente analítico de fit
    p = np.exp(np.log(base) + np.column_stack([np.zeros(200), Z @ theta[:8].reshape(4, 2) + theta[8:]]))
    p /= p.sum(1, keepdims=True)
    r = (p - onehot)[:, 1:] / 200
    grad = np.concatenate([(Z.T @ r).ravel() + 0.02 * theta[:8], r.sum(0)])
    np.testing.assert_allclose(grad, approx_fprime(theta, loss, 1e-7), atol=1e-5)


def test_residual_lgbm_starts_from_the_opening(df):
    X = build(df, "apertura")
    X = X[X[ml.OPEN].notna().all(axis=1)]
    y = ml.outcome_index(df.loc[X.index, "result"])
    p = ml.fit_predict("residuo_lgbm", {"leaves": 2, "rounds": 1, "lr": 1e-6, "min_leaf": 20}, X, y, X)
    np.testing.assert_allclose(p, X[ml.OPEN].to_numpy(float), atol=1e-4)
