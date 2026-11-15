"""Fase 2: registro de experimentos, métricas y modelos de referencia, con datos inventados.

Los tests antifuga comprueban la propiedad que importa: cambiar resultados a partir de un día T no
cambia ninguna predicción de partidos jugados hasta T (incluido el propio día T, porque los partidos
del mismo día no se ven entre sí). Un modelo con fuga deliberada debe ser detectado.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src import experiments as ex
from src import metrics as mt
from src import referencias as rf
from src.dixon_coles import DixonColes, _nll_grad, score_probs
from src.dixon_coles import predict_walk_forward as dc_walk_forward
from src.ratings import Elo, OrderedLogit, PiRating, check_input, run_online, walk_forward_probs

# ----------------------------------------------------------------------------- datos inventados


def synthetic(seed: int = 3, seasons=(2012, 2013, 2014), teams_per_league: int = 6) -> pd.DataFrame:
    """Tres ligas (dos del mismo país con un ascenso/descenso por temporada) y resultados Poisson."""
    rng = np.random.default_rng(seed)
    leagues = {lg: [f"{lg}-t{i}" for i in range(teams_per_league)] for lg in ("E0", "E1", "D1")}
    strength = {t: rng.normal(0, 0.35) - 0.3 * (lg == "E1") for lg, ts in leagues.items() for t in ts}
    rows = []
    for season in seasons:
        label = f"{season}/{(season + 1) % 100:02d}"
        for lg, ts in leagues.items():
            day = pd.Timestamp(f"{season}-08-10") + pd.Timedelta(days=int(lg == "D1"))
            fixtures = [(h, a) for h in ts for a in ts if h != a]
            rng.shuffle(fixtures)
            for k in range(0, len(fixtures), len(ts) // 2):
                for h, a in fixtures[k : k + len(ts) // 2]:
                    hg = rng.poisson(np.exp(0.25 + 0.1 + strength[h] - strength[a]))
                    ag = rng.poisson(np.exp(0.25 + strength[a] - strength[h]))
                    rows.append({"date": day, "season": label, "league": lg, "home_team_id": h,
                                 "away_team_id": a, "home_goals": hg, "away_goals": ag})  # fmt: skip
                day += pd.Timedelta(days=7)
        # ascenso y descenso entre E0 y E1
        worst = min(leagues["E0"], key=strength.get)
        best = max(leagues["E1"], key=strength.get)
        leagues["E0"] = [best if t == worst else t for t in leagues["E0"]]
        leagues["E1"] = [worst if t == best else t for t in leagues["E1"]]
    df = pd.DataFrame(rows).sort_values(["date", "league"], kind="stable").reset_index(drop=True)
    df["result"] = np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], ["H", "D"], "A")
    df["match_id"] = df.index.astype(str)
    for s, v in (("home", 0.45), ("draw", 0.27), ("away", 0.28)):
        df[f"ref_prob_{s}"] = v
    return df


def scramble_from(df: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """Resultados nuevos al azar en todos los partidos con fecha ≥ cutoff."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    late = out["date"] >= cutoff
    out.loc[late, "home_goals"] = rng.integers(0, 6, late.sum())
    out.loc[late, "away_goals"] = rng.integers(0, 6, late.sum())
    hg, ag = out["home_goals"], out["away_goals"]
    out["result"] = np.select([hg > ag, hg == ag], ["H", "D"], "A")
    return out


SEASONS = ["2013/14", "2014/15"]


def elo_probs(df):
    return walk_forward_probs(df, run_online(df, Elo()), SEASONS, fit_from="2013/14")


def pi_probs(df):
    return walk_forward_probs(df, run_online(df, PiRating()), SEASONS, fit_from="2013/14")


def dc_probs(df):
    return dc_walk_forward(df, DixonColes(), predict_from=pd.Timestamp("2013-08-01"))


def leaky_probs(df):
    """Fuga deliberada: usa el resultado del propio partido."""
    p = elo_probs(df)
    p.loc[df["result"] == "H", "p_home"] += 0.01
    return p


def assert_past_unchanged(predict_fn, df: pd.DataFrame, cutoff: pd.Timestamp) -> int:
    base = predict_fn(df)
    alt = predict_fn(scramble_from(df, cutoff))
    upto = (df["date"] <= cutoff).to_numpy() & base.notna().all(axis=1).to_numpy()
    assert upto.sum() > 0
    np.testing.assert_allclose(alt.to_numpy()[upto], base.to_numpy()[upto], rtol=0, atol=1e-12)
    return int(upto.sum())


@pytest.fixture(scope="module")
def df():
    return synthetic()


# ----------------------------------------------------------------------------- antifuga


@pytest.mark.parametrize("fn", [elo_probs, pi_probs, dc_probs], ids=["elo", "pi", "dixon_coles"])
@pytest.mark.parametrize("q", [0.34, 0.5, 0.67, 0.85])
def test_changing_future_results_does_not_change_past_predictions(df, fn, q):
    days = np.sort(df["date"].unique())
    cutoff = pd.Timestamp(days[int(q * len(days))])  # un día con partidos: también debe quedar igual
    assert_past_unchanged(fn, df, cutoff)


def test_the_check_detects_a_deliberate_leak(df):
    with pytest.raises(AssertionError):
        assert_past_unchanged(leaky_probs, df, pd.Timestamp(np.sort(df["date"].unique())[-20]))


def test_same_day_matches_do_not_see_each_other(df):
    """Cambiar un partido de un día no cambia la predicción de los demás partidos de ese día."""
    day = df.loc[df["season"] == "2014/15", "date"].iloc[0]
    idx = df.index[df["date"] == day]
    alt = df.copy()
    alt.loc[idx[0], ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    for model in (Elo(), PiRating()):
        a, b = run_online(df, model), run_online(alt, model)
        np.testing.assert_array_equal(a[idx], b[idx])


def test_season_mapping_uses_only_previous_seasons(df):
    """Las probabilidades de 2013/14 no cambian si se alteran todos los resultados de 2014/15."""
    later = scramble_from(df, pd.Timestamp("2014-07-01"))
    a, b = elo_probs(df), elo_probs(later)
    m = (df["season"] == "2013/14").to_numpy()
    np.testing.assert_allclose(a.to_numpy()[m], b.to_numpy()[m])


def test_final_test_rows_are_rejected(df):
    late = df.copy()
    late.loc[late.index[-1], "date"] = pd.Timestamp("2025-08-15")
    with pytest.raises(ValueError, match="prueba final"):
        check_input(late)
    with pytest.raises(ValueError, match="prueba final"):
        run_online(late, Elo())


def test_load_history_never_opens_final_test_files(tmp_path):
    good = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSCH,PSCD,PSCA\n"
            "SP1,17/08/2024,Real Equipo,Club Uno,2,1,H,1.95,3.5,4.2\n")  # fmt: skip
    (tmp_path / "SP1_2425.csv").write_text(good, encoding="utf-8")
    (tmp_path / "SP1_2526.csv").write_bytes(b"\xff\xfe esto no es un CSV \x00")
    (tmp_path / "SP1_2627.csv").write_bytes(b"\x00\x01")
    out = rf.load_history(tmp_path)
    assert len(out) == 1 and out["season"].tolist() == ["2024/25"]


# ----------------------------------------------------------------------------- modelos


def test_elo_is_zero_sum_and_rewards_the_winner():
    h, a = Elo().update(np.array([1500.0]), np.array([1500.0]), 3, 0)
    assert h[0] > 1500 > a[0] and h[0] + a[0] == pytest.approx(3000)


def test_pi_rating_moves_towards_the_observed_goal_difference():
    m = PiRating()
    h, a = m.update(np.zeros(2), np.zeros(2), 2, 0)
    assert h[0] > 0 and h[1] > 0 and a[1] < 0 and a[0] < 0
    assert m.feature(h, a) > 0


def test_ordered_logit_probabilities_are_valid_and_monotone():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    y = np.where(x + rng.logistic(size=4000) > 0.6, 0, np.where(x + rng.logistic(size=4000) > -0.6, 1, 2))
    p = OrderedLogit().fit(x, y).predict(np.linspace(-2, 2, 9))
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert np.all(np.diff(p[:, 0]) > 0) and np.all(np.diff(p[:, 2]) < 0)


def test_dixon_coles_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n, n_t, n_l = 300, 8, 2
    hi, ai = rng.integers(0, n_t, n), rng.integers(0, n_t, n)
    x, y = rng.poisson(1.4, n).astype(float), rng.poisson(1.1, n).astype(float)
    w = rng.uniform(0.2, 1, n)
    tl, nf = rng.integers(0, n_l, n_t), (rng.uniform(size=n_t) < 0.3).astype(float)
    theta = np.concatenate([[0.2, 0.25, -0.08], rng.normal(0, 0.2, 2 * n_t + 2 * n_l)])
    args = (hi, ai, x, y, w, tl, nf, DixonColes(), n_t, n_l)
    err = check_grad(lambda t: _nll_grad(t, *args)[0], lambda t: _nll_grad(t, *args)[1], theta)
    assert err < 1e-5


def test_score_probs_are_coherent():
    p = score_probs(np.array([2.0, 1.0, 0.5]), np.array([0.5, 1.0, 2.0]), -0.1)
    assert np.allclose(p.sum(1), 1)
    assert p[0, 0] > p[0, 2] and p[2, 2] > p[2, 0]
    assert p[1, 0] == pytest.approx(p[1, 2])


def test_dixon_coles_learns_team_strength(df):
    p = dc_probs(df)
    strong = df.groupby("home_team_id")["home_goals"].mean().idxmax()
    m = (df["home_team_id"] == strong) & p.notna().all(axis=1)
    assert p.loc[m, "p_home"].mean() > 0.45


# ----------------------------------------------------------------------------- métricas


def test_metrics_known_values():
    y = np.array([0, 1, 2])
    uniform = np.full((3, 3), 1 / 3)
    assert np.allclose(mt.log_loss(uniform, y), np.log(3))
    assert np.allclose(mt.rps(np.eye(3)[y], y), 0)
    # local seguro y gana el visitante: el peor RPS posible es 1
    assert mt.rps(np.array([[1.0, 0, 0]]), np.array([2]))[0] == pytest.approx(1)
    assert mt.rps(np.array([[0.5, 0.5, 0]]), np.array([2]))[0] == pytest.approx((0.25 + 1) / 2)


def test_calibration_of_a_perfect_forecaster_is_near_zero():
    rng = np.random.default_rng(5)
    p = rng.dirichlet([4, 3, 3], size=20000)
    y = np.array([rng.choice(3, p=row) for row in p])
    assert mt.ece(p, y) < 0.02
    assert mt.ece(np.tile([0.8, 0.1, 0.1], (20000, 1)), y) > 0.2


def test_cluster_bootstrap_interval_covers_the_mean_and_pairs_series():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(100), 10)
    a = rng.normal(1, 1, 1000)
    out = mt.cluster_bootstrap({"a": a, "a_menos_a": a - a}, clusters, n_boot=500)
    assert out.loc["a", "ic_inf"] < a.mean() < out.loc["a", "ic_sup"]
    assert out.loc["a_menos_a", "ic_inf"] == out.loc["a_menos_a", "ic_sup"] == 0


# ----------------------------------------------------------------------------- registro


def test_registry_only_appends_and_detects_tampering(tmp_path, df):
    path = tmp_path / "registro.jsonl"
    h = ex.data_hash(df)
    ex.log("elo", "ajuste", {"k": 20}, {"logloss": 1.01}, h, path=path)
    ex.log("pi", "ajuste", {"lam": 0.035}, {"logloss": 1.02}, h, path=path)
    rows = ex.read(path)
    assert [r["variante"] for r in rows] == ["elo", "pi"]
    ex.verify(path)
    lines = path.read_text().splitlines()
    edited = json.loads(lines[0]) | {"metricas": {"logloss": 0.90}}
    path.write_text(json.dumps(edited, ensure_ascii=False, sort_keys=True) + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError, match="alterado"):
        ex.verify(path)
    with pytest.raises(ValueError, match="alterado"):  # no deja seguir añadiendo sobre un registro roto
        ex.log("dc", "ajuste", {}, {}, h, path=path)


def test_registry_rejects_unknown_phase(tmp_path):
    with pytest.raises(ValueError):
        ex.log("elo", "final", {}, {}, "x", path=tmp_path / "r.jsonl")


def test_data_hash_changes_with_any_result(df):
    alt = df.copy()
    alt.loc[5, "home_goals"] += 1
    assert ex.data_hash(df) != ex.data_hash(alt)
    assert ex.data_hash(df) == ex.data_hash(df.copy())
