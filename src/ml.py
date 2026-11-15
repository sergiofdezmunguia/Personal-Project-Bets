"""Fase 3: modelos de ML 1X2 por ventanas temporales.

- `lgbm`, `catboost`: árboles potenciados sobre ratings, Dixon-Coles, forma y calendario, con la
  hora de decisión **saque**. No ven ninguna cuota.
- `residuo_lgbm`, `residuo_logit`: parten de la **apertura** de Pinnacle sin margen y aprenden solo
  la corrección (log p = log p_apertura + f(x), normalizado). Usan la hora de decisión **apertura**.

Validación: para predecir la temporada S se entrena con las temporadas desde `TRAIN_FROM` hasta
S − 1 (ventana creciente, un modelo por temporada). Ninguna fila de S ni posterior entra en su ajuste,
y los escalados se ajustan solo con el entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from src.features import check_columns
from src.metrics import outcome_index

TRAIN_FROM = "2013/14"  # 2012/13 solo arranca ratings y forma
PROB = ["p_home", "p_draw", "p_away"]
OPEN = ["open_home", "open_draw", "open_away"]
DECISION = {"lgbm": "saque", "catboost": "saque", "residuo_lgbm": "apertura", "residuo_logit": "apertura"}
THREADS = 4


def gbm_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in OPEN]


def logit_design(X: pd.DataFrame) -> pd.DataFrame:
    """Variables compactas del residuo lineal (sin escalar)."""
    eps = 1e-6
    o = X[OPEN].clip(lower=eps)
    d = pd.DataFrame(index=X.index)
    d["open_lo_ha"] = np.log(o["open_home"] / o["open_away"])
    d["open_lo_d"] = np.log(o["open_draw"])
    for k in ("home", "draw", "away"):
        d[f"dc_vs_open_{k}"] = np.log(X[f"dc_{k}"].clip(lower=eps) / o[f"open_{k}"])
    d["elo_diff"], d["pi_diff"] = X["elo_diff"], X["pi_diff"]
    for st, n in (("gf", 20), ("ga", 20), ("tf", 20), ("ta", 20), ("pts", 5), ("pts", 20)):
        d[f"{st}{n}_diff"] = X[f"home_{st}{n}"] - X[f"away_{st}{n}"]
    d["rest_diff"] = X["home_rest"] - X["away_rest"]
    d["tier"] = X["tier"]
    return d


# ----------------------------------------------------------------------------- residuo lineal


class OffsetSoftmax:
    """Regresión softmax con desplazamiento: logit_k = log p0_k + b_k + x·w_k (k = empate,
    visitante; local es la clase base). Penalización L2 `alpha` sobre w, no sobre b."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def _logits(self, theta, Z, base):
        k = Z.shape[1]
        W = theta[: 2 * k].reshape(k, 2)
        b = theta[2 * k :]
        add = np.column_stack([np.zeros(len(Z)), Z @ W + b])
        return np.log(base) + add

    def fit(self, X: pd.DataFrame, base: np.ndarray, y: np.ndarray) -> OffsetSoftmax:
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1).fillna(1)
        Z = self._z(X)
        k = Z.shape[1]
        onehot = np.eye(3)[y]

        def f(theta):
            lg = self._logits(theta, Z, base)
            ls = log_softmax(lg, axis=1)
            p = np.exp(ls)
            W = theta[: 2 * k]
            loss = -(ls * onehot).sum(1).mean() + self.alpha * (W**2).sum()
            r = (p - onehot)[:, 1:] / len(Z)
            gW = Z.T @ r
            grad = np.concatenate([gW.ravel() + 2 * self.alpha * W, r.sum(0)])
            return loss, grad

        res = minimize(f, np.zeros(2 * k + 2), jac=True, method="L-BFGS-B")
        self.theta_ = res.x
        return self

    def _z(self, X):
        return ((X - self.mean_) / self.std_).fillna(0).to_numpy(float)

    def predict(self, X: pd.DataFrame, base: np.ndarray) -> np.ndarray:
        return softmax(self._logits(self.theta_, self._z(X), base), axis=1)


# ----------------------------------------------------------------------------- un ajuste


def fit_predict(variant: str, params: dict, X_tr: pd.DataFrame, y_tr: np.ndarray,
                X_te: pd.DataFrame, seed: int = 0) -> np.ndarray:  # fmt: skip
    check_columns(X_tr.columns)
    if variant == "lgbm":
        import lightgbm as lgb

        cols = gbm_columns(X_tr)
        p = _lgb_params(params, seed)
        booster = lgb.train(p, lgb.Dataset(X_tr[cols], y_tr), num_boost_round=params["rounds"])
        return booster.predict(X_te[cols])
    if variant == "catboost":
        from catboost import CatBoostClassifier

        cols = gbm_columns(X_tr)
        m = CatBoostClassifier(loss_function="MultiClass", depth=params["depth"], iterations=params["iterations"],
                               learning_rate=params["lr"], l2_leaf_reg=params.get("l2", 3.0),
                               random_seed=seed, thread_count=THREADS, verbose=False,
                               allow_writing_files=False)  # fmt: skip
        m.fit(X_tr[cols], y_tr)
        return m.predict_proba(X_te[cols])
    if variant == "residuo_lgbm":
        import lightgbm as lgb

        cols = list(X_tr.columns)
        base_tr, base_te = np.log(X_tr[OPEN].to_numpy(float)), np.log(X_te[OPEN].to_numpy(float))
        p = _lgb_params(params, seed)
        ds = lgb.Dataset(X_tr[cols], y_tr, init_score=base_tr)
        booster = lgb.train(p, ds, num_boost_round=params["rounds"])
        return softmax(booster.predict(X_te[cols], raw_score=True) + base_te, axis=1)
    if variant == "residuo_logit":
        m = OffsetSoftmax(params["alpha"]).fit(logit_design(X_tr), X_tr[OPEN].to_numpy(float), y_tr)
        return m.predict(logit_design(X_te), X_te[OPEN].to_numpy(float))
    raise ValueError(f"Variante desconocida: {variant}")


def _lgb_params(params: dict, seed: int) -> dict:
    return {"objective": "multiclass", "num_class": 3, "learning_rate": params["lr"],
            "num_leaves": params["leaves"], "min_data_in_leaf": params.get("min_leaf", 200),
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": params.get("l2", 1.0), "num_threads": THREADS, "seed": seed,
            "deterministic": True, "force_row_wise": True, "verbose": -1}  # fmt: skip


# ----------------------------------------------------------------------------- por ventanas


def usable(variant: str, X: pd.DataFrame) -> np.ndarray:
    """Filas en las que la variante puede predecir (los residuos necesitan la apertura)."""
    if DECISION[variant] == "apertura":
        return X[OPEN].notna().all(axis=1).to_numpy()
    return np.ones(len(X), bool)


def walk_forward(df: pd.DataFrame, X: pd.DataFrame, variant: str, params: dict, seasons: list[str],
                 train_from: str = TRAIN_FROM) -> pd.DataFrame:  # fmt: skip
    """Probabilidades de `variant` para las temporadas pedidas; un modelo por temporada entrenado
    solo con temporadas anteriores."""
    y = outcome_index(df["result"])
    season = df["season"].to_numpy()
    ok = usable(variant, X)
    out = pd.DataFrame(np.nan, index=df.index, columns=PROB)
    for s in seasons:
        train = (season >= train_from) & (season < s) & ok
        test = (season == s) & ok
        if not train.any() or not test.any():
            continue
        out.loc[test, :] = fit_predict(variant, params, X[train], y[train], X[test])
    return out
