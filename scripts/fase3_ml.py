"""Fase 3: modelos de ML 1X2 por ventanas, frente al cierre y la apertura de Pinnacle.

    python scripts/fase3_ml.py ajuste       # rejilla en 2014/15–2016/17 → seleccion_fase3.json
    python scripts/fase3_ml.py evaluacion   # 2017/18–2024/25 con la selección → docs/ml.md

Cada variante evaluada añade una fila a `experiments/registro.jsonl` (nunca se sobrescribe).
Elo, pi-ratings y Dixon-Coles usan los hiperparámetros elegidos en el ajuste de la fase 2
(`experiments/seleccion.json`). La temporada 2025/26 no se lee (ver `src/referencias.py`).

Salidas que se suben (solo agregados): `experiments/registro.jsonl`,
`experiments/seleccion_fase3.json`, `docs/ml.md` y `docs/figuras/calibracion-ml.png`.
"""

from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import experiments as ex  # noqa: E402
from src import features as ft  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import ml  # noqa: E402
from src import referencias as rf  # noqa: E402
from src.dixon_coles import DixonColes  # noqa: E402
from src.ratings import Elo, PiRating  # noqa: E402

RAW = ROOT / "data" / "raw" / "football-data"
SEL2 = ROOT / "experiments" / "seleccion.json"
SELECTION = ROOT / "experiments" / "seleccion_fase3.json"
REPORT = ROOT / "docs" / "ml.md"
FIGURE = ROOT / "docs" / "figuras" / "calibracion-ml.png"
TUNING = ["2014/15", "2015/16", "2016/17"]  # 2013/14 es el primer año de entrenamiento
CLOSE, OPEN = "mercado_cierre", "mercado_apertura"
DC_FROM = pd.Timestamp("2013-07-01")
NAMES = {CLOSE: "Mercado, cierre (Pinnacle sin margen)", OPEN: "Mercado, apertura (Pinnacle sin margen)",
         "dixon_coles": "Dixon-Coles (fase 2)", "lgbm": "LightGBM", "catboost": "CatBoost",
         "residuo_lgbm": "Residuo sobre apertura, LightGBM",
         "residuo_logit": "Residuo sobre apertura, logit"}  # fmt: skip


def grid() -> list[tuple[str, dict]]:
    """Rejilla fijada antes de mirar resultados. Todas las variantes se registran."""
    out: list[tuple[str, dict]] = []
    for leaves, rounds in itertools.product((7, 31), (300, 1000)):
        out.append(("lgbm", {"leaves": leaves, "rounds": rounds, "lr": 0.02, "min_leaf": 200}))
    for depth, it in itertools.product((4, 6), (500, 1500)):
        out.append(("catboost", {"depth": depth, "iterations": it, "lr": 0.03}))
    for leaves, rounds in itertools.product((3, 7), (100, 400)):
        out.append(("residuo_lgbm", {"leaves": leaves, "rounds": rounds, "lr": 0.01, "min_leaf": 500, "l2": 10.0}))
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        out.append(("residuo_logit", {"alpha": alpha}))
    return out


def features(df: pd.DataFrame, n_jobs: int) -> dict[str, pd.DataFrame]:
    sel = json.loads(SEL2.read_text())["seleccion"]
    kw = {"elo": Elo(**sel["elo"]["parametros"]), "pi": PiRating(**sel["pi"]["parametros"]),
          "dc": DixonColes(**sel["dixon_coles"]["parametros"]), "dc_from": DC_FROM, "n_jobs": n_jobs}  # fmt: skip
    out = {}
    for decision in ("saque", "apertura"):
        t = time.time()
        out[decision] = ft.build(df, decision, **kw)
        print(f"variables {decision}: {out[decision].shape[1]} columnas ({time.time() - t:.0f} s)")
    return out


def predict(df, X, variant, params, seasons) -> pd.DataFrame:
    mask = df["season"].isin(seasons)
    if variant == CLOSE:
        return rf.market_probs(df, "pin_close").where(mask, np.nan)
    if variant == OPEN:
        return rf.market_probs(df, "pin_open").where(mask, np.nan)
    if variant == "dixon_coles":  # la misma predicción de la fase 2 (ancla del lunes)
        p = X["saque"][["dc_home", "dc_draw", "dc_away"]].set_axis(rf.PROB, axis=1)
        return p.where(mask, np.nan)
    return ml.walk_forward(df, X[ml.DECISION[variant]], variant, params, seasons)


def evaluate(df, preds, n_boot):
    """Las dos referencias sobre el mismo conjunto común."""
    c = rf.evaluate(df, preds, reference=CLOSE, n_boot=n_boot)
    o = rf.evaluate(df, preds, reference=OPEN, n_boot=n_boot)
    for name, m in c["modelos"].items():
        for k in ("delta_logloss", "delta_logloss_ic", "delta_rps", "delta_rps_ic"):
            if k in o["modelos"][name]:
                m[k.replace("delta", "dap")] = o["modelos"][name][k]
    return c


def short(m: dict) -> dict:
    keep = ("logloss", "logloss_ic", "rps", "rps_ic", "brier", "ece", "delta_logloss", "delta_logloss_ic",
            "delta_rps", "delta_rps_ic", "dap_logloss", "dap_logloss_ic", "dap_rps", "dap_rps_ic")  # fmt: skip
    rnd = lambda v: [round(x, 5) for x in v] if isinstance(v, list) else round(float(v), 5)  # noqa: E731
    return {k: rnd(m[k]) for k in keep if k in m}


# ----------------------------------------------------------------------------- ajuste


def run_tuning(df, X) -> None:
    h = ex.data_hash(df[df["season"] <= TUNING[-1]])
    markets = {v: predict(df, X, v, {}, TUNING) for v in (CLOSE, OPEN)}
    best: dict[str, tuple[float, dict]] = {}
    for variant, params in grid():
        t = time.time()
        p = predict(df, X, variant, params, TUNING)
        res = evaluate(df, markets | {variant: p}, n_boot=500)
        m = res["modelos"][variant]
        ex.log(variant, "ajuste", params, short(m) | {"n": res["n"]}, h,
               notes=f"fase 3; temporadas {TUNING[0]}–{TUNING[-1]}; entrenamiento desde {ml.TRAIN_FROM}")  # fmt: skip
        print(f"{variant:14s} {json.dumps(params):70s} logloss {m['logloss']:.4f} "
              f"Δcierre {m['delta_logloss']:+.4f} Δapertura {m['dap_logloss']:+.4f} n {res['n']} "
              f"({time.time() - t:.0f} s)", flush=True)  # fmt: skip
        if variant not in best or m["logloss"] < best[variant][0]:
            best[variant] = (m["logloss"], params)
    sel = {v: {"parametros": p, "logloss_ajuste": round(ll, 5)} for v, (ll, p) in best.items()}
    SELECTION.write_text(json.dumps({"regla": f"mínimo log loss en ajuste {TUNING[0]}–{TUNING[-1]}",
                                     "hash_datos": h, "seleccion": sel}, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
    print(f"Selección guardada en {SELECTION.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- evaluación


def fmt_ci(m, key, nd=4, sign=False):
    lo, hi = m[f"{key}_ic"]
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(m[key])} [{f.format(lo)}, {f.format(hi)}]"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])  # fmt: skip


def movement(df, preds, ok) -> list[list[str]]:
    """¿La corrección del modelo sobre la apertura apunta hacia donde se mueve el mercado?
    Correlación entre (p_modelo − p_apertura) y (p_cierre − p_apertura), por resultado."""
    rows = []
    po, pc = preds[OPEN].loc[ok, rf.PROB].to_numpy(float), preds[CLOSE].loc[ok, rf.PROB].to_numpy(float)
    week = df.loc[ok, "date"].dt.strftime("%G-%V").to_numpy()
    for v in ("residuo_lgbm", "residuo_logit"):
        pm = preds[v].loc[ok, rf.PROB].to_numpy(float)
        cells = [NAMES[v]]
        for k in range(3):
            a, b = pm[:, k] - po[:, k], pc[:, k] - po[:, k]
            r = np.corrcoef(a, b)[0, 1]
            cells.append(f"{r:+.3f} {corr_ci(a, b, week)}")
        rows.append(cells)
    return rows


def corr_ci(a, b, week, n_boot=1000, seed=0):
    codes, uniq = pd.factorize(pd.Series(week))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        cov = np.average((a - ma) * (b - mb), weights=w)
        out.append(cov / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def plot_calibration(cal: dict[str, pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b61d1", "#52514e"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, outcome in zip(axes, ("local", "empate", "visitante"), strict=True):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], color="#9a9893", lw=1, ls="--", zorder=1)
        for (name, t), c, mk in zip(cal.items(), colors, markers, strict=False):
            g = t[(t["resultado"] == outcome) & (t["n"] >= 30)]
            ax.plot(g["prevista"], g["observada"], color=c, lw=1.6, marker=mk, ms=5, label=NAMES[name],
                    markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=2)  # fmt: skip
        ax.set_title(f"Gana {outcome}" if outcome != "empate" else "Empate", color="#0b0b0b", fontsize=11)
        ax.set_xlabel("Probabilidad prevista", color="#52514e")
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, 0.95)
        ax.grid(color="#e6e5e1", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#d3d2cd")
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("Frecuencia observada", color="#52514e")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Calibración por ventanas, 2017/18–2024/25 (tramos con ≥ 30 partidos)", color="#0b0b0b")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=130, facecolor=fig.get_facecolor())


def importance(df, X, params) -> list[tuple[str, float]]:
    """Ganancia media de cada variable en el LightGBM de la última temporada evaluada (descriptivo)."""
    import lightgbm as lgb

    Xs = X["saque"]
    tr = ((df["season"] >= ml.TRAIN_FROM) & (df["season"] < rf.EVALUATION[-1])).to_numpy()
    cols = ml.gbm_columns(Xs)
    b = lgb.train(ml._lgb_params(params, 0), lgb.Dataset(Xs.loc[tr, cols], ml.outcome_index(df.loc[tr, "result"])),
                  num_boost_round=params["rounds"])  # fmt: skip
    gain = pd.Series(b.feature_importance("gain"), index=cols)
    return list((gain / gain.sum()).sort_values(ascending=False).head(10).items())


def run_evaluation(df, X, n_boot) -> None:
    sel = json.loads(SELECTION.read_text())["seleccion"]
    h = ex.data_hash(df)
    configs = {CLOSE: {}, OPEN: {}, "dixon_coles": {}, **{v: s["parametros"] for v, s in sel.items()}}
    preds = {}
    for v, params in configs.items():
        t = time.time()
        preds[v] = predict(df, X, v, params, rf.EVALUATION)
        print(f"{v}: {time.time() - t:.0f} s", flush=True)
    overall = evaluate(df, preds, n_boot)
    recent = ["2021/22", "2022/23", "2023/24", "2024/25"]
    sub = {s: evaluate(df, {k: p.where(df["season"].eq(s), np.nan) for k, p in preds.items()}, 1000)
           for s in rf.EVALUATION}  # fmt: skip
    rec = evaluate(df, {k: p.where(df["season"].isin(recent), np.nan) for k, p in preds.items()}, n_boot)
    by_league = {lg: evaluate(df, {k: p.where(df["league"].eq(lg), np.nan) for k, p in preds.items()}, 500)
                 for lg in sorted(df["league"].unique())}  # fmt: skip
    for v in configs:
        if v in (CLOSE, "dixon_coles"):
            continue  # el cierre es la referencia; Dixon-Coles ya se registró en la fase 2
        ex.log(v, "evaluacion", configs[v],
               short(overall["modelos"][v]) | {"n": overall["n"], "semanas": overall["semanas"]}, h,
               notes=f"fase 3; temporadas {rf.EVALUATION[0]}–{rf.EVALUATION[-1]}; selección de {SELECTION.name}")  # fmt: skip
    ok = np.logical_and.reduce([p[rf.PROB].notna().all(axis=1).to_numpy() for p in preds.values()])
    y = mt.outcome_index(df.loc[ok, "result"])
    cal = {v: mt.calibration_table(p.loc[ok, rf.PROB].to_numpy(float), y) for v, p in preds.items()}
    plot_calibration(cal)
    write_report(df, X, preds, ok, overall, rec, sub, by_league, cal, configs, h)


def write_report(df, X, preds, ok, overall, rec, sub, by_league, cal, configs, h) -> None:
    order = list(configs)
    others = [v for v in order if v != CLOSE]
    rows = []
    for v in order:
        m = overall["modelos"][v]
        dc = fmt_ci(m, "delta_logloss", sign=True) if v != CLOSE else "—"
        do = fmt_ci(m, "dap_logloss", sign=True) if v != OPEN else "—"
        dr = fmt_ci(m, "delta_rps", sign=True) if v != CLOSE else "—"
        rows.append([NAMES[v], fmt_ci(m, "logloss"), dc, do, fmt_ci(m, "rps"), dr, f"{m['ece']:.4f}"])
    main = md_table(["Modelo", "Log loss [IC 95 %]", "Δ log loss vs cierre", "Δ log loss vs apertura",
                     "RPS [IC 95 %]", "Δ RPS vs cierre", "ECE"], rows)  # fmt: skip
    rows = []
    for v in order:
        m = rec["modelos"][v]
        rows.append([NAMES[v], fmt_ci(m, "logloss"), fmt_ci(m, "delta_logloss", sign=True) if v != CLOSE else "—",
                     fmt_ci(m, "dap_logloss", sign=True) if v != OPEN else "—"])  # fmt: skip
    recent = md_table(["Modelo", "Log loss", "Δ vs cierre", "Δ vs apertura"], rows)
    ml_models = [v for v in order if v in ml.DECISION]
    srows = []
    for s, r in sub.items():
        cells = [s, str(r["n"]), f"{r['modelos'][CLOSE]['logloss']:.4f}"]
        cells += [fmt_ci(r["modelos"][v], "delta_logloss", sign=True) for v in ml_models]
        srows.append(cells)
    seasons = md_table(["Temporada", "N", "Log loss cierre"] + [f"Δ {NAMES[v]}" for v in ml_models], srows)
    srows = []
    for s, r in sub.items():
        cells = [s] + [fmt_ci(r["modelos"][v], "dap_logloss", sign=True) for v in ml_models]
        srows.append(cells)
    seasons_open = md_table(["Temporada"] + [f"Δ {NAMES[v]}" for v in ml_models], srows)
    lrows = []
    for lg, r in by_league.items():
        cells = [lg, str(r["n"]), f"{r['modelos'][CLOSE]['logloss']:.4f}"]
        cells += [f"{r['modelos'][v]['delta_logloss']:+.4f} / {r['modelos'][v]['dap_logloss']:+.4f}"
                  for v in ml_models]  # fmt: skip
        lrows.append(cells)
    leagues = md_table(["Liga", "N", "Log loss cierre"] + [f"{NAMES[v]} (Δ cierre / Δ apertura)" for v in ml_models],
                       lrows)  # fmt: skip
    cal_rows = []
    for outcome in ("local", "empate", "visitante"):
        t0 = cal[CLOSE][cal[CLOSE]["resultado"] == outcome]
        for tramo in t0["tramo"]:
            cells = [outcome, tramo]
            for v in order:
                t = cal[v]
                r = t[(t["resultado"] == outcome) & (t["tramo"] == tramo)]
                cells.append("—" if r.empty or r["n"].iloc[0] < 30 else
                             f"{r['prevista'].iloc[0]:.3f} → {r['observada'].iloc[0]:.3f}")  # fmt: skip
            cal_rows.append(cells)
    calib = md_table(["Resultado", "Tramo"] + [NAMES[v] for v in order], cal_rows)
    mov = md_table(["Modelo", "Local", "Empate", "Visitante"], movement(df, preds, ok))
    imp = "\n".join(f"| `{c}` | {g:.1%} |" for c, g in importance(df, X, configs["lgbm"]))
    params = "\n".join(f"- **{NAMES[v]}:** `{json.dumps(p, ensure_ascii=False)}`" for v, p in configs.items() if p)
    reg = ex.read()
    tuning = [r for r in reg if r["fase"] == "ajuste" and r["variante"] in ml.DECISION]
    n_tried = len({(r["variante"], json.dumps(r["parametros"], sort_keys=True)) for r in tuning})
    n_eval = sum(1 for r in reg if r["fase"] == "evaluacion" and r["variante"] in ml.DECISION) // len(ml_models)
    text = f"""# Modelos de ML 1X2 (fase 3)

Generado con `scripts/fase3_ml.py evaluacion` el {date.today().isoformat()}. Solo agregados; los
datos de partidos no se suben. Hash de los datos evaluados: `{h}`.

## Qué se compara

- **LightGBM y CatBoost** sobre {len(ml.gbm_columns(X['saque']))} variables previas al partido: Elo y
  pi-ratings (nivel y diferencia), probabilidades de Dixon-Coles, forma de los últimos 5 y 20 partidos
  (goles, tiros, tiros a puerta, córners y puntos, a favor y en contra), descanso, partidos jugados en
  la temporada, liga y categoría. Deciden **antes del saque**: ven los resultados de días anteriores.
  **No ven ninguna cuota.**
- **Modelos de residuos sobre la apertura:** parten de la apertura de Pinnacle sin margen y aprenden
  solo la corrección, log p = log p_apertura + f(x). Dos versiones: LightGBM con la apertura como
  `init_score` y una regresión softmax con desplazamiento y penalización L2. Deciden **cuando se
  recoge la apertura** (viernes por la tarde para partidos de viernes a lunes, martes por la tarde
  para martes a jueves, según football-data): solo ven resultados anteriores a ese día.
- Referencias: el **cierre** y la **apertura** de Pinnacle sin margen, y Dixon-Coles de la fase 2.

## Cómo se ha medido

- **Ventanas.** Para predecir la temporada S se entrena un modelo con {ml.TRAIN_FROM}…S − 1. Las
  variables de entrenamiento son las que existían en su día (ratings en línea, Dixon-Coles por
  ventanas), así que no hay predicciones «del futuro» apiladas.
- **Ajuste.** Rejilla fijada antes de mirar, en 2014/15–2016/17 ({n_tried} variantes distintas;
  regla: mínimo log loss). Elo, pi y Dixon-Coles usan los hiperparámetros elegidos en la fase 2.
  Evaluaciones de ML registradas hasta hoy: {n_eval}. **2025/26 no se ha leído.**
- **Conjunto común.** {overall['n']} partidos con probabilidad de todos los modelos y ambas cuotas
  de Pinnacle ({overall['n_descartados']} descartados).
- **Intervalos.** IC 95 % por bootstrap de semanas ISO ({overall['semanas']} semanas, 2.000
  remuestras). Δ son diferencias emparejadas partido a partido; **Δ > 0 significa peor que la
  referencia.**
- **Antifuga.** `tests/test_ml.py`: alterar resultados y estadísticas desde un día c no cambia ninguna
  variable ni predicción de partidos cuya ancla sea ≤ c (también los posteriores a c con la apertura),
  y cambiar las cuotas de cierre no cambia ninguna predicción. Seis mutantes con fugas deliberadas
  (ancla equivocada, media sin agrupar, día incluido, cierre como variable, entrenar con la temporada
  evaluada, escalado con la prueba) son detectados.

Hiperparámetros elegidos:

{params}

## Resultado: 2017/18–2024/25

{main}

ECE: error de calibración esperado (10 tramos, media de los tres resultados).

### Solo 2021/22–2024/25

{recent}

## Por temporada

Δ log loss frente al **cierre** (IC 95 %):

{seasons}

Δ log loss frente a la **apertura** (IC 95 %):

{seasons_open}

## Por liga (Δ log loss frente al cierre / frente a la apertura)

{leagues}

## ¿La corrección anticipa el movimiento del mercado?

Correlación entre la corrección del modelo (p_modelo − p_apertura) y el movimiento de Pinnacle
(p_cierre − p_apertura), con IC 95 % por bootstrap de semanas. Es descriptivo: una correlación
positiva sería la señal que buscaría la fase 4 (CLV), no un beneficio.

{mov}

## Variables más útiles para LightGBM

Ganancia relativa en el modelo entrenado hasta 2023/24 (descriptivo).

| Variable | Ganancia |
|---|---|
{imp}

## Calibración

Probabilidad media prevista → frecuencia observada, tramos de 0,1 («—»: menos de 30 partidos).

![Curvas de calibración](figuras/calibracion-ml.png)

{calib}

## Lectura

{interpretation(overall, sub, others)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"Informe en {REPORT.relative_to(ROOT)}")


def verdict(lo, hi, ref):
    if lo > 0:
        return f"peor que {ref} (IC entero por encima de 0)"
    if hi < 0:
        return f"**mejor que {ref}: sospechoso hasta auditarlo**"
    return f"indistinguible de {ref} con esta muestra"


def interpretation(overall, sub, others) -> str:
    lines = []
    for v in others:
        m = overall["modelos"][v]
        txt = f"- **{NAMES[v]}:** Δ {m['delta_logloss']:+.4f}, {verdict(*m['delta_logloss_ic'], 'el cierre')}"
        if v != OPEN:
            txt += f"; frente a la apertura Δ {m['dap_logloss']:+.4f}, {verdict(*m['dap_logloss_ic'], 'la apertura')}"
            better = sum(1 for r in sub.values() if r["modelos"][v]["dap_logloss_ic"][1] < 0)
            worse = sum(1 for r in sub.values() if r["modelos"][v]["dap_logloss_ic"][0] > 0)
            txt += f". Por temporadas frente a la apertura: mejor en {better} de {len(sub)}, peor en {worse}"
        lines.append(txt + ".")
    return "\n".join(lines)


def main(argv=None) -> int:
    from threadpoolctl import threadpool_limits

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["ajuste", "evaluacion"])
    ap.add_argument("--datos", type=Path, default=RAW)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    with threadpool_limits(1, user_api="blas"):
        df = rf.load_history(args.datos)
        print(f"{len(df)} partidos, {df['season'].min()} → {df['season'].max()}")
        X = features(df, args.jobs)
        if args.modo == "ajuste":
            run_tuning(df, X)
        else:
            run_evaluation(df, X, args.boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
