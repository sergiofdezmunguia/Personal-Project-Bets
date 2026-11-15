"""Fase 4: normalización del margen (Shin y potencia) y CLV histórico de apuestas simuladas.

    python scripts/fase4_margen_clv.py ajuste       # 2013/14–2016/17: elige normalización → seleccion_fase4.json
    python scripts/fase4_margen_clv.py evaluacion   # 2017/18–2024/25 → docs/fase4-resultados.md

- **Ajuste.** Cierre y apertura de Pinnacle con las tres normalizaciones en las temporadas de ajuste
  de la fase 2. Regla fijada antes de mirar: la normalización con menor log loss del **cierre** pasa a
  ser la de referencia (para el cierre y la apertura). No hay más parámetros: Shin y potencia
  resuelven su z o su k partido a partido.
- **Evaluación.** Las referencias con las tres normalizaciones y los modelos de residuos de la fase 3
  **con los hiperparámetros ya elegidos** (`seleccion_fase3.json`, sin reajuste), reentrenados por
  ventanas con la apertura normalizada de cada forma. Después, CLV histórico: apuestas planas a la
  apertura de Pinnacle cuando p_modelo · cuota − 1 supera un umbral fijado de antemano (0, 2 % y
  5 %), frente al cierre de Pinnacle sin margen. Todo se registra en `experiments/registro.jsonl`.

2025/26 no se lee. Las variables se guardan en caché en `data/processed/` (excluido de git).
"""

from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import clv as cl  # noqa: E402
from src import experiments as ex  # noqa: E402
from src import features as ft  # noqa: E402
from src import margin as mg  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import ml  # noqa: E402
from src import referencias as rf  # noqa: E402
from src.dixon_coles import DixonColes  # noqa: E402
from src.ratings import Elo, PiRating  # noqa: E402

RAW = ROOT / "data" / "raw" / "football-data"
SEL2 = ROOT / "experiments" / "seleccion.json"
SEL3 = ROOT / "experiments" / "seleccion_fase3.json"
SELECTION = ROOT / "experiments" / "seleccion_fase4.json"
REPORT = ROOT / "docs" / "fase4-resultados.md"
CACHE = ROOT / "data" / "processed"
DC_FROM = pd.Timestamp("2013-07-01")
METHODS = mg.METHODS
THRESHOLDS = (0.0, 0.02, 0.05)  # fijados antes de mirar
SHORT = {"proporcional": "prop", "shin": "shin", "potencia": "pot"}
RESIDUALS = ("residuo_lgbm", "residuo_logit")


def mk(prefix: str, method: str) -> str:
    return f"mercado_{prefix}_{SHORT[method]}"


def name(v: str) -> str:
    parts = {"mercado_cierre": "Cierre", "mercado_apertura": "Apertura", "residuo_lgbm": "Residuo LightGBM",
             "residuo_logit": "Residuo logit", "dc_apertura": "Dixon-Coles (ancla de apertura)"}  # fmt: skip
    for k, txt in parts.items():
        if v.startswith(k):
            m = v[len(k) + 1 :]
            label = {"prop": "proporcional", "shin": "Shin", "pot": "potencia"}.get(m, m)
            return f"{txt}, {label}" if m else txt
    return v


def short(m: dict) -> dict:
    rnd = lambda v: [round(x, 6) for x in v] if isinstance(v, list) else round(float(v), 6)  # noqa: E731
    return {k: rnd(v) for k, v in m.items() if isinstance(v, (float, int, list)) and not isinstance(v, bool)}


def market_preds(df, seasons) -> dict[str, pd.DataFrame]:
    mask = df["season"].isin(seasons)
    return {mk(pre, m): rf.market_probs(df, f"pin_{'close' if pre == 'cierre' else 'open'}", m).where(mask, np.nan)
            for pre in ("cierre", "apertura") for m in METHODS}  # fmt: skip


def multi_ref(df, preds, refs, n_boot) -> dict:
    """Métricas en el conjunto común con Δ frente a varias referencias: `d[ref]` = (media, IC)."""
    base = None
    for ref in refs:
        r = rf.evaluate(df, preds, reference=ref, n_boot=n_boot)
        if base is None:
            base = r
            for m in base["modelos"].values():
                m["d"] = {}
        for nm, m in r["modelos"].items():
            if "delta_logloss" in m:
                base["modelos"][nm]["d"][ref] = (m["delta_logloss"], m["delta_logloss_ic"])
    for m in base["modelos"].values():
        for k in ("delta_logloss", "delta_logloss_ic", "delta_rps", "delta_rps_ic"):
            m.pop(k, None)
    return base


# ----------------------------------------------------------------------------- ajuste


def run_tuning(df) -> None:
    h = ex.data_hash(df[df["season"] <= rf.TUNING[-1]])
    preds = market_preds(df, rf.TUNING)
    res = multi_ref(df, preds, [mk("cierre", "proporcional")], n_boot=500)
    for v, m in res["modelos"].items():
        mets = short({k: m[k] for k in ("logloss", "logloss_ic", "rps", "brier", "ece")}) | {"n": res["n"]}
        ex.log(v, "ajuste", {"normalizacion": v.rsplit("_", 1)[1]}, mets, h,
               notes=f"fase 4; normalización del margen; temporadas {rf.TUNING[0]}–{rf.TUNING[-1]}")  # fmt: skip
        print(f"{v:24s} logloss {m['logloss']:.5f} rps {m['rps']:.5f}")
    ll = {meth: res["modelos"][mk("cierre", meth)]["logloss"] for meth in METHODS}
    best = min(ll, key=ll.get)
    SELECTION.write_text(json.dumps({
        "regla": f"mínimo log loss del cierre de Pinnacle en ajuste {rf.TUNING[0]}–{rf.TUNING[-1]}",
        "hash_datos": h, "logloss_cierre": {k: round(v, 6) for k, v in ll.items()},
        "logloss_apertura": {k: round(res["modelos"][mk("apertura", k)]["logloss"], 6) for k in METHODS},
        "normalizacion": best, "umbrales_clv": list(THRESHOLDS)}, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
    print(f"Normalización elegida: {best} → {SELECTION.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- evaluación


def opening_features(df, n_jobs) -> pd.DataFrame:
    sel = json.loads(SEL2.read_text())["seleccion"]
    path = CACHE / f"fase4_X_apertura_{ex.data_hash(df)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    t = time.time()
    X = ft.build(df, "apertura", elo=Elo(**sel["elo"]["parametros"]), pi=PiRating(**sel["pi"]["parametros"]),
                 dc=DixonColes(**sel["dixon_coles"]["parametros"]), dc_from=DC_FROM, n_jobs=n_jobs)  # fmt: skip
    print(f"variables de apertura: {X.shape[1]} columnas ({time.time() - t:.0f} s)", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    X.to_pickle(path)
    return X


def run_evaluation(df, n_jobs, n_boot) -> None:
    norm = json.loads(SELECTION.read_text())["normalizacion"]
    sel3 = json.loads(SEL3.read_text())["seleccion"]
    h = ex.data_hash(df)
    X = opening_features(df, n_jobs)
    mask = df["season"].isin(rf.EVALUATION)
    preds = market_preds(df, rf.EVALUATION)
    preds["dc_apertura"] = X[["dc_home", "dc_draw", "dc_away"]].set_axis(rf.PROB, axis=1).where(mask, np.nan)
    for meth in METHODS:
        Xm = ft.with_opening(X, df, meth)
        for v in RESIDUALS:
            t = time.time()
            preds[f"{v}_{SHORT[meth]}"] = ml.walk_forward(df, Xm, v, sel3[v]["parametros"], rf.EVALUATION)
            print(f"{v} {meth}: {time.time() - t:.0f} s", flush=True)
    c_prop, c_sel, a_prop, a_sel = (mk("cierre", "proporcional"), mk("cierre", norm),
                                    mk("apertura", "proporcional"), mk("apertura", norm))  # fmt: skip
    refs = list(dict.fromkeys([c_prop, c_sel, a_prop, a_sel]))
    overall = multi_ref(df, preds, refs, n_boot)
    seasons = {s: multi_ref(df, {k: p.where(df["season"].eq(s), np.nan) for k, p in preds.items()}, [a_sel], 1000)
               for s in rf.EVALUATION}  # fmt: skip
    ok = np.logical_and.reduce([p[rf.PROB].notna().all(axis=1).to_numpy() for p in preds.values()])
    robust = drop_best(df, preds, ok, a_sel)
    for v, m in overall["modelos"].items():
        if v == "dc_apertura":
            continue  # descriptivo: Dixon-Coles ya está registrado en la fase 2
        mets = short({k: m[k] for k in ("logloss", "logloss_ic", "rps", "brier", "ece")})
        mets |= {f"d_{r}": round(d[0], 6) for r, d in m["d"].items()}
        mets |= {f"d_{r}_ic": [round(x, 6) for x in d[1]] for r, d in m["d"].items()}
        mets |= {"n": overall["n"], "semanas": overall["semanas"]}
        params = {"normalizacion": v.rsplit("_", 1)[1]}
        if v.startswith("residuo"):
            params |= sel3[v.rsplit("_", 1)[0]]["parametros"]
        ex.log(v, "evaluacion", params, mets, h,
               notes=f"fase 4; normalización del margen; {rf.EVALUATION[0]}–{rf.EVALUATION[-1]}; residuos sin reajuste ({SEL3.name})")  # fmt: skip
    clv_res = run_clv(df, X, preds, ok, norm, h, n_boot)
    write_report(df, overall, seasons, robust, clv_res, norm, refs, h)


def drop_best(df, preds, ok, ref) -> dict:
    """Δ log loss frente a `ref` quitando el 1 % de partidos más favorables a cada modelo."""
    y = mt.outcome_index(df.loc[ok, "result"])
    base = mt.log_loss(preds[ref].loc[ok, rf.PROB].to_numpy(float), y)
    out = {}
    for v, p in preds.items():
        if v == ref:
            continue
        d = np.sort(mt.log_loss(p.loc[ok, rf.PROB].to_numpy(float), y) - base)
        out[v] = (float(d.mean()), float(d[int(len(d) * 0.01):].mean()))
    return out


def run_clv(df, X, preds, ok, norm, h, n_boot) -> dict:
    """CLV histórico: apuestas a la apertura de Pinnacle frente a su cierre sin margen (`norm`)."""
    ev_mask = df["season"].isin(rf.EVALUATION).to_numpy() & ok
    sub = df[ev_mask]
    anchor = ft.opening_anchor(sub["date"])
    snaps = cl.history_snapshots(sub, anchor)
    dec = anchor.dt.tz_localize("UTC")
    models = [f"residuo_lgbm_{SHORT[norm]}", f"residuo_logit_{SHORT[norm]}", "dc_apertura"]
    week = dict(zip(sub["match_id"], sub["date"].dt.strftime("%G-%V"), strict=True))
    result = dict(zip(sub["match_id"], sub["result"].map({"H": "home", "D": "draw", "A": "away"}), strict=True))
    season = dict(zip(sub["match_id"], sub["season"], strict=True))
    out = {}
    for v in models:
        p = preds[v].loc[ev_mask, rf.PROB].set_axis(sub["match_id"].to_numpy())
        p["decision_utc"] = dec.to_numpy()
        for thr in THRESHOLDS:
            bets, info = cl.simulate(snaps, p, bet_book="pinnacle", close_book="pinnacle", threshold=thr, method=norm)
            cl_w = bets["event_key"].map(week).to_numpy()
            won = (bets["outcome"] == bets["event_key"].map(result)).to_numpy()
            s = cl.summarize(bets, cl_w, n_boot=n_boot, won=won)
            alt = cl.clv(bets[["event_key", "outcome", "price", "p_model", "ev", "price_captured_utc"]],
                         cl.wide_1x2(cl.closing(snaps), "pinnacle"), method="proporcional")  # fmt: skip
            s["clv_ev_prop"] = float(alt["clv_ev"].mean()) if len(alt) else float("nan")
            if len(bets):
                srt = np.sort(bets["clv_ev"].to_numpy())
                s["clv_sin_1pc"] = float(srt[: len(srt) - max(1, len(srt) // 100)].mean())
                by = bets.assign(season=bets["event_key"].map(season)).groupby("season")["clv_ev"]
                s["por_temporada"] = {k: [int(len(g)), round(float(g.mean()), 5)] for k, g in by}
                s["por_resultado"] = bets.groupby("outcome").size().to_dict()
            s |= {"info": info}
            out[(v, thr)] = s
            ex.log(f"clv_{v}", "evaluacion", {"umbral": thr, "normalizacion_cierre": norm, "casa": "pinnacle",
                                              "momento": "apertura football-data"},
                   short({k: s[k] for k in s if k in ("n", "semanas", "clv_pos", "ev_medio", "clv_ev", "clv_ev_ic",
                                                       "clv_log", "clv_log_ic", "roi", "roi_ic", "clv_sin_1pc",
                                                       "clv_ev_prop")}), h,
                   notes="fase 4; CLV histórico, apuestas planas de 1 unidad a la apertura de Pinnacle")  # fmt: skip
            print(f"CLV {v} u={thr}: n={s.get('n')} clv={s.get('clv_ev', float('nan')):+.4f} "
                  f"roi={s.get('roi', float('nan')):+.4f}", flush=True)  # fmt: skip
    return out


# ----------------------------------------------------------------------------- informe


def ci(x, nd=4):
    v, (lo, hi) = x
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])  # fmt: skip


def write_report(df, overall, seasons, robust, clv_res, norm, refs, h) -> None:
    sel = json.loads(SELECTION.read_text())
    tun = md_table(["Normalización", "Log loss cierre", "Log loss apertura"],
                   [[k, f"{sel['logloss_cierre'][k]:.5f}", f"{sel['logloss_apertura'][k]:.5f}"] for k in METHODS])  # fmt: skip
    rows = []
    for v, m in overall["modelos"].items():
        cells = [name(v), f"{m['logloss']:.5f} [{m['logloss_ic'][0]:.4f}, {m['logloss_ic'][1]:.4f}]"]
        cells += [ci(m["d"][r], 5) if r in m["d"] else "—" for r in refs]
        cells.append(f"{m['ece']:.4f}")
        rows.append(cells)
    main = md_table(["Modelo", "Log loss [IC 95 %]"] + [f"Δ vs {name(r)}" for r in refs] + ["ECE"], rows)
    a_sel = mk("apertura", norm)
    res_names = [v for v in overall["modelos"] if v.startswith("residuo")]
    srows = [[s, str(r["n"])] + [ci(r["modelos"][v]["d"][a_sel], 5) for v in res_names] for s, r in seasons.items()]
    by_season = md_table(["Temporada", "N"] + [name(v) for v in res_names], srows)
    rrows = [[name(v), f"{a:+.5f}", f"{b:+.5f}"] for v, (a, b) in robust.items()]
    rob = md_table(["Modelo", f"Δ vs {name(a_sel)}", "Δ sin el 1 % más favorable"], rrows)
    crows = []
    for (v, thr), s in clv_res.items():
        if s["n"] == 0:
            crows.append([name(v), f"{thr:.0%}", "0", "—", "—", "—", "—", "—"])
            continue
        crows.append([name(v), f"{thr:.0%}", str(s["n"]), ci((s["clv_ev"], s["clv_ev_ic"])),
                      f"{s['clv_sin_1pc']:+.4f}", f"{s['clv_pos']:.1%}", f"{s['clv_ev_prop']:+.4f}",
                      ci((s["roi"], s["roi_ic"]))])  # fmt: skip
    clv_t = md_table(["Modelo", "Umbral", "Apuestas", f"CLV (cierre {norm}) [IC 95 %]", "CLV sin el 1 % mejor",
                      "% CLV > 0", "CLV (cierre proporcional)", "ROI plano [IC 95 %]"], crows)  # fmt: skip
    seas_rows = []
    for (v, thr), s in clv_res.items():
        if s["n"]:
            seas_rows.append([name(v), f"{thr:.0%}"] + [f"{s['por_temporada'].get(x, [0, float('nan')])[1]:+.4f} "
                                                        f"({s['por_temporada'].get(x, [0])[0]})" for x in rf.EVALUATION])  # fmt: skip
    clv_s = md_table(["Modelo", "Umbral"] + rf.EVALUATION, seas_rows)
    reg = ex.read()
    n4 = sum(1 for r in reg if "fase 4" in r.get("notas", ""))
    text = f"""# Fase 4: normalización del margen y CLV histórico (resultados generados)

Generado con `scripts/fase4_margen_clv.py evaluacion` el {date.today().isoformat()}. Solo agregados.
Hash de los datos: `{h}`. Filas de la fase 4 en el registro hasta hoy: {n4}. Lectura y auditoría en
[`clv.md`](clv.md).

## Ajuste (2013/14–2016/17): qué normalización usar

Regla fijada antes de mirar: la de menor log loss del cierre de Pinnacle. **Elegida: {norm}.**

{tun}

## Evaluación 2017/18–2024/25 ({overall['n']} partidos, {overall['semanas']} semanas)

Δ log loss emparejada (IC 95 % por bootstrap de semanas); **Δ > 0 es peor que la referencia**.
Los residuos usan los hiperparámetros de la fase 3 sin reajustar; solo cambia la normalización de la
apertura que reciben (como desplazamiento y como variable).

{main}

### Residuos frente a la apertura con {norm}, por temporada

{by_season}

### Robustez: quitar el 1 % de partidos más favorables

{rob}

## CLV histórico de apuestas simuladas a la apertura de Pinnacle

Una apuesta plana por partido, al resultado de mayor valor esperado p_modelo · cuota_apertura − 1,
si supera el umbral. CLV = cuota tomada · p_cierre sin margen − 1. Pinnacle no opera en España:
es una medida de calidad del precio, no una estrategia ejecutable. Sin comisión (casa, no exchange).

{clv_t}

CLV medio por temporada (apuestas):

{clv_s}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"Informe en {REPORT.relative_to(ROOT)}")


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
        if args.modo == "ajuste":
            run_tuning(df)
        else:
            run_evaluation(df, args.jobs, args.boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
