"""Fase 2: modelos de referencia 1X2 por ventanas temporales.

    python scripts/fase2_referencias.py ajuste       # rejilla en 2013/14–2016/17 → seleccion.json
    python scripts/fase2_referencias.py evaluacion   # 2017/18–2024/25 con la selección → docs/

Cada variante evaluada añade una fila a `experiments/registro.jsonl` (nunca se sobrescribe).
La temporada 2025/26 no se lee (ver `src/referencias.py`).

Salidas que sí se suben (solo agregados): `experiments/registro.jsonl`,
`experiments/seleccion.json`, `docs/referencias.md` y `docs/figuras/calibracion-referencias.png`.
"""

from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")  # WSL: BLAS con todos los hilos va muchísimo más lento

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import experiments as ex  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import referencias as rf  # noqa: E402

RAW = ROOT / "data" / "raw" / "football-data"
SELECTION = ROOT / "experiments" / "seleccion.json"
REPORT = ROOT / "docs" / "referencias.md"
FIGURE = ROOT / "docs" / "figuras" / "calibracion-referencias.png"
REF = "mercado_cierre"
NAMES = {"mercado_cierre": "Mercado, cierre (Pinnacle sin margen)",
         "mercado_apertura": "Mercado, apertura (Pinnacle sin margen)",
         "elo": "Elo", "pi": "Pi-ratings", "dixon_coles": "Dixon-Coles"}  # fmt: skip


def grid() -> list[tuple[str, dict]]:
    """Rejilla de ajuste, fijada antes de mirar resultados. Todas las variantes se registran."""
    out: list[tuple[str, dict]] = [("mercado_apertura", {})]
    for k, gm, reg in itertools.product((15, 20, 30), (True, False), (0.0, 0.2)):
        out.append(("elo", {"k": k, "goal_mult": gm, "season_regress": reg}))
    for lam, gamma in itertools.product((0.035, 0.06), (0.5, 0.7)):
        out.append(("pi", {"lam": lam, "gamma": gamma}))
    for xi, new in itertools.product((0.001, 0.002, 0.004), (0.0, 0.15)):
        out.append(("dixon_coles", {"xi": xi, "newcomer": new}))
    return out


def short(m: dict) -> dict:
    """Métricas para el registro, redondeadas."""
    keep = ("logloss", "logloss_ic", "rps", "rps_ic", "brier", "ece", "delta_logloss",
            "delta_logloss_ic", "delta_rps", "delta_rps_ic")  # fmt: skip
    rnd = lambda v: [round(x, 5) for x in v] if isinstance(v, list) else round(float(v), 5)  # noqa: E731
    return {k: rnd(m[k]) for k in keep if k in m}


# ----------------------------------------------------------------------------- ajuste


def run_tuning(df: pd.DataFrame, n_jobs: int) -> None:
    h = ex.data_hash(df[df["season"].isin([rf.BURN_IN, *rf.TUNING])])
    market = rf.predict(df, REF, {}, rf.TUNING)
    base = rf.evaluate(df, {REF: market}, n_boot=500)
    ex.log(REF, "ajuste", {}, short(base["modelos"][REF]) | {"n": base["n"]}, h)
    best: dict[str, tuple[float, dict]] = {}
    for variant, params in grid():
        t = time.time()
        p = rf.predict(df, variant, params, rf.TUNING, n_jobs=n_jobs)
        res = rf.evaluate(df, {REF: market, variant: p}, n_boot=500)
        m = res["modelos"][variant]
        ex.log(variant, "ajuste", params, short(m) | {"n": res["n"]}, h,
               notes=f"temporadas {rf.TUNING[0]}–{rf.TUNING[-1]}")  # fmt: skip
        print(f"{variant:16s} {json.dumps(params):60s} logloss {m['logloss']:.4f} "
              f"Δ {m['delta_logloss']:+.4f} n {res['n']} ({time.time() - t:.0f} s)")  # fmt: skip
        if variant not in best or m["logloss"] < best[variant][0]:
            best[variant] = (m["logloss"], params)
    sel = {v: {"parametros": p, "logloss_ajuste": round(ll, 5)} for v, (ll, p) in best.items()}
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps({"regla": "mínimo log loss en ajuste " + f"{rf.TUNING[0]}–{rf.TUNING[-1]}",
                                     "hash_datos": h, "seleccion": sel}, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
    print(f"Selección guardada en {SELECTION.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- evaluación


def fmt_ci(m: dict, key: str, nd: int = 4, sign: bool = False) -> str:
    lo, hi = m[f"{key}_ic"]
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(m[key])} [{f.format(lo)}, {f.format(hi)}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])  # fmt: skip


def plot_calibration(cal: dict[str, pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # paleta validada (dataviz)
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, outcome in zip(axes, ("local", "empate", "visitante"), strict=True):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], color="#9a9893", lw=1, ls="--", zorder=1)
        for (name, t), c, mk in zip(cal.items(), colors, markers, strict=False):
            g = t[(t["resultado"] == outcome) & (t["n"] >= 30)]
            ax.plot(g["prevista"], g["observada"], color=c, lw=2, marker=mk, ms=6, label=NAMES[name],
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=2)  # fmt: skip
        ax.set_title(f"Gana {outcome}" if outcome != "empate" else "Empate", color="#0b0b0b", fontsize=11)
        ax.set_xlabel("Probabilidad prevista", color="#52514e")
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, 0.95)
        ax.grid(color="#e6e5e1", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#d3d2cd")
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("Frecuencia observada", color="#52514e")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Calibración por ventanas, 2017/18–2024/25 (tramos con ≥ 30 partidos)", color="#0b0b0b")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=130, facecolor=fig.get_facecolor())


def run_evaluation(df: pd.DataFrame, n_jobs: int, n_boot: int) -> None:
    sel = json.loads(SELECTION.read_text())
    h = ex.data_hash(df)
    configs = {REF: {}, **{v: s["parametros"] for v, s in sel["seleccion"].items()}}
    preds = {}
    for v, params in configs.items():
        t = time.time()
        preds[v] = rf.predict(df, v, params, rf.EVALUATION, n_jobs=n_jobs)
        print(f"{v}: {time.time() - t:.0f} s")
    overall = rf.evaluate(df, preds, n_boot=n_boot)
    recent = ["2021/22", "2022/23", "2023/24", "2024/25"]
    sub = {s: rf.evaluate(df, {k: p.where(df["season"].eq(s), np.nan) for k, p in preds.items()}, n_boot=1000)
           for s in rf.EVALUATION}  # fmt: skip
    rec = rf.evaluate(df, {k: p.where(df["season"].isin(recent), np.nan) for k, p in preds.items()},
                      n_boot=n_boot)  # fmt: skip
    by_league = {}
    for lg in sorted(df["league"].unique()):
        r = rf.evaluate(df, {k: p.where(df["league"].eq(lg), np.nan) for k, p in preds.items()}, n_boot=500)
        by_league[lg] = r
    for v in configs:
        ex.log(v, "evaluacion", configs[v],
               short(overall["modelos"][v]) | {"n": overall["n"], "semanas": overall["semanas"]}, h,
               notes=f"temporadas {rf.EVALUATION[0]}–{rf.EVALUATION[-1]}; selección de {SELECTION.name}")  # fmt: skip

    ok = np.logical_and.reduce([p[rf.PROB].notna().all(axis=1).to_numpy() for p in preds.values()])
    y = mt.outcome_index(df.loc[ok, "result"])
    cal = {v: mt.calibration_table(p.loc[ok, rf.PROB].to_numpy(float), y) for v, p in preds.items()}
    plot_calibration(cal)
    write_report(overall, rec, sub, by_league, cal, configs, h)


def write_report(overall, rec, sub, by_league, cal, configs, h) -> None:
    order = list(configs)
    rows = []
    for v in order:
        m = overall["modelos"][v]
        d = (fmt_ci(m, "delta_logloss", sign=True), fmt_ci(m, "delta_rps", sign=True)) if v != REF else ("—", "—")
        rows.append([NAMES[v], fmt_ci(m, "logloss"), fmt_ci(m, "rps"), d[0], d[1], f"{m['ece']:.4f}"])
    main = md_table(["Modelo", "Log loss [IC 95 %]", "RPS [IC 95 %]", "Δ log loss vs cierre",
                     "Δ RPS vs cierre", "ECE"], rows)  # fmt: skip
    rows = []
    for v in order:
        m = rec["modelos"][v]
        d = fmt_ci(m, "delta_logloss", sign=True) if v != REF else "—"
        rows.append([NAMES[v], fmt_ci(m, "logloss"), fmt_ci(m, "rps"), d])
    recent = md_table(["Modelo", "Log loss", "RPS", "Δ log loss vs cierre"], rows)
    others = [v for v in order if v != REF]
    season_rows = []
    for s, r in sub.items():
        cells = [s, str(r["n"]), f"{r['modelos'][REF]['logloss']:.4f}"]
        cells += [fmt_ci(r["modelos"][v], "delta_logloss", sign=True) for v in others]
        season_rows.append(cells)
    seasons = md_table(["Temporada", "N", "Log loss cierre"] + [f"Δ {NAMES[v]}" for v in others], season_rows)
    league_rows = []
    for lg, r in by_league.items():
        cells = [lg, str(r["n"]), f"{r['modelos'][REF]['logloss']:.4f}"]
        cells += [f"{r['modelos'][v]['delta_logloss']:+.4f}" for v in others]
        league_rows.append(cells)
    leagues = md_table(["Liga", "N", "Log loss cierre"] + [f"Δ {NAMES[v]}" for v in others], league_rows)
    cal_rows = []
    for outcome in ("local", "empate", "visitante"):
        t0 = cal[REF][cal[REF]["resultado"] == outcome]
        for tramo in t0["tramo"]:
            cells = [outcome, tramo]
            for v in order:
                t = cal[v]
                r = t[(t["resultado"] == outcome) & (t["tramo"] == tramo)]
                cells.append("—" if r.empty or r["n"].iloc[0] < 30 else
                             f"{r['prevista'].iloc[0]:.3f} → {r['observada'].iloc[0]:.3f} ({r['n'].iloc[0]})")  # fmt: skip
            cal_rows.append(cells)
    calib = md_table(["Resultado", "Tramo"] + [NAMES[v] for v in order], cal_rows)
    params = "\n".join(f"- **{NAMES[v]}:** `{json.dumps(p, ensure_ascii=False)}`" for v, p in configs.items() if p)
    n_tried = sum(1 for r in ex.read() if r["fase"] == "ajuste" and r["variante"] != REF)
    n_eval = sum(1 for r in ex.read() if r["fase"] == "evaluacion") // max(len(configs), 1)
    text = f"""# Modelos de referencia 1X2 (fase 2)

Generado con `scripts/fase2_referencias.py evaluacion` el {date.today().isoformat()}. Solo agregados;
los datos de partidos no se suben. Hash de los datos evaluados: `{h}`.

## Cómo se ha medido

- **Ventanas temporales.** Elo y pi-ratings se actualizan tras cada día de partidos y su traducción a
  1X2 (logit ordenado) se ajusta, para cada temporada, solo con las anteriores. Dixon-Coles se
  reajusta cada lunes con los partidos anteriores a ese lunes (4 años como máximo, peso
  exp(−ξ·días)). Ninguna predicción usa partidos de su mismo día ni posteriores (tests en
  `tests/test_referencias.py`).
- **Temporadas.** 2012/13 arranca los ratings. 2013/14–2016/17 sirvió para elegir hiperparámetros
  ({n_tried} variantes registradas, regla: mínimo log loss). 2017/18–2024/25 es la evaluación (evaluaciones registradas
  hasta hoy: {n_eval}). **2025/26 no se ha leído.**
- **Conjunto común.** {overall['n']} partidos de 8 ligas con probabilidad de todos los modelos
  ({overall['n_descartados']} descartados por faltar alguna cuota). La referencia es el cierre de
  Pinnacle sin margen (normalización proporcional).
- **Intervalos.** IC 95 % por bootstrap de semanas ({overall['semanas']} semanas ISO, 2.000
  remuestras): los partidos de una misma jornada no son independientes. Las diferencias (Δ) son
  emparejadas, partido a partido, frente al cierre; **Δ > 0 significa peor que el mercado.**

Hiperparámetros elegidos en el ajuste:

{params}

## Resultado: 2017/18–2024/25

{main}

ECE: error de calibración esperado (10 tramos, media de local, empate y visitante).

### Solo 2021/22–2024/25 (ventana de validación del plan)

{recent}

## Por temporada (Δ log loss frente al cierre, IC 95 %)

{seasons}

2019/20 y 2020/21 incluyen partidos sin público.

## Por liga (Δ log loss frente al cierre)

{leagues}

## Calibración

Probabilidad media prevista → frecuencia observada (partidos), por tramos de 0,1. «—»: menos de
30 partidos en el tramo.

![Curvas de calibración](figuras/calibracion-referencias.png)

{calib}

## Lectura

{interpretation(overall, sub, others)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"Informe en {REPORT.relative_to(ROOT)}")


def interpretation(overall, sub, others) -> str:
    lines = []
    for v in others:
        m = overall["modelos"][v]
        lo, hi = m["delta_logloss_ic"]
        if lo > 0:
            verdict = "peor que el cierre, con el intervalo entero por encima de 0"
        elif hi < 0:
            verdict = "**mejor que el cierre: sospechoso hasta auditarlo**"
        else:
            verdict = "indistinguible del cierre con esta muestra"
        worse = sum(1 for r in sub.values() if r["modelos"][v]["delta_logloss_ic"][0] > 0)
        better = sum(1 for r in sub.values() if r["modelos"][v]["delta_logloss_ic"][1] < 0)
        lines.append(f"- **{NAMES[v]}:** Δ log loss {m['delta_logloss']:+.4f}, {verdict}. Por temporadas: "
                     f"peor con IC por encima de 0 en {worse} de {len(sub)}; mejor con IC por debajo de 0 en {better}.")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["ajuste", "evaluacion"])
    ap.add_argument("--datos", type=Path, default=RAW, help="carpeta con los CSV de football-data")
    ap.add_argument("--jobs", type=int, default=5, help="procesos para Dixon-Coles (uno por país)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    with threadpool_limits(1):
        df = rf.load_history(args.datos)
        print(f"{len(df)} partidos, {df['season'].min()} → {df['season'].max()}")
        if args.modo == "ajuste":
            run_tuning(df, args.jobs)
        else:
            run_evaluation(df, args.jobs, args.boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
