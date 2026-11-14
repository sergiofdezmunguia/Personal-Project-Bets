"""Unifica el histórico de football-data y escribe el informe de calidad.

    python scripts/informe_calidad.py

Salidas:
- data/processed/football_data.csv.gz: la tabla unificada (no se sube: datos de terceros).
- docs/calidad-datos.md: solo recuentos y proporciones (sí se sube).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import football_data as fd  # noqa: E402

RAW = ROOT / "data" / "raw" / "football-data"
OUT = ROOT / "data" / "processed" / "football_data.csv.gz"
REPORT = ROOT / "docs" / "calidad-datos.md"


def pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def table(df, fmt=pct) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in cols]) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for idx, r in df.iterrows():
        cells = [fmt(v) if isinstance(v, float) else str(v) for v in r.tolist()]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join([head, sep, *rows])


def main() -> int:
    df = fd.load_all(RAW)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    q = fd.quality(df)

    per = q["por_temporada"]
    cover = per.drop(columns="partidos").groupby(level="season").mean()
    counts = per["partidos"].unstack("league").fillna(0).astype(int)
    ref = q["referencia"]
    ref.columns = ["sin referencia" if c is None or c != c else c for c in ref.columns]
    margin = q["margen_referencia"]
    # Temporadas cerradas con un nº de partidos distinto del habitual de su liga (p. ej. COVID).
    usual = per["partidos"].groupby(level="league").agg(lambda s: s.mode().iloc[0])
    closed = per.index.get_level_values("season") != df["season"].max()
    odd = per[closed & (per["partidos"].values != usual.reindex(per.index.get_level_values("league")).values)]
    odd = odd.assign(partidos=odd["partidos"].astype(str))
    odd.index = [f"{lg} {se}" for lg, se in odd.index]

    lines = [
        "# Calidad de datos: football-data.co.uk",
        "",
        f"Generado con `scripts/informe_calidad.py` el {date.today().isoformat()}. Solo recuentos;",
        "los CSV no se suben al repositorio.",
        "",
        "## Resumen",
        "",
        f"- Partidos: **{q['partidos']:,}**".replace(",", ".") + f" en {df['league'].nunique()} ligas y "
        f"{df['season'].nunique()} temporadas ({df['season'].min()} → {df['season'].max()}).",
        f"- Duplicados por `match_id`: {q['duplicados']}. Fechas ilegibles: {q['fechas_ilegibles']}.",
        f"- Sin resultado: {q['sin_resultado']} (partidos aún no jugados o aplazados).",
        f"- Resultado incoherente con los goles: {q['resultado_incoherente']}. "
        f"Cuotas ≤ 1 (erratas de la fuente; no cuentan como referencia): {q['cuotas_imposibles']}.",
        f"- Equipos distintos: {df['home_team_id'].nunique()}; ningún nombre se repite entre países.",
        "- `PP*` (2026/27) no es Pinnacle: su margen es de otro tipo de casa (ver `docs/decisions.md`).",
        "",
        "## Partidos por liga y temporada",
        "",
        table(counts, fmt=str),
        "",
        "## Cobertura media por temporada",
        "",
        "Proporción de partidos con el dato (media de las ligas).",
        "",
        table(cover),
        "",
        "## Fuente de la referencia de cierre",
        "",
        "Pinnacle si existe; si no, Betfair Exchange; si no, la media del mercado.",
        "",
        table(ref, fmt=str),
        "",
        "## Margen medio de la referencia",
        "",
        table(margin, fmt=lambda v: "—" if v != v else f"{v * 100:.1f} %"),
        "",
        "## Temporadas a revisar",
        "",
        table(odd) if len(odd) else "Ninguna.",
        "",
        "League One y Ligue 1 de 2019/20 se cortaron por el COVID. Ligue 1 juega con 18 equipos",
        "(306 partidos) desde 2023/24: es un cambio de formato, no un hueco.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{q['partidos']} partidos → {OUT.relative_to(ROOT)}; informe en {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
