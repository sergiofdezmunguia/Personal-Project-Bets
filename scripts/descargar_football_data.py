"""Descarga el histórico de football-data.co.uk a data/raw/football-data/.

Lo ejecuta Sergio desde su terminal, no un agente: el robots.txt de football-data bloquea a los
bots de IA (ClaudeBot, Claude-Web, Anthropic-AI) y se respeta.

    python scripts/descargar_football_data.py               # ligas y temporadas por defecto
    python scripts/descargar_football_data.py --ligas SP1 E0 --desde 2019
    python scripts/descargar_football_data.py --forzar      # vuelve a bajar la temporada en curso

Solo biblioteca estándar. Una petición por segundo; no vuelve a bajar lo que ya existe salvo la
temporada en curso (que cambia cada semana) o con --forzar.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LIGAS = {
    "SP1": "La Liga",
    "SP2": "LaLiga 2",
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
DESDE = 2012  # Pinnacle apertura/cierre desde 2012/13 (docs/plan-modelo.md)
DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw" / "football-data"
PAUSA_S = 1.0


def temporada_actual(hoy: date) -> int:
    """Año de inicio de la temporada en curso (empieza en julio)."""
    return hoy.year if hoy.month >= 7 else hoy.year - 1


def codigo(inicio: int) -> str:
    return f"{inicio % 100:02d}{(inicio + 1) % 100:02d}"  # 2012 → "1213"


def descargar(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib (uso personal)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ligas", nargs="+", default=list(LIGAS), choices=list(LIGAS))
    ap.add_argument("--desde", type=int, default=DESDE, help="año de inicio de la primera temporada")
    ap.add_argument("--forzar", action="store_true", help="vuelve a bajar todo")
    args = ap.parse_args(argv)

    actual = temporada_actual(date.today())
    DESTINO.mkdir(parents=True, exist_ok=True)
    nuevos = saltados = faltan = 0
    for liga in args.ligas:
        for inicio in range(args.desde, actual + 1):
            temp = codigo(inicio)
            fichero = DESTINO / f"{liga}_{temp}.csv"
            if fichero.exists() and not args.forzar and inicio != actual:
                saltados += 1
                continue
            datos = descargar(BASE.format(season=temp, league=liga))
            time.sleep(PAUSA_S)
            if not datos or not datos.strip():
                faltan += 1
                print(f"  {liga} {temp}: no publicado")
                continue
            fichero.write_bytes(datos)
            nuevos += 1
            filas = datos.count(b"\n") - 1
            print(f"  {liga} {temp} ({LIGAS[liga]}): {filas} partidos")
    print(f"\nListo: {nuevos} descargados, {saltados} ya estaban, {faltan} sin publicar → {DESTINO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
