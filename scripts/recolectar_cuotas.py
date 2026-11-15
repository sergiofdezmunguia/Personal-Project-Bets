"""Recolector de cuotas propias con hora de captura (fase 4). Solo lee cuotas; no apuesta.

    python scripts/recolectar_cuotas.py football-data            # fixtures.csv, sin clave
    python scripts/recolectar_cuotas.py the-odds-api --ligas SP1,E0   # necesita ODDS_API_KEY en .env
    python scripts/recolectar_cuotas.py resumen                  # qué hay guardado

Guarda en `data/odds/cuotas.sqlite` (excluido de git) o en `BETS_ODDS_DIR`. Repetir una captura
sin cambios no añade filas. Pensado para lanzarse desde cron o el Programador de tareas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import odds_collector as oc  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fuente", choices=["football-data", "the-odds-api", "resumen"])
    ap.add_argument("--dir", type=Path, default=None, help="carpeta de datos (por defecto data/odds)")
    ap.add_argument("--ligas", default="SP1,SP2,E0,E1,E2,D1,I1,F1")
    args = ap.parse_args(argv)
    store = oc.Store((args.dir or oc.default_dir()) / "cuotas.sqlite")
    if args.fuente == "football-data":
        print(json.dumps(oc.collect_football_data(store), ensure_ascii=False))
    elif args.fuente == "the-odds-api":
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        print(json.dumps(oc.collect_odds_api(store, args.ligas.split(",")), ensure_ascii=False))
    else:
        s = store.snapshots()
        print(f"{len(s)} instantáneas, {s['event_key'].nunique() if len(s) else 0} partidos")
        if len(s):
            print(s.groupby(["source", "market", "bookmaker"]).size().to_string())
        print(store.captures().tail(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
