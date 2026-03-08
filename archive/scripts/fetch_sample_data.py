"""Script para hacer unas cuantas llamadas de ejemplo a la API de BeSoccer
y guardar las respuestas en data/raw para analizarlas con calma.

Usa los métodos de `BeSoccerClient`:
  - get_my_leagues
  - get_league_status
  - get_matches_day
  - get_matches_by_league
  - get_league_table
"""

from pathlib import Path
import json
from datetime import datetime

from src.api_client import BeSoccerClient
from src.config import RAW_DATA_DIR


def save_json(obj, name):  # type: (object, str) -> Path
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(RAW_DATA_DIR) / ("%s_%s.json" % (name, ts))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("Guardado %s en %s" % (name, path))
    return path


def main():
    client = BeSoccerClient()

    # 1) Tus ligas / categorías
    my_leagues = client.get_my_leagues()
    save_json(my_leagues, "my_leagues")

    # 2) Estado de la liga 1 (asumimos que es la interesante; se puede cambiar)
    league_status = client.get_league_status(league_id=1, year=2023)
    save_json(league_status, "league_status_1_2023")

    # 3) Partidos del día
    matchs_day = client.get_matches_day()
    save_json(matchs_day, "matchsday")

    # 4) Partidos de la liga 1 (parámetros extra se ajustarán según doc)
    matchs_league = client.get_matches_by_league(league_id=1)
    save_json(matchs_league, "matchs_league_1")

    # 5) Clasificación de la liga 1, grupo 1
    table = client.get_league_table(league_id=1, group=1)
    save_json(table, "tables_league_1_group_1")


if __name__ == "__main__":
    main()
