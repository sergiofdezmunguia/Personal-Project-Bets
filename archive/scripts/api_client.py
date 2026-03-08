import time
import requests

from .config import BESOCCER_API_KEY, BESOCCER_BASE_URL


class BeSoccerClient:
    """Cliente para la API de BeSoccer basada en parámetros `req=...`.

    Ejemplos de uso:
      - categorias: req=categories&filter=my_leagues
      - partidos del día: req=matchsday
      - partidos por liga: req=matchs&league=1
      - clasificación: req=tables&league=1&group=1
    """

    def __init__(self, api_key=None, base_url=None, rate_limit_per_minute=60):
        if api_key is None:
            api_key = BESOCCER_API_KEY
        if base_url is None:
            base_url = BESOCCER_BASE_URL

        if not api_key:
            raise ValueError("BESOCCER_API_KEY no definido. Configúralo en el .env.")

        # La base real para esta API suele ser el script api.php
        # Ej: http://apiclient.besoccerapps.com/scripts/api/api.php
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_call_timestamp = 0.0

    def _throttle(self):
        min_interval = 60.0 / float(self.rate_limit_per_minute)
        now = time.time()
        elapsed = now - self._last_call_timestamp
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_timestamp = time.time()

    def _get(self, params=None):
        """Lanza un GET simple contra api.php con los parámetros dados."""
        self._throttle()
        url = self.base_url  # ya debe apuntar a .../api.php
        if params is None:
            params = {}
        params.setdefault("key", self.api_key)
        params.setdefault("format", "json")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    # --- Métodos de alto nivel para los endpoints que has mencionado ---

    def get_my_leagues(self):
        return self._get({
            "req": "categories",
            "filter": "my_leagues",
            "tz": "Europe/Madrid",
        })

    def get_league_status(self, league_id, year):
        return self._get({
            "req": "league_status",
            "id": league_id,
            "year": year,
            "tz": "Europe/Madrid",
        })

    def get_competitions_detail(self, competition_ids):
        comp_str = ",".join(str(c) for c in competition_ids)
        return self._get({
            "req": "data_competitions",
            "competitions": comp_str,
            "tz": "Europe/Madrid",
        })

    def get_countries_competitions(self, filter_="eu"):
        return self._get({
            "req": "countries_competitions",
            "filter": filter_,
            "tz": "Europe/Madrid",
        })

    def get_matches_day(self):
        return self._get({
            "req": "matchsday",
            "tz": "Europe/Madrid",
        })

    def get_matches_by_league(self, league_id, **extra_params):
        params = {
            "req": "matchs",
            "league": league_id,
            "tz": "Europe/Madrid",
        }
        params.update(extra_params)
        return self._get(params)

    def get_league_table(self, league_id, group=1):
        return self._get({
            "req": "tables",
            "league": league_id,
            "group": group,
        })
