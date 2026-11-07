"""Synthetic league with the schema soccerdata's Understat reader saves to CSV.

Tests never touch the network, so they do not depend on Understat being up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = [f"Team {name}" for name in "ABCDEFGHIJ"]


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round robin: every team plays once per round."""
    rotation = list(teams)
    rounds = []
    for _ in range(len(teams) - 1):
        half = len(teams) // 2
        rounds.append(list(zip(rotation[:half], reversed(rotation[half:]))))
        rotation.insert(1, rotation.pop())
    return rounds + [[(away, home) for home, away in fixtures] for fixtures in rounds]


def make_league(seed: int = 7, seasons: tuple[int, ...] = (2023, 2024)):
    rng = np.random.default_rng(seed)
    schedule, stats = [], []
    game_id = 1000
    for season in seasons:
        for week, fixtures in enumerate(round_robin(TEAMS)):
            saturday = pd.Timestamp(f"{season}-08-19") + pd.Timedelta(weeks=week)
            for slot, (home, away) in enumerate(fixtures):
                game_id += 1
                # Several kick-off times on each of two days per round.
                kickoff = saturday + pd.Timedelta(days=slot % 2, hours=14 + 2 * (slot % 3))
                if week == 2 and slot == 0:
                    # Postponed to a midweek date: game_id no longer follows date order.
                    kickoff += pd.Timedelta(weeks=5, days=3)
                home_goals, away_goals = int(rng.poisson(1.5)), int(rng.poisson(1.2))
                keys = {
                    "league": "ESP-La Liga",
                    "season": f"{season % 100}{season % 100 + 1}",
                    "game": f"{kickoff.date()} {home}-{away}",
                    "league_id": 1,
                    "season_id": season,
                    "game_id": game_id,
                    "date": kickoff,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
                schedule.append({**keys, "has_data": True})
                stats.append(
                    {
                        **keys,
                        "home_points": 3 if home_goals > away_goals else int(home_goals == away_goals),
                        "away_points": 3 if away_goals > home_goals else int(home_goals == away_goals),
                        "home_xg": rng.gamma(4, 0.35),
                        "away_xg": rng.gamma(4, 0.3),
                        "home_ppda": rng.uniform(6, 16),
                        "away_ppda": rng.uniform(6, 16),
                        "home_deep_completions": int(rng.integers(1, 15)),
                        "away_deep_completions": int(rng.integers(1, 15)),
                    }
                )
    return pd.DataFrame(schedule), pd.DataFrame(stats)


@pytest.fixture
def league() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_league()
