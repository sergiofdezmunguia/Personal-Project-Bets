"""Unified data extraction pipeline for football betting ML.

This script prioritizes sources that currently work in the environment and saves
all outputs under data/raw.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import soccerdata as sd

from src.config import RAW_DATA_DIR


def save_csv(df, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(f"  - Saved: {filepath}")


def extract_understat(leagues: list[str], seasons: list[str]) -> dict[str, str]:
    print("\n[Understat] Starting extraction...")
    understat = sd.Understat(leagues=leagues, seasons=seasons)

    schedule = understat.read_schedule()
    team_stats = understat.read_team_match_stats()
    players = understat.read_player_season_stats()

    save_csv(schedule, RAW_DATA_DIR / "understat_schedule.csv")
    save_csv(team_stats, RAW_DATA_DIR / "understat_team_match_stats.csv")
    save_csv(players, RAW_DATA_DIR / "understat_players.csv")

    return {
        "status": "ok",
        "schedule_rows": str(len(schedule)),
        "team_stats_rows": str(len(team_stats)),
        "players_rows": str(len(players)),
    }


def extract_clubelo(snapshot_date: str) -> dict[str, str]:
    print("\n[ClubElo] Starting extraction...")
    elo = sd.ClubElo()
    ratings = elo.read_by_date(snapshot_date)
    save_csv(ratings, RAW_DATA_DIR / "clubelo_ratings.csv")
    return {"status": "ok", "rows": str(len(ratings)), "snapshot_date": snapshot_date}


def main() -> None:
    # These values are intentionally explicit to keep the run reproducible.
    leagues = ["ESP-La Liga", "ENG-Premier League"]
    seasons = ["2023", "2024"]
    snapshot_date = "2024-01-01"

    summary: dict[str, dict[str, str]] = {}

    tasks = [
        ("understat", lambda: extract_understat(leagues=leagues, seasons=seasons)),
        ("clubelo", lambda: extract_clubelo(snapshot_date=snapshot_date)),
    ]

    print("=== Football Data Extraction Pipeline ===")
    for name, fn in tasks:
        try:
            result = fn()
            summary[name] = result
            print(f"[{name}] OK")
        except Exception as exc:  # pragma: no cover - operational path
            summary[name] = {"status": "error", "error": str(exc)}
            print(f"[{name}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Summary ===")
    for source_name, info in summary.items():
        print(f"- {source_name}: {info}")


if __name__ == "__main__":
    main()
