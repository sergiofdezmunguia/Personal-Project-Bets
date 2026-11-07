"""Which matches enter the base table, and how the target is defined."""

from __future__ import annotations

import numpy as np

from src.train_xgboost import prepare_base_table


def test_target_is_three_or_more_goals(league):
    schedule, stats = league
    schedule, stats = schedule.head(3).copy(), stats.head(3).copy()
    schedule["home_goals"], schedule["away_goals"] = [1, 2, 0], [1, 1, 0]

    base = prepare_base_table(schedule, stats).sort_values("game_id")

    assert base["target_over_2_5"].tolist() == [0, 1, 0]


def test_match_without_result_is_excluded(league):
    schedule, stats = league
    game_id = schedule["game_id"].iloc[5]
    schedule = schedule.astype({"home_goals": float})
    schedule.loc[schedule["game_id"] == game_id, "home_goals"] = np.nan

    assert game_id not in set(prepare_base_table(schedule, stats)["game_id"])


def test_match_without_team_stats_is_excluded(league):
    schedule, stats = league
    game_id = schedule["game_id"].iloc[5]
    stats = stats[stats["game_id"] != game_id]

    assert game_id not in set(prepare_base_table(schedule, stats)["game_id"])


def test_match_without_xg_data_is_excluded(league):
    schedule, stats = league
    game_id = schedule["game_id"].iloc[5]
    schedule.loc[schedule["game_id"] == game_id, "has_data"] = False

    assert game_id not in set(prepare_base_table(schedule, stats)["game_id"])
