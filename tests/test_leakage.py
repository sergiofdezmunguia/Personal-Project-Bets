"""A match's features may only use matches played before it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.train_xgboost import (
    FEATURE_COLS,
    MIN_PRIOR_MATCHES,
    ROLLING_WINDOW,
    build_team_form_features,
    prepare_base_table,
)

# Metric -> (column when the team plays at home, column when it plays away).
METRICS = {
    "xg_for": ("home_xg", "away_xg"),
    "xg_against": ("away_xg", "home_xg"),
    "points": ("home_points", "away_points"),
    "ppda": ("home_ppda", "away_ppda"),
    "deep_comp": ("home_deep_completions", "away_deep_completions"),
}


@pytest.fixture
def base(league) -> pd.DataFrame:
    return prepare_base_table(*league)


def expected_form(base: pd.DataFrame) -> dict[tuple[int, str, str], float]:
    """Form per (game_id, side, metric), written without pandas groupby on purpose."""
    history: dict[str, list[dict[str, float]]] = {}
    expected = {}
    for row in base.sort_values("date").to_dict("records"):
        for side, team in [("home", row["home_team"]), ("away", row["away_team"])]:
            past = history.setdefault(team, [])
            for metric in METRICS:
                recent = [match[metric] for match in past[-ROLLING_WINDOW:]]
                enough = len(recent) >= MIN_PRIOR_MATCHES
                expected[row["game_id"], side, metric] = np.mean(recent) if enough else np.nan
        for team, column_index in [(row["home_team"], 0), (row["away_team"], 1)]:
            history[team].append({m: row[cols[column_index]] for m, cols in METRICS.items()})
    return expected


def assert_same(actual: float, expected: float, label) -> None:
    if np.isnan(expected):
        assert pd.isna(actual), label
    else:
        assert actual == pytest.approx(expected), label


def test_first_matches_of_every_team_have_no_form(base):
    """Regression: an ungrouped rolling filled these from the previous team."""
    feats = build_team_form_features(base).sort_values("date")
    seen: dict[str, int] = {}
    for row in feats.to_dict("records"):
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            if seen.get(team, 0) < MIN_PRIOR_MATCHES:
                assert pd.isna(row[f"{side}_xg_for_roll5"]), team
            seen[team] = seen.get(team, 0) + 1


@pytest.mark.parametrize("side", ["home", "away"])
@pytest.mark.parametrize("metric", METRICS)
def test_rolling_matches_an_independent_calculation(base, metric, side):
    expected = expected_form(base)
    for row in build_team_form_features(base).to_dict("records"):
        key = (row["game_id"], side, metric)
        assert_same(row[f"{side}_{metric}_roll5"], expected[key], key)


@pytest.mark.parametrize(
    ("column", "metric"),
    [("roll_xg_for_diff", "xg_for"), ("roll_points_diff", "points"), ("roll_ppda_diff", "ppda")],
)
def test_differences_are_home_minus_away(base, column, metric):
    expected = expected_form(base)
    for row in build_team_form_features(base).to_dict("records"):
        home = expected[row["game_id"], "home", metric]
        away = expected[row["game_id"], "away", metric]
        assert_same(row[column], home - away, (row["game_id"], column))


def test_model_features_are_only_prior_form():
    """Target, goals and same-match stats must never enter the feature list."""
    assert all(col.endswith("_roll5") or col.startswith("roll_") for col in FEATURE_COLS)
    assert len(set(FEATURE_COLS)) == len(FEATURE_COLS)


def test_order_follows_date_not_game_id(base):
    """Control: the fixture must contain a postponed match, or the tests above prove less."""
    assert not base.sort_values("date")["game_id"].is_monotonic_increasing


def test_changing_a_match_only_affects_later_matches(base):
    target = base.sort_values("date").iloc[len(base) // 2]
    changed = base.copy()
    mask = changed["game_id"] == target["game_id"]
    for columns in METRICS.values():
        changed.loc[mask, list(columns)] = 999

    before = build_team_form_features(base).set_index("game_id").sort_index()
    after = build_team_form_features(changed).set_index("game_id").sort_index()

    not_later = before.index[before["date"] <= target["date"]]
    pd.testing.assert_frame_equal(
        before.loc[not_later, FEATURE_COLS], after.loc[not_later, FEATURE_COLS]
    )

    # Control: the perturbation must reach the future, or this test proves nothing.
    later = before.index.difference(not_later)
    assert not before.loc[later, FEATURE_COLS].equals(after.loc[later, FEATURE_COLS])


def test_features_do_not_change_when_the_future_is_removed(base):
    cutoff = base["date"].sort_values().iloc[len(base) // 2]
    full = build_team_form_features(base).set_index("game_id")
    past_only = build_team_form_features(base[base["date"] <= cutoff]).set_index("game_id")
    pd.testing.assert_frame_equal(
        full.loc[past_only.index, FEATURE_COLS], past_only[FEATURE_COLS]
    )


def test_features_do_not_depend_on_row_order(base):
    expected = build_team_form_features(base).set_index("game_id").sort_index()[FEATURE_COLS]
    shuffled = base.sample(frac=1, random_state=1).reset_index(drop=True)
    actual = build_team_form_features(shuffled).set_index("game_id").sort_index()[FEATURE_COLS]
    pd.testing.assert_frame_equal(expected, actual)
