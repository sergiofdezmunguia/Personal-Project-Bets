"""Splits are by match date and never reach the final test season."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import FINAL_TEST_START
from src.train_xgboost import (
    FEATURE_COLS,
    build_team_form_features,
    prepare_base_table,
    split_by_date,
    temporal_split,
)


def matches_on(days: list[str], per_day: int = 10) -> pd.DataFrame:
    """Several kick-off times per day, so a cut by rows would split a day."""
    rows = [
        {"game_id": f"{day}-{i}", "date": pd.Timestamp(day) + pd.Timedelta(hours=12 + i)}
        for day in days
        for i in range(per_day)
    ]
    return pd.DataFrame(rows)


def days_of(df: pd.DataFrame) -> set:
    return set(df["date"].dt.normalize())


def test_no_match_day_falls_on_two_sides():
    # 11 days of 10 matches: the 70 % and 85 % row cuts land in the middle of a day.
    days = [str(d.date()) for d in pd.date_range("2024-01-01", periods=11)]
    train, valid, test = split_by_date(matches_on(days))
    assert not days_of(train) & days_of(valid)
    assert not days_of(valid) & days_of(test)
    assert not days_of(train) & days_of(test)


def test_splits_are_in_time_order_and_keep_every_match(league):
    df = build_team_form_features(prepare_base_table(*league))
    train, valid, test = split_by_date(df)
    assert train["date"].max().normalize() < valid["date"].min().normalize()
    assert valid["date"].max().normalize() < test["date"].min().normalize()
    kept = pd.concat([train, valid, test])["game_id"]
    assert sorted(kept) == sorted(df["game_id"])


def test_split_sizes_are_close_to_the_requested_fractions():
    days = [str(d.date()) for d in pd.date_range("2024-01-01", periods=100)]
    train, valid, test = split_by_date(matches_on(days))
    assert (len(train), len(valid), len(test)) == (700, 150, 150)


def test_split_fails_when_there_are_too_few_days():
    with pytest.raises(ValueError, match="match days"):
        split_by_date(matches_on(["2024-01-01"]))


def test_temporal_split_uses_only_model_features(league):
    df = build_team_form_features(prepare_base_table(*league))
    bundle = temporal_split(df)
    assert list(bundle.train_x.columns) == FEATURE_COLS
    assert bundle.train_x.notna().all().all()


@pytest.mark.parametrize(
    ("offset", "rejected"),
    [(pd.Timedelta(hours=-1), False), (pd.Timedelta(0), True), (pd.Timedelta(days=400), True)],
)
def test_final_test_season_cannot_enter_development(league, offset, rejected):
    df = build_team_form_features(prepare_base_table(*league))
    df.loc[df.index[-1], "date"] = FINAL_TEST_START + offset
    if rejected:
        with pytest.raises(ValueError, match="final test"):
            temporal_split(df)
    else:
        temporal_split(df)


def test_final_test_row_with_missing_features_is_also_rejected(league):
    df = build_team_form_features(prepare_base_table(*league))
    last = df.index[-1]
    df.loc[last, "date"] = FINAL_TEST_START
    df.loc[last, FEATURE_COLS] = float("nan")
    with pytest.raises(ValueError, match="final test"):
        temporal_split(df)


def test_split_by_date_rejects_the_final_test_season_on_its_own():
    with pytest.raises(ValueError, match="final test"):
        split_by_date(matches_on(["2024-01-01", "2024-01-02", "2025-08-15"]))
