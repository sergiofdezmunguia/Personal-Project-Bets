"""End-to-end run on synthetic CSVs: same inputs, same results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import extract_pipeline, train_xgboost


def run_training(monkeypatch, league, workdir: Path) -> dict:
    raw, models = workdir / "raw", workdir / "models"
    raw.mkdir(parents=True)
    schedule, stats = league
    schedule.to_csv(raw / "understat_schedule.csv", index=False)
    stats.to_csv(raw / "understat_team_match_stats.csv", index=False)

    monkeypatch.setattr(train_xgboost, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(train_xgboost, "MODELS_DIR", models)
    train_xgboost.main()

    assert (models / "xgboost_over25.joblib").exists()
    return json.loads((models / "xgboost_over25_metrics.json").read_text())


def test_training_is_reproducible(monkeypatch, league, tmp_path):
    first = run_training(monkeypatch, league, tmp_path / "first")
    second = run_training(monkeypatch, league, tmp_path / "second")
    assert first == second
    assert set(first) == {"valid", "test"}


def test_model_fits_only_on_train(monkeypatch, league):
    bundle = train_xgboost.temporal_split(
        train_xgboost.build_team_form_features(train_xgboost.prepare_base_table(*league))
    )
    seen = {}

    class FakeModel:
        def __init__(self, **_):
            pass

        def fit(self, x, y):
            seen["x"], seen["y"] = x, y

    monkeypatch.setattr(train_xgboost, "XGBClassifier", FakeModel)
    train_xgboost.train_model(bundle)

    pd.testing.assert_frame_equal(seen["x"], bundle.train_x)
    pd.testing.assert_series_equal(seen["y"], bundle.train_y)


def test_each_metric_is_computed_on_its_own_split(monkeypatch, league, tmp_path):
    real_split, real_evaluate = train_xgboost.temporal_split, train_xgboost.evaluate
    bundles, scored = [], {}

    def spy_split(df):
        bundles.append(real_split(df))
        return bundles[-1]

    def spy_evaluate(model, x, y, split_name):
        scored[split_name] = (x, y)
        return real_evaluate(model, x, y, split_name)

    monkeypatch.setattr(train_xgboost, "temporal_split", spy_split)
    monkeypatch.setattr(train_xgboost, "evaluate", spy_evaluate)
    run_training(monkeypatch, league, tmp_path)

    bundle = bundles[0]
    pd.testing.assert_frame_equal(scored["valid"][0], bundle.valid_x)
    pd.testing.assert_series_equal(scored["valid"][1], bundle.valid_y)
    pd.testing.assert_frame_equal(scored["test"][0], bundle.test_x)
    pd.testing.assert_series_equal(scored["test"][1], bundle.test_y)


class FakeUnderstat:
    """Stands in for soccerdata; `tables` is what the source answers."""

    tables: dict = {}

    def __init__(self, leagues, seasons):
        pass

    def read_schedule(self):
        return self.tables["schedule"]

    def read_team_match_stats(self):
        return self.tables["team_stats"]

    def read_player_season_stats(self):
        return pd.DataFrame({"player": ["x"]})


def understat_table(league_seasons: list[tuple[str, str]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(league, season, f"game {i}") for i, (league, season) in enumerate(league_seasons)],
        names=["league", "season", "game"],
    )
    return pd.DataFrame({"game_id": range(len(league_seasons))}, index=index)


@pytest.mark.parametrize(
    "bad_table",
    [
        # What soccerdata returns when the source has none of the requested seasons.
        pd.DataFrame(index=["league", "season", "game"]),
        # One requested league-season is missing.
        understat_table([("A", "2324")]),
    ],
)
def test_bad_understat_response_does_not_replace_csvs(monkeypatch, tmp_path, bad_table):
    previous = tmp_path / "understat_schedule.csv"
    previous.write_text("previous data")
    monkeypatch.setattr(extract_pipeline, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(extract_pipeline.sd, "Understat", FakeUnderstat)
    monkeypatch.setattr(FakeUnderstat, "tables", {"schedule": bad_table, "team_stats": bad_table})

    with pytest.raises(ValueError, match="Understat"):
        extract_pipeline.extract_understat(["A", "B"], ["2324"])

    assert previous.read_text() == "previous data"


def test_complete_understat_response_is_saved(monkeypatch, tmp_path):
    table = understat_table([("A", "2324"), ("B", "2324")])
    monkeypatch.setattr(extract_pipeline, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(extract_pipeline.sd, "Understat", FakeUnderstat)
    monkeypatch.setattr(FakeUnderstat, "tables", {"schedule": table, "team_stats": table})

    result = extract_pipeline.extract_understat(["A", "B"], ["2324"])

    assert result["status"] == "ok"
    assert (tmp_path / "understat_schedule.csv").exists()


def test_training_fails_clearly_without_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(train_xgboost, "RAW_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        train_xgboost.main()


def test_extraction_exits_with_error_when_understat_fails(monkeypatch):
    def down(**_):
        raise ValueError("maintenance page")

    monkeypatch.setattr(extract_pipeline, "extract_understat", down)
    monkeypatch.setattr(extract_pipeline, "extract_clubelo", lambda **_: {"status": "ok"})
    with pytest.raises(SystemExit) as exit_info:
        extract_pipeline.main()
    assert exit_info.value.code == 1


def test_extraction_tolerates_clubelo_failure(monkeypatch):
    def down(**_):
        raise ValueError("502")

    monkeypatch.setattr(extract_pipeline, "extract_understat", lambda **_: {"status": "ok"})
    monkeypatch.setattr(extract_pipeline, "extract_clubelo", down)
    extract_pipeline.main()
