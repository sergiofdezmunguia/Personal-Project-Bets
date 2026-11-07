"""Train a first betting model (Over 2.5 goals) with temporal validation.

Features are generated pre-match from rolling team form to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.config import FINAL_TEST_START, MODELS_DIR, RAW_DATA_DIR

ROLLING_WINDOW = 5
MIN_PRIOR_MATCHES = 3

FEATURE_COLS = [
    "home_xg_for_roll5",
    "home_xg_against_roll5",
    "home_points_roll5",
    "home_ppda_roll5",
    "home_deep_comp_roll5",
    "away_xg_for_roll5",
    "away_xg_against_roll5",
    "away_points_roll5",
    "away_ppda_roll5",
    "away_deep_comp_roll5",
    "roll_xg_for_diff",
    "roll_points_diff",
    "roll_ppda_diff",
]


@dataclass
class DatasetBundle:
    train_x: pd.DataFrame
    train_y: pd.Series
    valid_x: pd.DataFrame
    valid_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def load_raw_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_path = RAW_DATA_DIR / "understat_schedule.csv"
    team_stats_path = RAW_DATA_DIR / "understat_team_match_stats.csv"

    if not schedule_path.exists() or not team_stats_path.exists():
        raise FileNotFoundError(
            "Missing input CSVs. Run src/extract_pipeline.py or src/explore_understat.py first."
        )

    schedule = pd.read_csv(schedule_path)
    team_stats = pd.read_csv(team_stats_path)
    return schedule, team_stats


def prepare_base_table(schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    # Flatten index columns that were saved by pandas from a multi-index DataFrame.
    drop_cols = [c for c in ["league", "season", "game"] if c in schedule.columns]
    if drop_cols:
        schedule = schedule.drop(columns=drop_cols)

    drop_cols_stats = [c for c in ["league", "season", "game"] if c in team_stats.columns]
    if drop_cols_stats:
        team_stats = team_stats.drop(columns=drop_cols_stats)

    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    team_stats["date"] = pd.to_datetime(team_stats["date"], errors="coerce")

    key_cols = [
        "league_id",
        "season_id",
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    # Without has_data, xG is a placeholder 0 that would distort the form of both teams.
    model_df = schedule.loc[schedule["has_data"], key_cols].copy()

    model_df["total_goals"] = model_df["home_goals"] + model_df["away_goals"]
    model_df["target_over_2_5"] = (model_df["total_goals"] > 2.5).astype(int)

    # Keep only completed matches.
    model_df = model_df.dropna(subset=["date", "home_goals", "away_goals"])

    # Include tactical/team-level match stats for feature engineering.
    merged = model_df.merge(
        team_stats,
        on=["league_id", "season_id", "game_id", "date", "home_team", "away_team"],
        how="inner",
    )
    return merged


def build_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    home_view = pd.DataFrame(
        {
            "league_id": df["league_id"],
            "season_id": df["season_id"],
            "game_id": df["game_id"],
            "date": df["date"],
            "team": df["home_team"],
            "xg_for": df["home_xg"],
            "xg_against": df["away_xg"],
            "points": df["home_points"],
            "ppda": df["home_ppda"],
            "deep_comp": df["home_deep_completions"],
            "side": "home",
        }
    )

    away_view = pd.DataFrame(
        {
            "league_id": df["league_id"],
            "season_id": df["season_id"],
            "game_id": df["game_id"],
            "date": df["date"],
            "team": df["away_team"],
            "xg_for": df["away_xg"],
            "xg_against": df["home_xg"],
            "points": df["away_points"],
            "ppda": df["away_ppda"],
            "deep_comp": df["away_deep_completions"],
            "side": "away",
        }
    )

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    metrics = ["xg_for", "xg_against", "points", "ppda", "deep_comp"]
    for metric in metrics:
        # shift + rolling must both run inside each team, or a team's first matches
        # average the previous team's last ones.
        long_df[f"{metric}_roll5"] = long_df.groupby("team")[metric].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PRIOR_MATCHES).mean()
        )

    keep_cols = [
        "league_id",
        "season_id",
        "game_id",
        "side",
        "xg_for_roll5",
        "xg_against_roll5",
        "points_roll5",
        "ppda_roll5",
        "deep_comp_roll5",
    ]
    feat_long = long_df[keep_cols]

    home_feats = feat_long[feat_long["side"] == "home"].drop(columns=["side"])
    home_feats = home_feats.rename(
        columns={
            "xg_for_roll5": "home_xg_for_roll5",
            "xg_against_roll5": "home_xg_against_roll5",
            "points_roll5": "home_points_roll5",
            "ppda_roll5": "home_ppda_roll5",
            "deep_comp_roll5": "home_deep_comp_roll5",
        }
    )

    away_feats = feat_long[feat_long["side"] == "away"].drop(columns=["side"])
    away_feats = away_feats.rename(
        columns={
            "xg_for_roll5": "away_xg_for_roll5",
            "xg_against_roll5": "away_xg_against_roll5",
            "points_roll5": "away_points_roll5",
            "ppda_roll5": "away_ppda_roll5",
            "deep_comp_roll5": "away_deep_comp_roll5",
        }
    )

    out = df.merge(home_feats, on=["league_id", "season_id", "game_id"], how="left")
    out = out.merge(away_feats, on=["league_id", "season_id", "game_id"], how="left")

    out["roll_xg_for_diff"] = out["home_xg_for_roll5"] - out["away_xg_for_roll5"]
    out["roll_points_diff"] = out["home_points_roll5"] - out["away_points_roll5"]
    out["roll_ppda_diff"] = out["home_ppda_roll5"] - out["away_ppda_roll5"]

    return out


def reject_final_test_data(df: pd.DataFrame) -> None:
    if (df["date"] >= FINAL_TEST_START).any():
        raise ValueError(
            f"Data from {FINAL_TEST_START.date()} onwards is the final test set: "
            "it stays untouched until the model is frozen."
        )


def split_by_date(
    df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by calendar day, so all the matches of a day fall on the same side.

    Fractions are approximate: they round down to the last whole day that fits.
    """
    reject_final_test_data(df)
    match_day = df["date"].dt.normalize()
    share = match_day.value_counts().sort_index().cumsum() / len(df)

    train_end = share[share <= train_frac].index.max()
    valid_end = share[share <= train_frac + valid_frac].index.max()

    train = df[match_day <= train_end]
    valid = df[(match_day > train_end) & (match_day <= valid_end)]
    test = df[match_day > valid_end]

    if train.empty or valid.empty or test.empty:
        raise ValueError("Not enough match days to build train, valid and test splits.")
    return train, valid, test


def temporal_split(df: pd.DataFrame) -> DatasetBundle:
    """Build the development splits. Their `test` is not the final test season."""
    # Check before dropna: a final-test row with missing features must also stop the run.
    reject_final_test_data(df)

    data = df.dropna(subset=FEATURE_COLS + ["target_over_2_5"]).sort_values("date")
    train, valid, test = split_by_date(data)

    return DatasetBundle(
        train_x=train[FEATURE_COLS],
        train_y=train["target_over_2_5"],
        valid_x=valid[FEATURE_COLS],
        valid_y=valid["target_over_2_5"],
        test_x=test[FEATURE_COLS],
        test_y=test["target_over_2_5"],
    )


def train_model(bundle: DatasetBundle):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        # All cores took 53 s vs 0.04 s on a 22-core WSL, with identical predictions.
        n_jobs=4,
    )
    model.fit(bundle.train_x, bundle.train_y)
    return model


def evaluate(model, x: pd.DataFrame, y: pd.Series, split_name: str) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
    }
    print(f"[{split_name}] {metrics}")
    return metrics


def main() -> None:
    schedule, team_stats = load_raw_inputs()
    base_df = prepare_base_table(schedule, team_stats)
    feat_df = build_team_form_features(base_df)
    bundle = temporal_split(feat_df)

    model = train_model(bundle)

    metrics = {
        "valid": evaluate(model, bundle.valid_x, bundle.valid_y, "valid"),
        "test": evaluate(model, bundle.test_x, bundle.test_y, "test"),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_over25.joblib"
    metrics_path = MODELS_DIR / "xgboost_over25_metrics.json"

    joblib.dump(model, model_path)
    pd.Series(metrics).to_json(metrics_path, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
